"""The managed `llama-server` process: start, stop, switch models, health, logs, cleanup.

State machine (docs/API.md section 9, GET /v1/engine `managed.state`):

    not_installed   no engine build is installed
    stopped         installed, not running
    starting        process launched, its HTTP server is not answering yet
    loading_model   HTTP is up, /health says the model is still loading
    ready           /health is OK; requests are served
    stopping        being stopped
    error           the last start failed or the engine died (`last_error`)

Concurrency: `lease(model_id)` is how requests use the engine. It starts the
engine (or switches it to `model_id`) when needed, serialized under one
condition variable so concurrent requests can never start two engines. A
switch waits until requests on the current model finish; new requests for the
old model wait behind a pending switch. Requests for the loaded model run
concurrently up to the engine's parallel slots (a semaphore of that size).

Cleanup guarantees (see vvl.llama.procutil): PDEATHSIG on Linux, a lifeline
guardian on macOS, a kill-on-close Job object on Windows; SIGTERM then
SIGKILL escalation on stop; stop on VVL shutdown; and a pidfile under
VVL_HOME (pid + start time + command) so the next VVL start reaps an engine a
crashed VVL left behind without touching an unrelated process.

---------------------------------------------------------------------------
llama-server flags relied on (checked against `llama-server --help` of the
pinned build b12359; re-check these when the engine lock moves):

    -m / --model PATH          language model GGUF
    --mmproj PATH              vision projector GGUF
    --host 127.0.0.1 --port N  loopback only, port chosen by VVL
    --api-key                  set through the LLAMA_API_KEY environment
                               variable, never on the command line (so it is
                               not visible in `ps` or the pidfile)
    -a / --alias NAME          model name reported by /v1/models
    -c / --ctx-size N          total context; VVL passes context_size x parallel
    -np / --parallel N         request slots
    --kv-unified / --no-kv-unified   VVL passes --no-kv-unified so each slot
                               gets exactly context_size tokens
    -ngl / --n-gpu-layers N    0 for CPU builds; otherwise left to --fit
    -fit / --fit on            adjusts only unset arguments: with --ctx-size set
                               it decides how many layers fit on the GPU(s)
    --cors-origins localhost   no cross-origin browser access to the engine
    --log-colors off           plain log lines for the ring buffer
    --image-min-tokens / --image-max-tokens N   vision token bounds per image
    --jinja                    use the model's chat template
    --no-webui                 the bundled web page is not served
    --timeout N                server read/write timeout (seconds)
    --list-devices / --version used by the installer's self-test
    GET /health                503 while loading, 200 {"status":"ok"} when ready
    POST /v1/chat/completions  OpenAI-compatible; response_format json_schema
Device selection: CUDA_VISIBLE_DEVICES (with CUDA_DEVICE_ORDER=PCI_BUS_ID so
indexes match nvidia-smi and /v1/system), GGML_VK_VISIBLE_DEVICES for Vulkan.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import collections
import contextlib
import datetime
import json
import logging
import logging.handlers
import os
import re
import secrets
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from vvl import paths
from vvl.fileutil import atomic_write_text
from vvl.llama import inference as inference_mod
from vvl.llama import procutil
from vvl.llama.installer import classify_incompatible, engine_env
from vvl.local_models import file_path, is_installed

logger = logging.getLogger(__name__)

STATES = ("not_installed", "stopped", "starting", "loading_model", "ready", "stopping", "error")
LOAD_TIMEOUT_SECONDS = 900
STOP_TERM_TIMEOUT_SECONDS = 10.0
HEALTH_POLL_SECONDS = 0.25
IDLE_CHECK_SECONDS = 5.0
PORT_RANGE = (8660, 8760)
MAX_PORT_ATTEMPTS = 3
LOG_RING_LINES = 5000
LOG_FILE_BYTES = 5 * 1024 * 1024
LOG_FILE_BACKUPS = 3
PIDFILE_NAME = "engine.pid.json"
DEFAULT_RUNTIME_MARGIN = 1536 * 1024 * 1024
DEFAULT_PARALLEL = 2          # request slots per engine when nothing else decides

# Ports handed out in the last minute, so two engines starting together do not
# both pick the same free port before either has bound it.
_claimed_ports: dict[int, float] = {}
_claimed_ports_lock = threading.Lock()
MiB = 1024 * 1024

_PORT_IN_USE = re.compile(r"(address already in use|couldn't bind|failed to bind|bind\(\) failed|"
                          r"unable to bind|port .* in use)", re.I)
_OOM = re.compile(r"(out of memory|cudaMalloc failed|failed to allocate|ErrorOutOfDeviceMemory|"
                  r"unable to allocate|not enough memory|insufficient memory)", re.I)


def _now_iso(ts: float | None = None) -> str:
    moment = datetime.datetime.fromtimestamp(ts if ts is not None else time.time(), datetime.timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


class SupervisorError(Exception):
    """A start/stop request the supervisor refuses (rendered by the routes)."""

    def __init__(self, code: str, message: str, *, status: int = 409, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = details or {}


class EngineNotReady(Exception):
    """The engine could not be made ready for a request (HTTP 503 ENGINE_NOT_READY)."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


@dataclass(frozen=True)
class EngineHandle:
    """What a request needs to talk to the running engine."""

    base_url: str          # http://127.0.0.1:<port>/v1
    api_key: str
    model_name: str        # the alias the engine reports
    model_id: str          # local:<build_id>
    settings: dict         # merged inference settings for the build
    context_size: int      # per-request context window


# ------------------------------------------------------------------- logs --

class LogBuffer:
    """Ring buffer of engine output plus a rotating log file."""

    def __init__(self, size: int = LOG_RING_LINES):
        self._lines: collections.deque = collections.deque(maxlen=size)
        self._seq = 0
        self._lock = threading.Lock()
        self._secrets: list[str] = []
        self._file_logger: logging.Logger | None = None

    def set_secret(self, value: str | None) -> None:
        """Redact `value` from everything logged from now on (several engines may
        run at once, each with its own key, so recent secrets are all kept)."""
        if not value:
            return
        with self._lock:
            self._secrets = ([s for s in self._secrets if s != value] + [value])[-32:]

    def _file(self) -> logging.Logger | None:
        if self._file_logger is not None:
            return self._file_logger
        try:
            log_dir = paths.ensure_dir(paths.logs_dir())
            handler = logging.handlers.RotatingFileHandler(
                log_dir / "engine.log", maxBytes=LOG_FILE_BYTES, backupCount=LOG_FILE_BACKUPS, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            file_logger = logging.getLogger(f"vvl.engine.output.{id(self)}")
            file_logger.propagate = False
            file_logger.setLevel(logging.INFO)
            file_logger.handlers[:] = [handler]
            self._file_logger = file_logger
        except OSError:
            self._file_logger = None
        return self._file_logger

    def append(self, text: str, unit: int | None = None) -> None:
        text = text.rstrip("\r\n")
        with self._lock:
            for secret in self._secrets:
                if secret and secret in text:
                    text = text.replace(secret, "[redacted]")
            self._seq += 1
            self._lines.append({"seq": self._seq, "ts": _now_iso(), "text": text, "unit": unit})
        file_logger = self._file()
        if file_logger is not None:
            try:
                file_logger.info("%s", text)
            except Exception:  # noqa: BLE001
                pass

    def read(self, after: int | None = None, limit: int = 200) -> dict:
        """Lines with seq >= after (a cursor); without `after`, the last `limit` lines.

        `next` is the cursor for the following call."""
        with self._lock:
            lines = list(self._lines)
            next_seq = self._seq + 1
        if after is None:
            chosen = lines[-limit:] if limit else []
        else:
            chosen = [line for line in lines if line["seq"] >= after][:limit]
        if chosen:
            nxt = chosen[-1]["seq"] + 1
        elif after is None or after > next_seq:
            nxt = next_seq
        else:
            nxt = after
        return {"lines": [{"seq": line["seq"], "ts": line["ts"], "text": line["text"]} for line in chosen],
                "next": nxt}

    def close(self) -> None:
        file_logger, self._file_logger = self._file_logger, None
        if file_logger is not None:
            for handler in list(file_logger.handlers):
                try:
                    handler.close()
                except Exception:  # noqa: BLE001
                    pass
                file_logger.removeHandler(handler)

    def tail(self, count: int = 20, since_seq: int = 0, unit: int | None = None) -> list[str]:
        """The last lines, optionally only those of one engine (`unit`)."""
        with self._lock:
            lines = [line["text"] for line in self._lines
                     if line["seq"] >= since_seq and (unit is None or line.get("unit") == unit)]
        return lines[-count:]

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._seq


# ---------------------------------------------------------------- process --

class _EngineProcess:
    def __init__(self, popen, argv, port, api_key, model_id, alias, attempt, settings, context_size,
                 parallel, gpu_devices, backend, variant, first_log_seq):
        self.popen = popen
        self.argv = argv
        self.port = port
        self.api_key = api_key
        self.model_id = model_id
        self.alias = alias
        self.attempt = attempt
        self.settings = settings
        self.context_size = context_size
        self.parallel = parallel
        self.gpu_devices = gpu_devices
        self.backend = backend
        self.variant = variant
        self.first_log_seq = first_log_seq
        self.stop_requested = False
        self.exited = threading.Event()
        self.exit_code: int | None = None
        self.started_at = time.time()
        self.slots = threading.BoundedSemaphore(max(1, parallel))

    @property
    def pid(self) -> int:
        return self.popen.pid

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


class EngineSupervisor:
    def __init__(self, runtime, *, load_timeout: float | None = None, idle_check_seconds: float = IDLE_CHECK_SECONDS,
                 stop_timeout: float = STOP_TERM_TIMEOUT_SECONDS, port_range: tuple[int, int] | None = None,
                 unit: int = 0, logs: "LogBuffer | None" = None):
        self.rt = runtime
        # One supervisor runs one llama-server. The engine pool (pool.py) keeps
        # several of them; `unit` numbers them, and `overrides` carries the
        # pool's decisions for the next start (gpu_devices, parallel).
        self.unit = int(unit)
        self.overrides: dict = {}
        self.log_prefix = ""
        cfg = getattr(runtime, "config", {}) or {}
        self.load_timeout = float(load_timeout or cfg.get("VVL_ENGINE_LOAD_TIMEOUT") or LOAD_TIMEOUT_SECONDS)
        self.stop_timeout = float(cfg.get("VVL_ENGINE_STOP_TIMEOUT") or stop_timeout)
        self.idle_check_seconds = float(cfg.get("VVL_ENGINE_IDLE_CHECK_SECONDS") or idle_check_seconds)
        self.port_range = tuple(cfg.get("VVL_ENGINE_PORT_RANGE") or port_range or PORT_RANGE)
        self._owns_logs = logs is None
        self.logs = logs if logs is not None else LogBuffer()
        self._cond = threading.Condition(threading.RLock())
        self._state = "stopped"
        self._proc: _EngineProcess | None = None
        self._model_id: str | None = None
        self._attempt = 0
        self._leases = 0
        self._switch_to: str | None = None
        self._last_error: dict | None = None
        self._last_used = time.monotonic()
        self._launcher: procutil.Launcher | None = None
        self._background: threading.Thread | None = None
        self._closed = False
        self._last_report: dict | None = None

    # -- helpers -----------------------------------------------------------

    def _launcher_thread(self) -> procutil.Launcher:
        with self._cond:
            if self._launcher is None:
                self._launcher = procutil.Launcher()
            return self._launcher

    def _set(self, attempt: int, state: str, **fields) -> bool:
        """Change state if `attempt` is still current. Returns False when stale."""
        with self._cond:
            if attempt != self._attempt:
                return False
            self._state = state
            if "last_error" in fields:
                self._last_error = fields["last_error"]
            self._cond.notify_all()
            return True

    def _apply_generation(self, model_id, family, build, infer: dict, context_size: int, overrides: dict):
        """Context window and image limits from the user's choices (vvl/generation.py).

        The engine pool decides them (`overrides`); a supervisor used on its own
        applies the same rule, without the memory-based maximum.
        """
        if "context_size" in overrides:
            infer = dict(infer)
            infer.update(overrides.get("infer") or {})
            return infer, int(overrides["context_size"])
        try:
            from vvl import generation

            full = self.rt.settings()
            registry = self.rt.local_registry()
            infer = dict(infer)
            infer.update(generation.engine_params(model_id, full, registry))
            rule = generation.context_rule(model_id, full, registry,
                                           model_max=generation.model_max_context(self.rt.models_dir(), build),
                                           image_max_tokens=infer.get("image_max_tokens"))
            return infer, int(rule["tokens"] or rule["needed"])
        except Exception:  # noqa: BLE001 - fall back to the registry's values
            logger.exception("Model settings could not be applied; using the registry defaults.")
            return infer, context_size

    def _engine_settings(self) -> dict:
        try:
            return self.rt.engine_settings()
        except Exception:  # noqa: BLE001
            from vvl.settings import DEFAULTS

            return dict(DEFAULTS["engine"])

    def _install(self) -> dict | None:
        try:
            return self.rt.installer.active_install(self._engine_settings().get("variant", "auto"))
        except Exception:  # noqa: BLE001
            logger.exception("Could not read the engine install.")
            return None

    def _pidfile(self) -> Path:
        name = PIDFILE_NAME if self.unit == 0 else PIDFILE_NAME.replace(".pid.json", f"-{self.unit}.pid.json")
        return paths.home() / "run" / name

    def _log(self, text: str) -> None:
        self.logs.append(self.log_prefix + text if self.log_prefix else text, unit=self.unit)

    # -- public state --------------------------------------------------------

    def is_running(self) -> bool:
        with self._cond:
            return self._proc is not None and not self._proc.exited.is_set()

    def loaded_model(self) -> str | None:
        with self._cond:
            if self._state in ("starting", "loading_model", "ready") and self._proc is not None:
                return self._model_id
            return None

    def loaded_models(self) -> list[str]:
        model = self.loaded_model()
        return [model] if model else []

    def engine_pids(self) -> list[int]:
        with self._cond:
            proc = self._proc
            return [proc.pid] if proc is not None and not proc.exited.is_set() else []

    def state(self) -> str:
        with self._cond:
            state = self._state
            running = self._proc is not None and not self._proc.exited.is_set()
        if state in ("stopped", "error") and not running and self._install() is None:
            return "not_installed" if state == "stopped" else state
        return state

    def status(self) -> dict:
        install = self._install()
        settings_now = self._engine_settings()
        with self._cond:
            proc = self._proc
            running = proc is not None and not proc.exited.is_set()
            state = self._state
            if state == "stopped" and install is None:
                state = "not_installed"
            idle_stop_at = None
            idle_minutes = settings_now.get("idle_stop_minutes") or 0
            if state == "ready" and idle_minutes and self._leases == 0:
                remaining = self._last_used + idle_minutes * 60 - time.monotonic()
                idle_stop_at = _now_iso(time.time() + max(0.0, remaining))
            variant = None
            if proc is not None:
                variant = proc.variant
            elif install is not None:
                variant = {"id": install["variant"], "label": install.get("label"), "backend": install.get("backend")}
            return {
                "pinned_version": self.rt.engine_lock().release.tag,
                "installed": install is not None,
                "version": install.get("tag") if install else None,
                "build": install.get("version") if install else None,
                "variant": variant,
                "state": state,
                "running": running,
                "pid": proc.pid if running else None,
                "port": proc.port if running else None,
                "backend": (proc.backend if proc is not None else (install.get("backend") if install else None)),
                "loaded_model": self._model_id if running and state in ("starting", "loading_model", "ready") else None,
                "gpu_devices": proc.gpu_devices if running else None,
                "context_size": proc.context_size if running else None,
                "parallel": proc.parallel if running else None,
                "started_at": _now_iso(proc.started_at) if running else None,
                "last_error": dict(self._last_error) if self._last_error else None,
                "idle_stop_at": idle_stop_at,
                "active_requests": self._leases,
            }

    def snapshot(self) -> dict:
        """A cheap view of this engine for the pool (no settings or disk access)."""
        with self._cond:
            proc = self._proc
            running = proc is not None and not proc.exited.is_set()
            live = self._state in ("starting", "loading_model", "ready")
            return {
                "unit": self.unit,
                "state": self._state,
                "live": live,
                "running": running,
                "model_id": self._model_id if live else None,
                "leases": self._leases,
                "parallel": proc.parallel if running else None,
                "gpu_devices": proc.gpu_devices if running else None,
                "pid": proc.pid if running else None,
                "port": proc.port if running else None,
                "context_size": proc.context_size if running else None,
                "started_at": _now_iso(proc.started_at) if running else None,
                "last_error": dict(self._last_error) if self._last_error else None,
                "idle_for": time.monotonic() - self._last_used,
            }

    # -- start / stop ----------------------------------------------------------

    def _check_startable(self, model_id: str):
        from vvl.engines.resolver import parse_model_id

        ref = parse_model_id(model_id)
        if ref.source != "local":
            raise SupervisorError("INVALID_INPUT", "Only local:<build_id> models run in the managed engine.",
                                  status=400, details={"model_id": model_id})
        found = self.rt.local_registry().find_build(ref.name)
        if not found:
            raise SupervisorError("INVALID_INPUT", f"Unknown local model '{ref.name}'.", status=400,
                                  details={"model_id": model_id})
        family, build = found
        if self._install() is None:
            raise SupervisorError("ENGINE_NOT_INSTALLED", "Install the llama.cpp engine first (Engine tab).",
                                  status=409)
        if not is_installed(self.rt.models_dir(), build):
            raise SupervisorError("MODEL_NOT_AVAILABLE", f"{build.label} is not downloaded yet.", status=409,
                                  details={"model_id": model_id})
        return family, build

    def start(self, model_id: str) -> dict:
        """Start (or switch to) `model_id` in the background. Returns the status."""
        self._check_startable(model_id)
        with self._cond:
            if self._closed:
                raise SupervisorError("ENGINE_NOT_READY", "VoucherVisionLocal is shutting down.", status=503)
            if self._model_id == model_id and self._state in ("starting", "loading_model", "ready") \
                    and self._proc is not None and not self._proc.exited.is_set():
                return self.status()
            self._begin(model_id)
        return self.status()

    def _begin(self, model_id: str) -> int:
        """Under the lock: move to `starting` and launch in a worker thread."""
        self._attempt += 1
        attempt = self._attempt
        old = self._proc
        self._state = "starting"
        self._model_id = model_id
        self._last_error = None
        self._switch_to = None
        self._cond.notify_all()
        threading.Thread(target=self._launch, args=(attempt, model_id, old), name=f"vvl-engine-start-{attempt}",
                         daemon=True).start()
        return attempt

    def stop(self) -> dict:
        with self._cond:
            self._attempt += 1
            attempt = self._attempt
            proc = self._proc
            self._switch_to = None
            if proc is None or proc.exited.is_set():
                self._proc = None
                self._state = "stopped"
                self._model_id = None
                self._cond.notify_all()
                return self.status()
            self._state = "stopping"
            self._cond.notify_all()
        self._terminate(proc)
        with self._cond:
            if attempt == self._attempt:
                self._proc = None
                self._state = "stopped"
                self._model_id = None
                self._cond.notify_all()
        return self.status()

    def _terminate(self, proc: _EngineProcess) -> None:
        proc.stop_requested = True
        self._log(f"[vvl] stopping llama-server (pid {proc.pid})")
        launcher = self._launcher
        how = procutil.terminate_tree(proc.popen, term_timeout=self.stop_timeout,
                                      windows_job=launcher.windows_job(proc.pid) if launcher else None)
        proc.exited.wait(5)
        if launcher is not None:
            launcher.release(proc.pid)
        self._remove_pidfile(proc.pid)
        self._log(f"[vvl] llama-server {how} (pid {proc.pid})")

    # -- launching --------------------------------------------------------------

    def _launch(self, attempt: int, model_id: str, old: _EngineProcess | None) -> None:
        try:
            if old is not None and not old.exited.is_set():
                with self._cond:
                    if attempt == self._attempt:
                        self._state = "stopping"
                        self._cond.notify_all()
                self._terminate(old)
                with self._cond:
                    if self._proc is old:
                        self._proc = None
                if not self._set(attempt, "starting"):
                    return
            family, build = self._check_startable(model_id)
            install = self._install()
            settings_now = self._engine_settings()
            infer, context_size, parallel, backend = launch_parameters(family, build, install, settings_now)
            overrides = dict(self.overrides)
            infer, context_size = self._apply_generation(model_id, family, build, infer, context_size, overrides)
            if overrides.get("parallel"):
                parallel = int(overrides["parallel"])
            if "gpu_devices" in overrides:
                gpu_devices = overrides["gpu_devices"]
                if backend not in ("cpu", "metal"):
                    self._last_report = self.rt.system_report(False) or {}
            else:
                gpu_devices = self._choose_gpus(backend, build, infer, context_size * parallel,
                                                settings_now.get("gpu_devices", "auto"))
            models_dir = self.rt.models_dir()
            model_file = next(f for f in build.files if f.role == "model")
            mmproj_file = next((f for f in build.files if f.role == "mmproj"), None)
            last_problem = None
            failed_ports: set[int] = set()
            for _try in range(MAX_PORT_ATTEMPTS):
                with self._cond:
                    if attempt != self._attempt:
                        return
                port = self._pick_port(failed_ports)
                api_key = secrets.token_urlsafe(32)
                alias = build.id
                argv = build_command(
                    install["server_path"], model=str(file_path(models_dir, build, model_file)),
                    mmproj=str(file_path(models_dir, build, mmproj_file)) if mmproj_file else None,
                    port=port, alias=alias, context_total=context_size * parallel, parallel=parallel,
                    backend=backend, infer=infer)
                env = engine_env(Path(install["bin_dir_path"]))
                env["LLAMA_API_KEY"] = api_key
                env.update(device_env(backend, gpu_devices, report=self._last_report,
                                      devices=install.get("devices")))
                self.logs.set_secret(api_key)
                first_seq = self.logs.last_seq + 1
                self._log(f"[vvl] starting {model_id} on port {port}: {' '.join(argv)}")
                if gpu_devices is not None:
                    self._log(f"[vvl] GPUs: {gpu_devices if gpu_devices else 'none (CPU)'}")
                popen = self._launcher_thread().spawn(argv, env=env, cwd=install["bin_dir_path"])
                variant = {"id": install["variant"], "label": install.get("label"), "backend": backend}
                proc = _EngineProcess(popen, argv, port, api_key, model_id, alias, attempt, infer, context_size,
                                      parallel, gpu_devices, backend, variant, first_seq)
                with self._cond:
                    if attempt != self._attempt:
                        stale = True
                    else:
                        stale = False
                        self._proc = proc
                if stale:
                    self._terminate(proc)
                    return
                self._write_pidfile(proc)
                threading.Thread(target=self._pump, args=(proc,), name=f"vvl-engine-log-{proc.pid}",
                                 daemon=True).start()
                outcome = self._wait_ready(proc)
                if outcome == "ready":
                    with self._cond:
                        if attempt == self._attempt:
                            self._state = "ready"
                            self._last_used = time.monotonic()
                            self._cond.notify_all()
                    self._log(f"[vvl] {model_id} is ready ({time.time() - proc.started_at:.1f} s)")
                    return
                if outcome == "stale":
                    return
                tail = "\n".join(self.logs.tail(60, proc.first_log_seq, unit=self.unit))
                if outcome == "exited" and _PORT_IN_USE.search(tail):
                    last_problem = "port in use"
                    failed_ports.add(port)
                    self._log(f"[vvl] port {port} was taken; trying another port")
                    with self._cond:
                        if self._proc is proc:
                            self._proc = None
                    continue
                self._fail(attempt, proc, outcome, tail)
                return
            self._fail(attempt, None, "exited", f"could not find a free port ({last_problem})")
        except SupervisorError as exc:
            self._set(attempt, "error", last_error={"code": exc.code, "message": exc.message, "log_tail": [],
                                                    "at": _now_iso()})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Engine start failed.")
            self._set(attempt, "error", last_error={"code": "ENGINE_ERROR",
                                                    "message": f"The engine could not be started ({type(exc).__name__}: {exc}).",
                                                    "log_tail": self.logs.tail(20, unit=self.unit), "at": _now_iso()})

    def _wait_ready(self, proc: _EngineProcess) -> str:
        """ready | exited | timeout | stale"""
        deadline = time.monotonic() + self.load_timeout
        url = proc.base_url + "/health"
        headers = {"Authorization": f"Bearer {proc.api_key}"}
        with httpx.Client(timeout=httpx.Timeout(2.0, connect=1.0)) as client:
            while True:
                with self._cond:
                    if proc.attempt != self._attempt:
                        return "stale"
                if proc.exited.is_set() or proc.popen.poll() is not None:
                    proc.exited.wait(2)
                    return "exited"
                if time.monotonic() > deadline:
                    return "timeout"
                try:
                    response = client.get(url, headers=headers)
                    if response.status_code == 200:
                        return "ready"
                    if response.status_code == 503:
                        with self._cond:
                            if proc.attempt == self._attempt and self._state == "starting":
                                self._state = "loading_model"
                                self._cond.notify_all()
                except httpx.HTTPError:
                    pass
                time.sleep(HEALTH_POLL_SECONDS)

    def _fail(self, attempt: int, proc: _EngineProcess | None, outcome: str, tail: str) -> None:
        if proc is not None and not proc.exited.is_set():
            self._terminate(proc)
        code = proc.exit_code if proc is not None else None
        message = explain_failure(outcome, tail, code, self.load_timeout)
        incompatible = classify_incompatible(tail)
        if incompatible and proc is not None:
            try:
                self.rt.installer.record_failure(proc.variant["id"], incompatible)
                message += " This build was marked as not working here; install the recommended build again."
            except Exception:  # noqa: BLE001
                pass
        lines = [line for line in tail.splitlines() if line.strip()][-20:]
        with self._cond:
            if attempt == self._attempt:
                if self._proc is proc:
                    self._proc = None
                self._state = "error"
                self._last_error = {"code": "ENGINE_FAILED" if outcome != "timeout" else "ENGINE_TIMEOUT",
                                    "message": message, "log_tail": lines, "at": _now_iso(),
                                    "exit_code": code}
                self._cond.notify_all()
        logger.warning("Engine failed: %s", message)

    def _pump(self, proc: _EngineProcess) -> None:
        """Read the engine's output until EOF, then record its exit."""
        stream = proc.popen.stdout
        try:
            for raw in iter(stream.readline, b""):
                self._log(raw.decode("utf-8", "replace"))
        except (OSError, ValueError):
            pass
        finally:
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            proc.exit_code = proc.popen.wait(30)
        except Exception:  # noqa: BLE001
            proc.exit_code = proc.popen.poll()
        proc.exited.set()
        self._remove_pidfile(proc.pid)
        if not proc.stop_requested:
            self._log(f"[vvl] llama-server exited with code {proc.exit_code}")
            with self._cond:
                current = self._proc is proc
                was_ready = self._state == "ready"
            if current and was_ready:
                tail = "\n".join(self.logs.tail(60, proc.first_log_seq, unit=self.unit))
                self._fail(proc.attempt, proc, "crashed", tail)
        if self._launcher is not None:
            self._launcher.release(proc.pid)

    def _pick_port(self, exclude: set[int] | None = None) -> int:
        low, high = self.port_range
        now = time.monotonic()
        with _claimed_ports_lock:
            for port, until in list(_claimed_ports.items()):
                if until < now:
                    _claimed_ports.pop(port, None)
            for port in range(int(low), int(high)):
                if (exclude and port in exclude) or port in _claimed_ports:
                    continue
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    try:
                        sock.bind(("127.0.0.1", port))
                    except OSError:
                        continue
                _claimed_ports[port] = now + 60.0
                return port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    # -- GPU choice ---------------------------------------------------------------

    def _choose_gpus(self, backend: str, build, infer: dict, context_total: int, setting) -> list[int] | None:
        """GPU indexes (as in /v1/system) for the engine, [] for CPU, None = engine default."""
        if backend in ("cpu", "metal"):
            return choose_gpu_devices(None, backend, build, infer, context_total, setting)
        report = self.rt.system_report(True) or {}
        self._last_report = report
        return choose_gpu_devices(report, backend, build, infer, context_total, setting)

    def launch_preview(self, model_id: str, report: dict | None) -> dict | None:
        """What starting `model_id` would do right now, without starting anything.

        Uses the same settings, memory estimate and GPU choice as `_launch`, with
        `report` (a system report whose free-memory figures the caller keeps
        current) in place of a fresh hardware check. None when the model or the
        engine is not installed.
        """
        try:
            family, build = self._check_startable(model_id)
        except Exception:  # noqa: BLE001 - not startable (unknown, not downloaded, no engine)
            return None
        install = self._install()
        if install is None:
            return None
        settings_now = self._engine_settings()
        infer, context_size, parallel, backend = launch_parameters(family, build, install, settings_now)
        infer, context_size = self._apply_generation(model_id, family, build, infer, context_size, {})
        context_total = context_size * parallel
        return {
            "backend": backend,
            "context_total": context_total,
            "required_bytes": estimate_bytes(build, infer, context_total),
            "gpu_devices": choose_gpu_devices(report or {}, backend, build, infer, context_total,
                                              settings_now.get("gpu_devices", "auto")),
        }

    # -- leases (requests) ------------------------------------------------------

    @contextlib.contextmanager
    def lease(self, model_id: str, timeout: float | None = None):
        """Hold the engine for one request on `model_id`, starting/switching it if needed."""
        timeout = self.load_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        with self._cond:
            try:
                self._wait_for_lease(model_id, timeout, deadline)
            except BaseException:
                # A waiter that gives up must not leave its switch request behind.
                if self._switch_to == model_id:
                    self._switch_to = None
                    self._cond.notify_all()
                raise
        proc = self._proc
        handle = EngineHandle(base_url=proc.base_url + "/v1", api_key=proc.api_key, model_name=proc.alias,
                              model_id=model_id, settings=proc.settings, context_size=proc.context_size)
        acquired = False
        try:
            acquired = proc.slots.acquire(timeout=float(proc.settings.get("request_timeout_seconds") or 900))
            if not acquired:
                raise EngineNotReady("All engine slots stayed busy.")
            yield handle
        finally:
            if acquired:
                proc.slots.release()
            with self._cond:
                self._leases -= 1
                self._last_used = time.monotonic()
                self._cond.notify_all()

    def _wait_for_lease(self, model_id: str, timeout: float, deadline: float) -> None:
        """Under the lock: wait until the engine serves `model_id`, then count a lease."""
        my_attempt = None
        while True:
            if self._closed:
                raise EngineNotReady("VoucherVisionLocal is shutting down.")
            proc = self._proc
            alive = proc is not None and not proc.exited.is_set()
            state = self._state
            if state == "ready" and alive and self._model_id == model_id and self._switch_to in (None, model_id):
                self._leases += 1
                return
            if my_attempt is not None and self._attempt == my_attempt and state == "error":
                err = self._last_error or {}
                raise EngineNotReady(err.get("message") or "The engine could not be started.",
                                     {"last_error": err})
            if my_attempt is not None and self._attempt != my_attempt and state == "stopped":
                raise EngineNotReady("The engine was stopped while this request was waiting for it.")
            if state in ("starting", "loading_model", "stopping"):
                if self._model_id == model_id:
                    my_attempt = self._attempt
                # else: another model is starting; wait for it to settle.
            elif state == "ready" and alive and self._leases > 0:
                # Another model is serving requests: switch once they finish.
                if self._switch_to is None:
                    self._switch_to = model_id
            elif self._switch_to not in (None, model_id):
                pass  # another request's switch goes first
            else:
                self._check_startable_locked(model_id)
                my_attempt = self._begin(model_id)
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EngineNotReady(f"The engine did not become ready within {int(timeout)} seconds.",
                                     {"state": self._state})
            self._cond.wait(min(remaining, 1.0))

    def _check_startable_locked(self, model_id: str) -> None:
        try:
            self._check_startable(model_id)
        except SupervisorError as exc:
            raise EngineNotReady(exc.message, {"code": exc.code}) from None

    # -- background: idle stop ---------------------------------------------------

    def start_background(self) -> None:
        with self._cond:
            if self._background is not None:
                return
            self._background = threading.Thread(target=self._idle_loop, name="vvl-engine-idle", daemon=True)
            self._background.start()
        self._launcher_thread()

    def _idle_loop(self) -> None:
        while True:
            time.sleep(self.idle_check_seconds)
            with self._cond:
                if self._closed:
                    return
            self.check_idle()

    def check_idle(self) -> bool:
        minutes = self._engine_settings().get("idle_stop_minutes") or 0
        if not minutes:
            return False
        with self._cond:
            idle = (self._state == "ready" and self._leases == 0 and self._switch_to is None
                    and time.monotonic() - self._last_used >= minutes * 60)
        if idle:
            self._log(f"[vvl] no requests for {minutes} min; stopping the engine to free memory")
            self.stop()
            return True
        return False

    # -- pidfile and reaper ------------------------------------------------------

    def _write_pidfile(self, proc: _EngineProcess) -> None:
        identity = procutil.process_identity(proc.pid) or {}
        me = procutil.process_identity(os.getpid()) or {}
        record = {
            "pid": proc.pid,
            "create_time": identity.get("create_time"),
            "cmdline": identity.get("cmdline") or proc.argv,
            "port": proc.port,
            "model_id": proc.model_id,
            "vvl_pid": os.getpid(),
            "vvl_create_time": me.get("create_time"),
            "started_at": _now_iso(proc.started_at),
        }
        try:
            atomic_write_text(self._pidfile(), json.dumps(record, indent=1), mode=0o600)
        except OSError:
            logger.warning("Could not write the engine pidfile.")

    def _remove_pidfile(self, pid: int) -> None:
        path = self._pidfile()
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if record.get("pid") == pid:
            try:
                path.unlink()
            except OSError:
                pass

    def reap_stale(self) -> str:
        """Kill an engine a crashed VVL left behind. Returns what happened."""
        path = self._pidfile()
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return "none"
        except (OSError, ValueError):
            path.unlink(missing_ok=True)
            return "unreadable"
        pid = record.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            path.unlink(missing_ok=True)
            return "invalid"
        owner = record.get("vvl_pid")
        if isinstance(owner, int) and owner != os.getpid():
            owner_live = procutil.process_identity(owner)
            if owner_live is not None and record.get("vvl_create_time") is not None \
                    and owner_live.get("create_time") is not None \
                    and abs(owner_live["create_time"] - record["vvl_create_time"]) <= 1.0:
                logger.info("The engine in the pidfile belongs to another running VoucherVisionLocal; left alone.")
                return "owned"
        live = procutil.process_identity(pid)
        if live is None:
            path.unlink(missing_ok=True)
            return "gone"
        if not procutil.same_process(record, live):
            logger.info("Pid %s from the engine pidfile is now a different process; not touching it.", pid)
            path.unlink(missing_ok=True)
            return "reused"
        logger.warning("Stopping a stale llama-server (pid %s) left by an earlier VoucherVisionLocal.", pid)
        self._log(f"[vvl] stopping a stale llama-server (pid {pid}) left by an earlier run")
        procutil.kill_pid_tree(pid)
        procutil.wait_gone(pid, 10)
        path.unlink(missing_ok=True)
        return "killed"

    # -- shutdown ------------------------------------------------------------------

    def shutdown(self, timeout: float = 15.0) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        if self.is_running():
            self.stop()
        if self._owns_logs:
            self.logs.close()


# --------------------------------------------------------------- helpers --

def build_command(server: str, *, model: str, mmproj: str | None, port: int, alias: str, context_total: int,
                  parallel: int, backend: str, infer: dict) -> list[str]:
    """llama-server argv (flags checked against b12359's --help; see the module docstring)."""
    argv = [
        server,
        "--model", model,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--alias", alias,
        "--ctx-size", str(context_total),
        "--parallel", str(parallel),
        "--no-kv-unified",
        # --fit adjusts only what is not set: with ctx-size fixed it decides how
        # many layers go on the GPU (all of them when the model fits).
        "--fit", "on",
        "--jinja",
        "--no-webui",
        "--cors-origins", "localhost",
        "--log-colors", "off",
        "--timeout", str(int(infer.get("request_timeout_seconds") or 900)),
    ]
    if backend == "cpu":
        argv += ["--n-gpu-layers", "0"]
    if mmproj:
        argv += ["--mmproj", mmproj]
    if infer.get("image_min_tokens"):
        argv += ["--image-min-tokens", str(int(infer["image_min_tokens"]))]
    if infer.get("image_max_tokens"):
        argv += ["--image-max-tokens", str(int(infer["image_max_tokens"]))]
    extra = infer.get("server_args") or []
    argv += [str(a) for a in extra]
    return argv


def vulkan_indexes(selected: list[int], report: dict | None, devices: list[dict] | None) -> list[int]:
    """Map /v1/system GPU indexes to Vulkan device numbers.

    The installer's `--list-devices` output names the Vulkan devices in their
    enumeration order ("Vulkan0: <name> (...)"). A GPU that is the k-th card
    with its name in /v1/system (PCI order) maps to the k-th Vulkan device with
    that name, so two identical cards keep their order. Without that list (or
    without a name match) the k-th usable GPU maps to the k-th Vulkan device.
    """
    vk = [d for d in (devices or []) if str(d.get("name", "")).startswith("Vulkan")]
    if not vk:
        return sorted(selected)
    gpus = sorted((g for g in (report or {}).get("gpus") or [] if isinstance(g, dict)
                   and isinstance(g.get("index"), int)), key=lambda g: g["index"])
    by_index = {g["index"]: g for g in gpus}

    def number(dev) -> int:
        digits = re.sub(r"\D", "", str(dev.get("name")))
        return int(digits) if digits else vk.index(dev)

    out = []
    for index in sorted(selected):
        name = str((by_index.get(index) or {}).get("name") or "").strip().lower()
        same_name = [g["index"] for g in gpus if str(g.get("name") or "").strip().lower() == name] if name else []
        candidates = [d for d in vk if name and str(d.get("description") or "").strip().lower() == name]
        if name and index in same_name and len(candidates) == len(same_name):
            out.append(number(candidates[same_name.index(index)]))
            continue
        position = [g["index"] for g in gpus].index(index) if index in by_index else index
        if 0 <= position < len(vk):
            out.append(number(vk[position]))
    return out


def device_env(backend: str, gpu_devices: list[int] | None, *, report: dict | None = None,
               devices: list[dict] | None = None) -> dict:
    if gpu_devices is None:
        return {}
    if backend == "cuda":
        # CUDA_DEVICE_ORDER=PCI_BUS_ID (engine_env) makes these nvidia-smi indexes.
        return {"CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in gpu_devices) if gpu_devices else "-1"}
    if backend == "vulkan" and gpu_devices:
        return {"GGML_VK_VISIBLE_DEVICES": ",".join(str(i) for i in vulkan_indexes(gpu_devices, report, devices))}
    return {}


def launch_parameters(family, build, install: dict, settings_now: dict) -> tuple[dict, int, int, str]:
    """(inference settings, context size per request, parallel requests, backend) for one start."""
    infer = inference_mod.settings_for(family, build)
    context_size = settings_now["context_size"] if settings_now.get("context_size") != "auto" \
        else int(infer["context_size"])
    parallel = settings_now.get("parallel")
    parallel = DEFAULT_PARALLEL if parallel in (None, "auto") else int(parallel)
    backend = install.get("backend") or "cpu"
    return infer, int(context_size), parallel, backend


def choose_gpu_devices(report: dict | None, backend: str, build, infer: dict, context_total: int,
                       setting) -> list[int] | None:
    """GPU indexes (as in /v1/system) for the engine, [] for CPU, None = engine default.

    The one place the choice is made: the supervisor calls it with a fresh
    system report when it starts a model, and the resource monitor calls it to
    show where a model will go.
    """
    if backend == "cpu":
        return []
    if backend == "metal":
        return None
    report = report or {}
    gpus = [g for g in report.get("gpus") or [] if isinstance(g, dict)]
    capability = report.get("capability") or {}
    usable = [i for i in capability.get("gpu_indices") or [] if isinstance(i, int)]
    if setting != "auto" and isinstance(setting, list):
        return list(setting)
    if not usable:
        return None
    candidates = [g for g in gpus if g.get("index") in usable]
    required = estimate_bytes(build, infer, context_total)
    fit_verdict = None
    try:
        from vvl.system import fit as sys_fit

        build_dict = build.model_dump()
        build_dict["context_default"] = context_total
        fit_verdict = sys_fit.assess(report, build_dict).get("verdict")
    except Exception:  # noqa: BLE001
        pass
    free_known = [g for g in candidates if isinstance(g.get("vram_free_bytes"), int)]
    if free_known:
        best = max(free_known, key=lambda g: g["vram_free_bytes"])
        if fit_verdict in ("fits_gpu", None, "unknown") and required <= best["vram_free_bytes"]:
            return [best["index"]]
    return sorted(usable)


def estimate_bytes(build, infer: dict, context_total: int) -> int:
    try:
        from vvl.system import policy

        per_token = infer.get("kv_bytes_per_token") or policy.KV_BYTES_PER_TOKEN
    except Exception:  # noqa: BLE001
        per_token = infer.get("kv_bytes_per_token") or 160 * 1024
    return int(sum(f.size_bytes for f in build.files) + context_total * per_token + DEFAULT_RUNTIME_MARGIN)


def explain_failure(outcome: str, tail: str, exit_code, load_timeout: float) -> str:
    if outcome == "timeout":
        return f"The engine did not finish loading the model within {int(load_timeout)} seconds."
    if _OOM.search(tail or ""):
        return ("The engine ran out of memory while loading or running the model. Lower the context size or "
                "parallel requests on the Engine tab, close other programs using the GPU, or choose a smaller model.")
    incompatible = classify_incompatible(tail)
    if incompatible:
        return f"The engine could not use the GPU: {incompatible}."
    if outcome == "crashed":
        return f"The engine stopped unexpectedly (exit code {exit_code})."
    return f"The engine exited while starting (exit code {exit_code})."
