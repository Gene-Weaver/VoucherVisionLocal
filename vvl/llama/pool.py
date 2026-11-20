"""The engine pool: several llama-server engines behind one interface.

One `EngineSupervisor` runs one llama-server with one model. The pool keeps a
few of them and decides, with `planner.plan_layout`, which model runs on which
graphics card and with how many request slots:

- one model on every card it fits on, so all cards work on a batch;
- the OCR model and the parsing model loaded side by side when both fit;
- one model at a time, on all cards, when they do not fit together.

The rest of the app talks to the pool exactly as it talked to the single
supervisor (`lease`, `start`, `stop`, `status`, `logs`, ...). A request asks for
a model with `lease(model_id)`; the pool starts whatever is missing and hands
the request to the least busy engine that serves the model.

Changing what is loaded
-----------------------
Engines that already serve a wanted model are left alone, so a plan that comes
out slightly differently (free memory moves a little) never restarts a working
engine. New engines are placed in the memory that is actually free. Engines for
models that are no longer wanted are stopped once their requests finish.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time

from vvl.llama import inference as inference_mod
from vvl.llama.planner import GIB, Card, Layout, ModelNeed, Rules, UnitSpec, plan_layout
from vvl.llama.supervisor import (
    DEFAULT_RUNTIME_MARGIN,
    IDLE_CHECK_SECONDS,
    EngineNotReady,
    EngineSupervisor,
    LogBuffer,
    SupervisorError,
    _now_iso,
)
from vvl.local_models import is_installed

logger = logging.getLogger(__name__)

MAX_UNITS = 8
# Graphics memory left untouched on every card, on top of what other programs use.
CARD_RESERVE_BYTES = 2 * GIB
LIVE_STATES = ("starting", "loading_model", "ready")


def selected_local_models(selection: dict | None) -> list[tuple[str, tuple[str, ...]]]:
    """[(model id, roles)] for the local models of a saved selection, in order."""
    roles: dict[str, list[str]] = {}
    selection = selection if isinstance(selection, dict) else {}
    for model_id in selection.get("ocr") if isinstance(selection.get("ocr"), list) else []:
        if isinstance(model_id, str) and model_id.startswith("local:"):
            if "ocr" not in roles.setdefault(model_id, []):
                roles[model_id].append("ocr")
    parse = selection.get("parse")
    if isinstance(parse, str) and parse.startswith("local:"):
        if "parse" not in roles.setdefault(parse, []):
            roles[parse].append("parse")
    return [(model_id, tuple(r)) for model_id, r in roles.items()]


class EnginePool:
    def __init__(self, runtime):
        self.rt = runtime
        cfg = getattr(runtime, "config", {}) or {}
        self.idle_check_seconds = float(cfg.get("VVL_ENGINE_IDLE_CHECK_SECONDS") or IDLE_CHECK_SECONDS)
        self.logs = LogBuffer()
        self._cond = threading.Condition(threading.RLock())
        self._units: list[EngineSupervisor] = []
        self._specs: dict[int, UnitSpec] = {}        # unit number -> what it was started with
        self._active: dict[str, int] = {}             # requests in progress, per model
        self._unit_load: dict[int, int] = {}          # requests handed to each engine
        self._switch_to: str | None = None
        self._layout: Layout | None = None
        self._launch: dict[str, dict] = {}            # model id -> {"context_size", "infer", "signature"}
        self._closed = False
        self._background: threading.Thread | None = None
        self._unit(0)

    # ------------------------------------------------------------- units --

    def _unit(self, number: int) -> EngineSupervisor:
        with self._cond:
            while len(self._units) <= number:
                self._units.append(EngineSupervisor(self.rt, unit=len(self._units), logs=self.logs))
            return self._units[number]

    @property
    def load_timeout(self) -> float:
        return self._unit(0).load_timeout

    def _snapshots(self) -> list[dict]:
        with self._cond:
            units = list(self._units)
        return [u.snapshot() for u in units]

    def _live(self) -> list[dict]:
        return [s for s in self._snapshots() if s["live"]]

    def _free_unit(self) -> EngineSupervisor | None:
        taken = {s["unit"] for s in self._snapshots() if s["live"] or s["running"] or s["state"] == "stopping"}
        for number in range(MAX_UNITS):
            if number not in taken:
                return self._unit(number)
        return None

    # ---------------------------------------------------------- planning --

    def _engine_settings(self) -> dict:
        return self._unit(0)._engine_settings()

    def _install(self) -> dict | None:
        return self._unit(0)._install()

    def _rules(self, settings_now: dict) -> Rules:
        parallel = settings_now.get("parallel")
        if parallel in (None, "auto"):
            return Rules()
        return Rules(fixed_slots=max(1, int(parallel)))

    def _need(self, model_id: str, roles: tuple[str, ...], settings_now: dict) -> ModelNeed | None:
        """Memory need of an installed local model, or None when it cannot be started."""
        if not isinstance(model_id, str) or not model_id.startswith("local:"):
            return None
        found = self.rt.local_registry().find_build(model_id.split(":", 1)[1])
        if not found:
            return None
        family, build = found
        try:
            if not is_installed(self.rt.models_dir(), build):
                return None
        except Exception:  # noqa: BLE001
            return None
        infer = inference_mod.settings_for(family, build)
        try:
            from vvl.system import policy

            per_token = infer.get("kv_bytes_per_token") or policy.KV_BYTES_PER_TOKEN
        except Exception:  # noqa: BLE001
            per_token = infer.get("kv_bytes_per_token") or 160 * 1024
        base = int(sum(f.size_bytes for f in build.files) + DEFAULT_RUNTIME_MARGIN)
        launch = self.launch_params(model_id, family, build, infer, base, per_token)
        return ModelNeed(model_id=model_id, base_bytes=base, slot_bytes=int(launch["context_size"] * per_token),
                         roles=roles or ("ocr", "parse"), label=build.label)

    def launch_params(self, model_id: str, family, build, infer: dict | None = None, base: int | None = None,
                      per_token: int | None = None) -> dict:
        """Context window and image limits an engine for `model_id` starts with.

        From the user's choices (vvl/generation.py): a fixed size, the smallest
        size that holds the chosen settings, or, with reasoning on, as large as
        fits in graphics memory with one request slot, up to the model's maximum.
        `signature` changes only when the choices change (not with free memory),
        so a running engine is restarted only for a real change.
        """
        from vvl import generation

        infer = dict(infer or inference_mod.settings_for(family, build))
        try:
            full = self.rt.settings()
        except Exception:  # noqa: BLE001
            full = {}
        registry = self.rt.local_registry()
        image = generation.engine_params(model_id, full, registry)
        infer.update(image)
        model_max = generation.model_max_context(self.rt.models_dir(), build)
        rule = generation.context_rule(model_id, full, registry, model_max=model_max,
                                       image_max_tokens=infer.get("image_max_tokens"))
        context = rule["tokens"] or rule["needed"]
        if rule["mode"] == "auto_max":
            context = self._largest_context(build, base, per_token, rule, model_max) or rule["needed"]
        params = {
            "context_size": int(context),
            "infer": image,
            "rule": rule,
            "model_max": model_max,
            "signature": (rule["mode"], rule["tokens"] if rule["mode"] == "fixed" else rule["needed"],
                          tuple(sorted(image.items()))),
        }
        with self._cond:
            self._launch[model_id] = params
        return params

    def _largest_context(self, build, base: int | None, per_token: int | None, rule: dict,
                         model_max: int | None) -> int | None:
        """The largest context window (one request slot) that fits on the roomiest card, or on all cards."""
        if not base or not per_token:
            return None
        install = self._install()
        backend = (install or {}).get("backend") or "cpu"
        cards = self._cards(self._engine_settings(), backend, give_back=self._live())
        if not cards:
            return None
        best = max(c.budget_bytes for c in cards)
        room = best - base
        if room < rule["needed"] * per_token:
            room = sum(c.budget_bytes for c in cards) - base
        tokens = (room // per_token) // 1024 * 1024
        if model_max:
            tokens = min(tokens, model_max)
        return max(int(tokens), int(rule["needed"]))

    def _needs(self, settings_now: dict, also: str | None = None) -> list[ModelNeed]:
        """The working set: the selected local models, plus `also` (a model a request needs)."""
        try:
            selection = self.rt.settings().get("selection") or {}
        except Exception:  # noqa: BLE001
            selection = {}
        wanted = selected_local_models(selection)
        if also and also not in [m for m, _r in wanted]:
            wanted.append((also, ("ocr", "parse")))
        needs = []
        for model_id, roles in wanted:
            need = self._need(model_id, roles, settings_now)
            if need is not None:
                needs.append(need)
        return needs

    def _card_free(self) -> dict[int, int] | None:
        """Free graphics memory per usable card, as fresh as possible; None when unknown."""
        report = self.rt.system_report(False) or {}
        usable = [i for i in (report.get("capability") or {}).get("gpu_indices") or [] if isinstance(i, int)]
        if not usable:
            return None
        free: dict[int, int] = {}
        cfg = getattr(self.rt, "config", {}) or {}
        # A test or scratch run that supplies its own system report gets its
        # figures from that report (or from the sampler it also supplies).
        use_live = cfg.get("VVL_LIVE_SAMPLER") is not None or cfg.get("VVL_SYSTEM_REPORT") is None
        try:
            if use_live:
                from vvl.system import live

                sampler = cfg.get("VVL_LIVE_SAMPLER") or live.default_sampler()
                for g in live.build_gpus(report, sampler.sample()):
                    if g.get("index") in usable and isinstance(g.get("vram_free_bytes"), int):
                        free[g["index"]] = g["vram_free_bytes"]
        except Exception:  # noqa: BLE001
            free = {}
        if len(free) < len(usable):
            fresh = self.rt.system_report(True) or report
            for g in fresh.get("gpus") or []:
                if isinstance(g, dict) and g.get("index") in usable and g["index"] not in free \
                        and isinstance(g.get("vram_free_bytes"), int):
                    free[g["index"]] = g["vram_free_bytes"]
        return free or None

    def _cards(self, settings_now: dict, backend: str, *, give_back: list[dict] = (),
               free: dict[int, int] | None = None) -> list[Card]:
        """Per-card budgets. `give_back` lists running engines whose memory counts as free.

        `free` replaces the measurement: free bytes per card as they would be
        with none of our engines running (the resource monitor has those figures).
        """
        if backend == "cpu":
            return []
        if backend == "metal":
            report = self.rt.system_report(False) or {}
            budget = (report.get("capability") or {}).get("gpu_memory_budget_bytes")
            return [Card(0, int(budget))] if isinstance(budget, int) and budget > 0 else []
        if free is not None:
            give_back = ()
            free = dict(free)
        else:
            free = self._card_free()
        if not free:
            return []
        chosen = settings_now.get("gpu_devices")
        if isinstance(chosen, list):
            free = {i: v for i, v in free.items() if i in chosen}
        with self._cond:
            specs = dict(self._specs)
        for snap in give_back:
            spec = specs.get(snap["unit"])
            devices = (snap.get("gpu_devices") if snap.get("gpu_devices") is not None
                       else (spec.gpu_devices if spec else None)) or []
            for index in devices:
                if index in free and spec is not None:
                    free[index] += spec.bytes // max(1, len(devices))
        return [Card(i, max(0, v - CARD_RESERVE_BYTES)) for i, v in sorted(free.items())]

    def plan(self, active: str | None = None, *, free: dict[int, int] | None = None) -> Layout | None:
        """The layout the pool aims for, as if nothing of ours were running yet."""
        install = self._install()
        if install is None:
            return None
        settings_now = self._engine_settings()
        needs = self._needs(settings_now, also=active)
        if not needs:
            return Layout("none", [])
        backend = install.get("backend") or "cpu"
        cards = self._cards(settings_now, backend, give_back=self._live(), free=free)
        if active is None:
            loaded = [s["model_id"] for s in self._live()]
            active = next((n.model_id for n in needs if n.model_id in loaded), needs[0].model_id)
        return plan_layout(needs, cards, rules=self._rules(settings_now), active=active, backend=backend)

    def _changes(self, active: str) -> tuple[list[UnitSpec], list[dict], Layout]:
        """(engines to start, running engines to stop, target layout) so `active` gets served."""
        install = self._install()
        settings_now = self._engine_settings()
        backend = (install or {}).get("backend") or "cpu"
        needs = self._needs(settings_now, also=active)
        by_id = {n.model_id: n for n in needs}
        if active not in by_id:
            # Not startable: let the unit raise the precise reason.
            self._unit(0)._check_startable_locked(active)
            raise EngineNotReady("This model cannot be started.")
        rules = self._rules(settings_now)
        live = self._live()
        full = plan_layout(needs, self._cards(settings_now, backend, give_back=live), rules=rules, active=active,
                           backend=backend)
        wanted = set(full.models())
        with self._cond:
            launch = dict(self._launch)
        stale = [s for s in live if s["model_id"] in wanted
                 and self._unit(s["unit"]).overrides.get("signature") != (launch.get(s["model_id"]) or {}).get("signature")]
        to_stop = [s for s in live if s["model_id"] not in wanted] + stale
        kept = [s for s in live if s["model_id"] in wanted and s not in stale]
        missing = [m for m in full.models() if not any(s["model_id"] == m for s in kept)]
        if not missing:
            return [], to_stop, full
        if kept:
            # Fit the missing models into what is really free once `to_stop` is gone.
            cards = self._cards(settings_now, backend, give_back=to_stop)
            extra = plan_layout([by_id[m] for m in missing], cards, rules=rules, active=active, backend=backend)
            if extra.fits and extra.mode in ("single", "shared") and set(extra.models()) == set(missing):
                return extra.units, to_stop, full
            # No room beside the engines that are running: rebuild from scratch.
            return full.units, live, full
        return full.units, to_stop, full

    # --------------------------------------------------- applying a layout --

    def _stop_units(self, snaps: list[dict]) -> None:
        threads = []
        for snap in snaps:
            unit = self._unit(snap["unit"])
            thread = threading.Thread(target=unit.stop, name=f"vvl-engine-stop-{snap['unit']}", daemon=True)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join(30)
        with self._cond:
            for snap in snaps:
                self._specs.pop(snap["unit"], None)

    def _start_units(self, specs: list[UnitSpec]) -> int:
        """Start one engine per spec. Returns how many were started."""
        started = 0
        for spec in specs:
            unit = self._free_unit()
            if unit is None:
                logger.warning("No free engine slot for %s.", spec.model_id)
                break
            with self._cond:
                launch = dict(self._launch.get(spec.model_id) or {})
            unit.overrides = {"parallel": spec.parallel, "gpu_devices": spec.gpu_devices}
            if launch:
                unit.overrides.update({"context_size": launch["context_size"], "infer": dict(launch["infer"]),
                                       "signature": launch["signature"]})
            with self._cond:
                self._specs[unit.unit] = spec
            unit.start(spec.model_id)
            started += 1
        self._label_units()
        return started

    def _label_units(self) -> None:
        """Prefix log lines with the engine number once more than one engine exists."""
        with self._cond:
            many = len(self._specs) > 1
            for unit in self._units:
                unit.log_prefix = f"[engine {unit.unit + 1}] " if many else ""

    def _stale(self, snap: dict) -> bool:
        """True when a running engine was started with settings the user has since changed."""
        model_id = snap.get("model_id")
        found = self.rt.local_registry().find_build(model_id.split(":", 1)[1]) if model_id else None
        if not found:
            return False
        current = self.launch_params(model_id, *found)
        return self._unit(snap["unit"]).overrides.get("signature") != current["signature"]

    # ------------------------------------------------------------ requests --

    @contextlib.contextmanager
    def lease(self, model_id: str, timeout: float | None = None):
        """Hold one request slot of an engine serving `model_id`, starting engines as needed."""
        timeout = self.load_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        unit = self._acquire(model_id, timeout, deadline)
        try:
            with unit.lease(model_id, timeout=max(1.0, deadline - time.monotonic())) as handle:
                yield handle
        finally:
            with self._cond:
                self._active[model_id] = max(0, self._active.get(model_id, 0) - 1)
                self._unit_load[unit.unit] = max(0, self._unit_load.get(unit.unit, 0) - 1)
                self._cond.notify_all()

    def _acquire(self, model_id: str, timeout: float, deadline: float) -> EngineSupervisor:
        with self._cond:
            try:
                while True:
                    if self._closed:
                        raise EngineNotReady("VoucherVisionLocal is shutting down.")
                    if self._switch_to in (None, model_id):
                        serving = [s for s in self._live() if s["model_id"] == model_id
                                   and not self._stale(s)]
                        if serving:
                            # The least busy engine, counting requests already handed to
                            # it that are still waiting for it to load.
                            def load(snap):
                                spec = self._specs.get(snap["unit"])
                                slots = snap["parallel"] or (spec.parallel if spec else 1) or 1
                                return (self._unit_load.get(snap["unit"], 0) + 1) / slots

                            best = min(serving, key=lambda s: (load(s), s["state"] != "ready", s["unit"]))
                            if self._switch_to == model_id:
                                self._switch_to = None
                            self._unit_load[best["unit"]] = self._unit_load.get(best["unit"], 0) + 1
                            self._active[model_id] = self._active.get(model_id, 0) + 1
                            self._cond.notify_all()
                            return self._unit(best["unit"])
                        to_start, to_stop, layout = self._changes(model_id)
                        busy = any(self._active.get(s["model_id"], 0) > 0 for s in to_stop)
                        if busy:
                            # Other models are mid-request: switch once they finish.
                            self._switch_to = model_id
                        else:
                            self._layout = layout
                            self._stop_units(to_stop)
                            try:
                                started = self._start_units(to_start)
                            except SupervisorError as exc:
                                raise EngineNotReady(exc.message, {"code": exc.code}) from None
                            if not started:
                                raise EngineNotReady("The engine could not be started for this model.")
                            continue
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise EngineNotReady(f"The engine did not become ready within {int(timeout)} seconds.")
                    self._cond.wait(min(remaining, 1.0))
            except BaseException:
                if self._switch_to == model_id:
                    self._switch_to = None
                    self._cond.notify_all()
                raise

    # -------------------------------------------------- start / stop / state --

    def start(self, model_id: str) -> dict:
        """Start `model_id` now (Engine tab). Engines in the way are stopped at once."""
        self._unit(0)._check_startable(model_id)
        with self._cond:
            if self._closed:
                raise SupervisorError("ENGINE_NOT_READY", "VoucherVisionLocal is shutting down.", status=503)
            if not any(s["model_id"] == model_id for s in self._live()):
                to_start, to_stop, layout = self._changes(model_id)
                self._layout = layout
                self._switch_to = None
                self._stop_units(to_stop)
                self._start_units(to_start)
        return self.status()

    def stop(self) -> dict:
        with self._cond:
            self._switch_to = None
            self._layout = None
        snaps = [s for s in self._snapshots() if s["live"] or s["running"]]
        self._stop_units(snaps)
        with self._cond:
            self._cond.notify_all()
        return self.status()

    def is_running(self) -> bool:
        return any(s["running"] for s in self._snapshots())

    def loaded_models(self) -> list[str]:
        out: list[str] = []
        for snap in self._live():
            if snap["model_id"] and snap["model_id"] not in out:
                out.append(snap["model_id"])
        return out

    def loaded_model(self) -> str | None:
        models = self.loaded_models()
        return models[0] if models else None

    def state(self) -> str:
        return self.status()["state"]

    def status(self) -> dict:
        """The `managed` object of GET /v1/engine: the first engine's fields, plus all engines."""
        snaps = self._snapshots()
        live = [s for s in snaps if s["live"]]
        primary_unit = self._unit(live[0]["unit"] if live else 0)
        status = primary_unit.status()
        states = [s["state"] for s in live]
        if any(st in ("starting", "loading_model") for st in states):
            status["state"] = "loading_model" if "loading_model" in states else "starting"
        elif "ready" in states:
            status["state"] = "ready"
        elif not live:
            failed = [s for s in snaps if s["state"] == "error" and s["last_error"]]
            if failed and status["state"] not in ("error", "not_installed"):
                status["state"] = "error"
                status["last_error"] = failed[-1]["last_error"]
        with self._cond:
            specs = dict(self._specs)
            layout = self._layout
            active = dict(self._active)
        engines = []
        for snap in live:
            spec = specs.get(snap["unit"])
            engines.append({
                "number": snap["unit"] + 1,
                "model_id": snap["model_id"],
                "state": snap["state"],
                "gpu_devices": snap["gpu_devices"] if snap["gpu_devices"] is not None
                else (spec.gpu_devices if spec else None),
                "parallel": snap["parallel"] or (spec.parallel if spec else None),
                "context_size": snap["context_size"],
                "active_requests": snap["leases"],
                "pid": snap["pid"],
                "port": snap["port"],
                "started_at": snap["started_at"],
            })
        status["engines"] = engines
        status["loaded_models"] = self.loaded_models()
        status["active_requests"] = sum(active.values())
        status["parallel_total"] = sum(e["parallel"] or 0 for e in engines) or None
        status["layout"] = {
            "mode": layout.mode if layout is not None and live else "none",
            "waiting": list(layout.waiting) if layout is not None and live else [],
        }
        if len(engines) > 1:
            devices: list[int] = []
            for engine in engines:
                for index in engine["gpu_devices"] or []:
                    if index not in devices:
                        devices.append(index)
            status["gpu_devices"] = sorted(devices) if devices else status.get("gpu_devices")
        return status

    def engine_pids(self) -> list[int]:
        return [s["pid"] for s in self._snapshots() if s["running"] and s["pid"]]

    def launch_preview(self, model_id: str, report: dict | None) -> dict | None:
        """Kept for callers of the single-engine supervisor."""
        return self._unit(0).launch_preview(model_id, report)

    # ------------------------------------------------ background / lifecycle --

    def start_background(self) -> None:
        with self._cond:
            if self._background is not None:
                return
            self._background = threading.Thread(target=self._idle_loop, name="vvl-engine-idle", daemon=True)
            self._background.start()
        self._unit(0)._launcher_thread()

    def _idle_loop(self) -> None:
        while True:
            time.sleep(self.idle_check_seconds)
            with self._cond:
                if self._closed:
                    return
            self.check_idle()

    def check_idle(self) -> bool:
        """Stop engines nobody has used for the configured time. True when any was stopped."""
        with self._cond:
            if self._switch_to is not None or any(self._active.values()):
                return False
            units = list(self._units)
        stopped = False
        for unit in units:
            try:
                if unit.check_idle():
                    stopped = True
                    with self._cond:
                        self._specs.pop(unit.unit, None)
            except Exception:  # noqa: BLE001
                logger.exception("Idle check failed for engine %s.", unit.unit)
        if stopped:
            self._label_units()
        return stopped

    def reap_stale(self) -> str:
        """Clean up engines a crashed VoucherVisionLocal left behind."""
        results = []
        for number in range(MAX_UNITS):
            try:
                results.append(self._unit(number).reap_stale())
            except Exception:  # noqa: BLE001
                logger.exception("Could not check for a stale engine (%s).", number)
        for outcome in ("killed", "owned", "reused", "gone", "invalid", "unreadable"):
            if outcome in results:
                return outcome
        return "none"

    def shutdown(self, timeout: float = 15.0) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
            units = list(self._units)
        threads = []
        for unit in units:
            thread = threading.Thread(target=unit.shutdown, kwargs={"timeout": timeout}, daemon=True,
                                      name=f"vvl-engine-shutdown-{unit.unit}")
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join(timeout + 5)
        self.logs.close()


def describe_layout(layout: Layout | None, labels: dict[str, str] | None = None) -> str:
    """One plain sentence about a layout, for the Engine tab."""
    if layout is None or layout.mode == "none" or not layout.units:
        return ""
    labels = labels or {}
    parts = []
    for model_id in layout.models():
        units = [u for u in layout.units if u.model_id == model_id]
        slots = sum(u.parallel for u in units)
        name = labels.get(model_id, model_id)
        copies = f"{len(units)} copies" if len(units) > 1 else "1 copy"
        parts.append(f"{name}: {copies}, {slots} image{'s' if slots != 1 else ''} at a time")
    text = "; ".join(parts) + "."
    if layout.mode == "exclusive" and layout.waiting:
        text += " The selected models do not fit in graphics memory together, so they are loaded one at a time."
    return text


__all__ = ["EnginePool", "describe_layout", "selected_local_models", "CARD_RESERVE_BYTES", "MAX_UNITS", "_now_iso"]
