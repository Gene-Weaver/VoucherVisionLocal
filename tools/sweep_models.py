"""Sweep every downloaded local model over the bench images, end to end through the app.

Developer tool, not part of the app. For each installed local build it runs
the bench images through a dedicated `vvl serve` instance (scratch home,
scratch settings, scratch credentials file), with:

  - the same model for OCR and for parsing
  - Output length: long (both steps)
  - Context: automatic (no per-model size)
  - prompt SLTPvM_full.yaml
  - Image detail: high (the maximum; models without the control ignore it)
  - no elevation / WFO lookups (nothing leaves this machine)

and two reasoning settings per model:

  off   reasoning off (Muse cannot turn it off: it reasons at "low")
  max   reasoning on at the highest level the model has, capped by
        STRENGTH_CAP (Muse: medium, Will 2026-11-19, to save time);
        with automatic context this is the largest window that fits

Models without a reasoning mode (Qwen3-VL Instruct) get only "off".

Each image is sent as an OCR pass (`/process stage=ocr`) followed at once by
its parsing pass (`/v1/process/finish`) in the same worker, so the two steps
are timed separately. Images run concurrently, as many at a time as the
engine pool has request slots (what the app's batch does).

Usage (from the repo root):

  uv run python tools/sweep_models.py --plan              # print the matrix, run nothing
  uv run python tools/sweep_models.py --run               # run everything (resumable)
  uv run python tools/sweep_models.py --run --limit 1 --only qwen3-vl-2b-q8__off

Results: bench/sweep_<date>/  (one folder per configuration, one JSON per
image, engine log slices, summary.json, and index.html + data.js to browse).
Re-running skips images that already have a result; --retry-errors redoes
failed ones.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import secrets
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
PROMPT = "SLTPvM_full.yaml"
STRENGTH_CAP = {"muse": "medium"}      # family id -> highest strength level used for "max"
IMAGE_DETAIL = "high"
DEFAULT_MODELS_DIR = "/datac/VoucherVisionLocal_Models"
DEFAULT_PORT = 8670
LOAD_TIMEOUT = 45 * 60          # seconds to wait for engines to load (cold 122B from the HDD is slow)
HTTP_TIMEOUT = 2 * 3600         # one /process call: up to 2 x 900 s request timeout plus queueing


# ------------------------------------------------------------ the matrix --

def installed_builds(models_dir: str):
    from vvl.local_models import is_installed
    from vvl.registry.loader import load_local_models

    out = []
    for family in load_local_models().families:
        for build in family.builds:
            try:
                if is_installed(models_dir, build):
                    out.append((family, build))
            except Exception:  # noqa: BLE001
                pass
    return out


def configurations(models_dir: str) -> list[dict]:
    """Every (model, reasoning) pair: all 'off' runs first, then all 'max' runs, smallest model first."""
    from vvl import generation

    rows = []
    for family, build in installed_builds(models_dir):
        model_id = f"local:{build.id}"
        entry = generation.entry_for(model_id)
        size = sum(f.size_bytes for f in build.files)
        supported = generation.reasoning_supported(entry)
        levels = generation.strength_levels(entry)
        can_off = generation.can_turn_off(entry)
        base = {"build_id": build.id, "model_id": model_id, "label": build.label, "family": family.id,
                "family_label": family.label, "size_bytes": size, "reasoning_supported": supported}
        off = dict(base, key=f"{build.id}__off", reasoning="off", strength=None,
                   reasoning_note=("no reasoning mode" if not supported else
                                   ("cannot turn off: reasons at 'low'" if not can_off else "off")))
        rows.append(off)
        if supported:
            top = levels[-1] if levels else None
            cap = STRENGTH_CAP.get(family.id)
            if top and cap in levels:
                top = cap
            suffix = "max" if not levels or top == levels[-1] else top
            rows.append(dict(base, key=f"{build.id}__{suffix}", reasoning="on", strength=top,
                             reasoning_note=f"on, strength {top}" if top else "on"))
    order = {"off": 0, "on": 1}
    rows.sort(key=lambda r: (order[r["reasoning"]], r["size_bytes"], r["build_id"]))
    for i, row in enumerate(rows, 1):
        row["index"] = i
    return rows


def generation_body(cfg: dict) -> dict:
    step = {"reasoning": cfg["reasoning"], "output_length": "long"}
    if cfg["strength"]:
        step["strength"] = cfg["strength"]
    return {"ocr": dict(step, image_detail=IMAGE_DETAIL), "parse": dict(step),
            "context": {cfg["model_id"]: "auto"}}


# ------------------------------------------------------------ the server --

def free_port(preferred: int) -> int:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


class Server:
    """A dedicated `vvl serve` with scratch home, settings, prompts and credentials."""

    def __init__(self, home: Path, models_dir: str, port: int):
        self.home = home
        self.models_dir = models_dir
        self.port = free_port(port)
        self.token = secrets.token_urlsafe(32)
        self.popen: subprocess.Popen | None = None
        self.base = f"http://127.0.0.1:{self.port}"
        self.client = httpx.Client(base_url=self.base, headers={"X-VVL-Token": self.token},
                                   timeout=httpx.Timeout(HTTP_TIMEOUT, connect=10.0))

    @property
    def engine_log(self) -> Path:
        return self.home / "logs" / "engine.log"

    def start(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        settings_path = self.home / "settings.yaml"
        settings_path.write_text(
            "# Sweep server settings (written by tools/sweep_models.py).\n"
            f"models_dir: {self.models_dir}\n"
            "enrichment:\n  elevation: false\n  wfo_source: 'off'\n")
        private = self.home / "PRIVATE.scratch.yaml"     # empty on purpose: no keys are needed
        if not private.exists():
            private.write_text("# scratch credentials file for the sweep (no keys)\n")
        env = dict(os.environ)
        env.update({
            "VVL_HOME": str(self.home),
            "VVL_SETTINGS_PATH": str(settings_path),
            "VVL_PRIVATE_PATH": str(private),
            "VVL_PROMPTS_USER_DIR": str(self.home / "prompts_user"),
            "VVL_TOKEN": self.token,
            "VVL_OWNER_PID": str(os.getpid()),
        })
        (self.home / "prompts_user").mkdir(exist_ok=True)
        log = open(self.home / "server_stdout.log", "a")
        self.popen = subprocess.Popen(
            ["uv", "run", "--project", str(REPO), "vvl", "serve", "--port", str(self.port)],
            cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True)
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self.popen.poll() is not None:
                raise RuntimeError(f"vvl serve exited ({self.popen.returncode}); see {self.home / 'server_stdout.log'}")
            try:
                if httpx.get(f"{self.base}/health", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        raise RuntimeError("vvl serve did not answer /health within 180 s")

    def stop(self) -> None:
        if not self.popen or self.popen.poll() is not None:
            return
        try:
            self.client.post("/v1/engine/stop", timeout=60)
        except Exception:  # noqa: BLE001
            pass
        try:
            os.killpg(self.popen.pid, signal.SIGTERM)
            self.popen.wait(timeout=30)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(self.popen.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass

    def call(self, method: str, path: str, **kw) -> dict:
        r = self.client.request(method, path, **kw)
        try:
            data = r.json()
        except ValueError:
            data = {"error": r.text[:500]}
        if r.status_code >= 400:
            raise RuntimeError(f"{method} {path} -> {r.status_code}: {json.dumps(data)[:600]}")
        return data


# ------------------------------------------------------------- GPU memory --

def gpu_used() -> dict[int, int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
    except Exception:  # noqa: BLE001
        return {}
    used = {}
    for line in out.strip().splitlines():
        i, v = [p.strip() for p in line.split(",")]
        used[int(i)] = int(v)
    return used


class GpuSampler(threading.Thread):
    def __init__(self, period: float = 5.0):
        super().__init__(daemon=True)
        self.period = period
        self.peak: dict[int, int] = {}
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            for i, v in gpu_used().items():
                self.peak[i] = max(self.peak.get(i, 0), v)
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()


# ------------------------------------------------------------- one config --

PRINT_TIMING = re.compile(r"eval time =\s*([\d.]+) ms /\s*(\d+) tokens .*?([\d.]+) tokens per second")


def engine_speeds(log_text: str) -> dict:
    """Generation speed per request from llama-server print_timing lines (prompt lines excluded)."""
    gen = []
    for line in log_text.splitlines():
        if "prompt eval time" in line:
            continue
        m = PRINT_TIMING.search(line)
        if m:
            gen.append({"ms": float(m.group(1)), "tokens": int(m.group(2)), "tps": float(m.group(3))})
    if not gen:
        return {}
    return {"requests": len(gen), "gen_tokens_per_s_median": round(statistics.median(g["tps"] for g in gen), 1),
            "gen_tokens_max": max(g["tokens"] for g in gen)}


def wait_engines(server: Server, timeout: float) -> dict:
    began = time.monotonic()
    while True:
        status = server.call("GET", "/v1/engine")
        managed = status.get("managed") or {}
        engines = managed.get("engines") or []
        states = [e.get("state") for e in engines]
        if engines and all(s == "ready" for s in states):
            return status
        if any(s == "error" for s in states) or managed.get("state") == "error":
            raise RuntimeError(f"engine error: {managed.get('last_error') or states}")
        if time.monotonic() - began > timeout:
            raise RuntimeError(f"engines not ready after {int(timeout)} s: {states}")
        time.sleep(2.0)


def wait_stopped(server: Server, timeout: float = 120) -> None:
    began = time.monotonic()
    while time.monotonic() - began < timeout:
        managed = server.call("GET", "/v1/engine").get("managed") or {}
        if not managed.get("engines") or all(e.get("state") in ("stopped", None) for e in managed["engines"]):
            time.sleep(2.0)
            return
        time.sleep(1.0)


def summarize_record(rec) -> dict:
    if not isinstance(rec, dict):
        return {"fields": 0, "filled": 0}
    filled = sum(1 for v in rec.values() if isinstance(v, str) and v.strip())
    return {"fields": len(rec), "filled": filled}


def run_image(server: Server, cfg: dict, image: Path, limits: dict) -> dict:
    out = {"image": image.name, "config": cfg["key"], "started": dt.datetime.now().isoformat(timespec="seconds")}
    common = {"engines": cfg["model_id"], "llm_model": cfg["model_id"], "prompt": PROMPT,
              "include_wfo": "false", "include_cop90": "false"}
    t0 = time.monotonic()
    try:
        with open(image, "rb") as handle:
            staged = server.call("POST", "/process", data=dict(common, stage="ocr"),
                                 files={"file": (image.name, handle, "image/jpeg")})
        t1 = time.monotonic()
        out["ocr_seconds"] = round(t1 - t0, 2)
        info = next(iter((staged.get("ocr_info") or {}).values()), {}) or {}
        out["ocr_tokens_in"] = info.get("tokens_in")
        out["ocr_tokens_out"] = info.get("tokens_out")
        out["ocr_thinking_tokens"] = info.get("thinking_tokens")
        out["ocr_text"] = staged.get("ocr", "")
        if "stage_token" not in staged:
            raise RuntimeError(f"no stage_token in OCR reply: {json.dumps(staged)[:300]}")
        result = server.call("POST", "/v1/process/finish",
                             json={"stage_token": staged["stage_token"], "llm_model": cfg["model_id"],
                                   "prompt": PROMPT, "include_wfo": False, "include_cop90": False})
        t2 = time.monotonic()
        out["parse_seconds"] = round(t2 - t1, 2)
        parsing = result.get("parsing_info") or {}
        out["parse_tokens_in"] = parsing.get("input")
        out["parse_tokens_out"] = parsing.get("output")
        out["parse_thinking_tokens"] = parsing.get("thinking_tokens")
        record = result.get("formatted_json")
        out["record"] = record
        out["record_stats"] = summarize_record(record)
        out["ok"] = True
        out["llm_success"] = (result.get("success") or {}).get("llm")
        out["text_collage"] = (result.get("success") or {}).get("text_collage")
    except Exception as exc:  # noqa: BLE001
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"[:1500]
    out["total_seconds"] = round(time.monotonic() - t0, 2)
    out["ocr_hit_limit"] = bool(limits.get("ocr") and (out.get("ocr_tokens_out") or 0) >= limits["ocr"])
    out["parse_hit_limit"] = bool(limits.get("parse") and (out.get("parse_tokens_out") or 0) >= limits["parse"])
    out["record_empty"] = out.get("ok") is True and (out.get("record_stats") or {}).get("filled", 0) == 0
    if "jouvea" in image.name.lower():
        # VVGO "Jouvea Test": PASS when formatted_json.scientificName contains "Jouvea" (case-insensitive).
        name = str((out.get("record") or {}).get("scientificName") or "")
        out["jouvea_pass"] = "jouvea" in name.lower()
        out["jouvea_scientificName"] = name
    return out


def run_config(server: Server, cfg: dict, images: list[Path], out_dir: Path, retry_errors: bool) -> dict:
    cdir = out_dir / cfg["key"]
    cdir.mkdir(parents=True, exist_ok=True)
    meta_path = cdir / "config.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({k: cfg[k] for k in cfg})

    def done(image: Path) -> bool:
        p = cdir / f"{image.stem}.json"
        if not p.exists():
            return False
        if not retry_errors:
            return True
        try:
            return json.loads(p.read_text()).get("ok") is True
        except ValueError:
            return False

    todo = [im for im in images if not done(im)]
    if not todo:
        print(f"[{cfg['index']:>2}] {cfg['key']}: already complete", flush=True)
        return meta

    print(f"[{cfg['index']:>2}] {cfg['key']}: {len(todo)} image(s) to run", flush=True)
    server.call("PUT", "/v1/selection", json={"ocr": [cfg["model_id"]], "parse": cfg["model_id"]})
    gen = server.call("PUT", "/v1/generation", json=generation_body(cfg))
    meta["generation"] = gen

    def sent(step):
        models = (gen.get(step) or {}).get("models") or []
        return (models[0].get("sent") if models else {}) or {}

    limits = {"ocr": sent("ocr").get("max_tokens"), "parse": sent("parse").get("max_tokens")}
    meta["max_tokens"] = limits

    log_offset = server.engine_log.stat().st_size if server.engine_log.exists() else 0
    server.call("POST", "/v1/engine/stop")
    wait_stopped(server)
    baseline = gpu_used()
    sampler = GpuSampler()
    sampler.start()
    began = time.monotonic()
    try:
        server.call("POST", "/v1/engine/start", json={"model_id": cfg["model_id"]})
        status = wait_engines(server, LOAD_TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        sampler.stop()
        meta["failed"] = f"{type(exc).__name__}: {exc}"[:1500]
        meta["load_seconds"] = round(time.monotonic() - began, 1)
        print(f"     FAILED to start: {meta['failed'][:300]}", flush=True)
        _save_log_slice(server, log_offset, cdir)
        meta_path.write_text(json.dumps(meta, indent=1))
        try:
            server.call("POST", "/v1/engine/stop")
        except Exception:  # noqa: BLE001
            pass
        return meta
    meta.pop("failed", None)
    meta["load_seconds"] = round(time.monotonic() - began, 1)
    managed = status.get("managed") or {}
    meta["engines"] = [{k: e.get(k) for k in ("number", "gpu_devices", "parallel", "context_size")}
                       for e in managed.get("engines") or []]
    meta["plan"] = status.get("plan")
    concurrency = int(managed.get("parallel_total") or (status.get("plan") or {}).get("concurrency") or 1)
    meta["concurrency"] = concurrency
    meta["gpu_baseline_mib"] = baseline
    layout = ", ".join(f"GPU {e['gpu_devices']} x{e['parallel']} slots @ {e['context_size']} ctx"
                       for e in meta["engines"])
    print(f"     loaded in {meta['load_seconds']} s: {layout}; {concurrency} at a time; "
          f"max_tokens ocr {limits['ocr']} parse {limits['parse']}", flush=True)
    meta_path.write_text(json.dumps(meta, indent=1))

    run_began = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(run_image, server, cfg, im, limits): im for im in todo}
        for fut in as_completed(futures):
            im = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001
                res = {"image": im.name, "config": cfg["key"], "ok": False, "error": repr(exc)}
            (cdir / f"{im.stem}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
            flag = "ok " if res.get("ok") else "ERR"
            extra = (" [OCR hit limit]" if res.get("ocr_hit_limit") else "") + \
                    (" [parse hit limit]" if res.get("parse_hit_limit") else "") + \
                    (" [empty record]" if res.get("record_empty") else "")
            print(f"     {flag} {im.stem[:48]:<48} ocr {res.get('ocr_seconds', '-'):>7} s "
                  f"({res.get('ocr_tokens_out', '-')} tok)  parse {res.get('parse_seconds', '-'):>7} s "
                  f"({res.get('parse_tokens_out', '-')} tok){extra}"
                  + (f"  {res.get('error', '')[:160]}" if not res.get("ok") else ""), flush=True)
    sampler.stop()
    meta["batch_wall_seconds"] = round(time.monotonic() - run_began, 1)
    meta["batch_images"] = len(todo)
    meta["gpu_peak_mib"] = sampler.peak
    meta["engine_speed"] = engine_speeds(_save_log_slice(server, log_offset, cdir))
    meta["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    meta_path.write_text(json.dumps(meta, indent=1))
    server.call("POST", "/v1/engine/stop")
    return meta


def _save_log_slice(server: Server, offset: int, cdir: Path) -> str:
    try:
        with open(server.engine_log, "rb") as handle:
            handle.seek(offset)
            text = handle.read().decode("utf-8", "replace")
    except OSError:
        return ""
    with open(cdir / "engine.log", "a") as out:
        out.write(text)
    return text


# ------------------------------------------------------------- reporting --

def collect(out_dir: Path, configs: list[dict], images: list[Path]) -> dict:
    rows = []
    results = {}
    for cfg in configs:
        cdir = out_dir / cfg["key"]
        meta = json.loads((cdir / "config.json").read_text()) if (cdir / "config.json").exists() else {}
        per = []
        for im in images:
            p = cdir / f"{im.stem}.json"
            if p.exists():
                try:
                    per.append(json.loads(p.read_text()))
                except ValueError:
                    pass
        ok = [r for r in per if r.get("ok")]
        results[cfg["key"]] = {r["image"]: r for r in per}

        def mean(key):
            vals = [r[key] for r in ok if isinstance(r.get(key), (int, float))]
            return round(statistics.mean(vals), 1) if vals else None

        wall = meta.get("batch_wall_seconds")
        n = meta.get("batch_images")
        rows.append({
            **{k: cfg[k] for k in ("index", "key", "build_id", "label", "family_label", "size_bytes",
                                   "reasoning", "strength", "reasoning_note", "reasoning_supported")},
            "state": ("failed" if meta.get("failed") else
                      "done" if len(per) == len(images) else
                      "partial" if per else "pending"),
            "failed": meta.get("failed"),
            "images_done": len(per), "images_ok": len(ok),
            "errors": len(per) - len(ok),
            "ocr_hit_limit": sum(1 for r in ok if r.get("ocr_hit_limit")),
            "parse_hit_limit": sum(1 for r in ok if r.get("parse_hit_limit")),
            "empty_records": sum(1 for r in ok if r.get("record_empty")),
            "fields_filled_mean": (round(statistics.mean(r["record_stats"]["filled"] for r in ok), 1)
                                   if ok else None),
            "fields_total": next((r["record_stats"]["fields"] for r in ok if r.get("record_stats")), None),
            "ocr_seconds_mean": mean("ocr_seconds"), "parse_seconds_mean": mean("parse_seconds"),
            "total_seconds_mean": mean("total_seconds"),
            "ocr_tokens_out_mean": mean("ocr_tokens_out"), "parse_tokens_out_mean": mean("parse_tokens_out"),
            "images_per_min": round(n / wall * 60, 2) if wall and n else None,
            "batch_wall_seconds": wall,
            "load_seconds": meta.get("load_seconds"),
            "engines": meta.get("engines"), "concurrency": meta.get("concurrency"),
            "max_tokens": meta.get("max_tokens"),
            "context_mode": _context_mode(meta),
            "sent": _sent(meta),
            "gpu_peak_mib": meta.get("gpu_peak_mib"), "gpu_baseline_mib": meta.get("gpu_baseline_mib"),
            "engine_speed": meta.get("engine_speed"),
            "finished": meta.get("finished"),
            "jouvea": next(({"pass": r.get("jouvea_pass"), "scientificName": r.get("jouvea_scientificName"),
                             "ok": r.get("ok"), "error": r.get("error")}
                            for r in per if "jouvea" in r.get("image", "").lower()), None),
        })
    return {"rows": rows, "results": results}


def _context_mode(meta: dict):
    ctx = ((meta.get("generation") or {}).get("context") or {}).get("models") or []
    return ctx[0] if ctx else None


def _sent(meta: dict):
    gen = meta.get("generation") or {}
    out = {}
    for step in ("ocr", "parse"):
        models = (gen.get(step) or {}).get("models") or []
        out[step] = models[0].get("sent") if models else None
    return out


def write_report(out_dir: Path, configs: list[dict], images: list[Path], header: dict) -> None:
    data = collect(out_dir, configs, images)
    data["header"] = header
    data["images"] = [im.name for im in images]
    data["image_paths"] = {im.name: os.path.relpath(im, out_dir) for im in images}
    data["updated"] = dt.datetime.now().isoformat(timespec="seconds")
    (out_dir / "summary.json").write_text(json.dumps({"header": header, "updated": data["updated"],
                                                      "rows": data["rows"]}, indent=1))
    (out_dir / "data.js").write_text("window.SWEEP = " + json.dumps(data, ensure_ascii=False) + ";\n")
    template = REPO / "tools" / "sweep_viewer.html"
    if template.exists():
        shutil.copyfile(template, out_dir / "index.html")


def machine_header(models_dir: str, images_dir: Path) -> dict:
    gpus = []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
        gpus = [line.strip() for line in out.strip().splitlines()]
    except Exception:  # noqa: BLE001
        pass
    engine = None
    for manifest in sorted((Path(models_dir) / "engine").glob("*/*/vvl-install.json")):
        try:
            d = json.loads(manifest.read_text())
            engine = f"llama.cpp {d.get('version')} ({d.get('variant')})"
        except ValueError:
            pass
    try:
        version = (REPO / "VERSION").read_text().strip()
    except OSError:
        version = None
    return {"host": socket.gethostname(), "gpus": gpus, "engine": engine, "vvl_version": version,
            "prompt": PROMPT, "images_dir": str(images_dir), "models_dir": models_dir,
            "settings": "same model for OCR and parsing; output length long (both steps); context automatic; "
                        "image detail high; reasoning off, then max (Muse: medium); "
                        "sampling = model card defaults; no elevation/WFO lookups",
            "method": "dedicated vvl serve instance; per image /process stage=ocr then /v1/process/finish; "
                      "images sent concurrently up to the engine pool's request slots"}


# ------------------------------------------------------------------ main --

def print_plan(configs: list[dict], images: list[Path]) -> None:
    from vvl import generation

    print(f"{len(configs)} configurations x {len(images)} images = {len(configs) * len(images)} image runs\n")
    print(f"{'#':>2}  {'configuration':<40} {'size':>7}  reasoning sent                  max_tokens (ocr/parse)")
    for cfg in configs:
        settings = {"selection": {"ocr": [cfg["model_id"]], "parse": cfg["model_id"]},
                    "generation": generation_body(cfg) | {"context": {}}}
        req = generation.step_request(cfg["model_id"], "ocr", settings)
        preq = generation.step_request(cfg["model_id"], "parse", settings)
        rule = generation.context_rule(cfg["model_id"], settings)
        sent = json.dumps(req["extra_body"].get("chat_template_kwargs")) if req["extra_body"] else \
            (repr(req["prompt_suffix"]) if req["prompt_suffix"] else "-")
        print(f"{cfg['index']:>2}  {cfg['key']:<40} {cfg['size_bytes'] / 1e9:5.1f}GB  {sent:<30}  "
              f"{req['max_tokens']}/{preq['max_tokens']}  context {rule['mode']}"
              f"{'' if rule['mode'] == 'auto_max' else ' ' + str(rule['tokens'])}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="print the matrix and exit")
    mode.add_argument("--run", action="store_true", help="run the sweep (resumable)")
    mode.add_argument("--report", action="store_true", help="only rebuild summary.json / data.js / index.html")
    parser.add_argument("--out", default=str(REPO / "bench" / "sweep_2026-11-19"))
    parser.add_argument("--images", default=str(REPO / "bench" / "images"))
    parser.add_argument("--extra-images", default=str(REPO / "bench" / "extra_images"),
                        help="folder of extra test images run after the bench set (the VVGO Jouvea test image)")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N images")
    parser.add_argument("--only", default="", help="comma list of configuration keys or build ids")
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--home", default=str(Path.home() / ".local" / "share" / "VoucherVisionLocal-sweep"),
                        help="scratch VVL_HOME for the sweep server (outside Dropbox)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()

    images_dir = Path(args.images)
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    extra_dir = Path(args.extra_images) if args.extra_images else None
    if extra_dir and extra_dir.is_dir():
        images += sorted(p for p in extra_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    all_images = list(images)
    if args.limit:
        images = images[:args.limit]
    configs = configurations(args.models_dir)
    all_configs = list(configs)
    if args.only:
        wanted = {w.strip() for w in args.only.split(",") if w.strip()}
        configs = [c for c in configs if c["key"] in wanted or c["build_id"] in wanted]

    out_dir = Path(args.out)
    if args.plan:
        print_plan(configs, images)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    header = machine_header(args.models_dir, images_dir)
    (out_dir / "plan.json").write_text(json.dumps({"header": header, "configs": all_configs,
                                                   "images": [p.name for p in all_images]}, indent=1))
    if args.report:
        write_report(out_dir, all_configs, all_images, header)
        print(f"Report: {out_dir / 'index.html'}")
        return 0

    server = Server(Path(args.home), args.models_dir, args.port)
    stopping = {"flag": False}

    def on_signal(signum, _frame):
        stopping["flag"] = True
        server.stop()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, on_signal)
    print(f"Starting sweep server on port {server.port} (home {server.home})", flush=True)
    server.start()
    try:
        write_report(out_dir, all_configs, all_images, header)
        for cfg in configs:
            try:
                run_config(server, cfg, images, out_dir, args.retry_errors)
            except Exception:  # noqa: BLE001
                print(f"     configuration {cfg['key']} crashed:\n{traceback.format_exc()}", flush=True)
                try:
                    server.call("POST", "/v1/engine/stop")
                except Exception:  # noqa: BLE001
                    pass
            write_report(out_dir, all_configs, all_images, header)
            if stopping["flag"]:
                break
    finally:
        server.stop()
        write_report(out_dir, all_configs, all_images, header)
    print(f"Done. Report: {out_dir / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
