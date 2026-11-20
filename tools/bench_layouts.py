"""Measure engine layouts: how many llama-server copies and request slots are fastest.

This is a developer tool, not part of the app. It answers one question with
measurements on the current machine: for a given model, is it faster to run
one engine with many request slots, several engine copies on one graphics
card, or copies spread across cards?

For every layout it starts the engines directly (same command line the app
uses), then times two stages over the same set of label collages:

  OCR     image + OCR prompt -> text
  parse   text + parsing prompt + JSON schema -> record

Label collages are built once up front with the real TextCollage step, which
is timed separately because it runs on the CPU, one image at a time.

Usage (from the repo root):

  uv run python tools/bench_layouts.py --build qwen3-vl-2b-q8 --gpus 1 \
      --layouts 1x1,1x2,1x4,1x8,2x2,4x1,4x2

A layout is COPIESxSLOTS. Copies are placed round-robin on the cards named by
--gpus (indexes as shown by nvidia-smi / the Engine tab). Results are printed
as a table and written to bench/results/.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import json
import os
import queue
import secrets
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from vvl.engines.base import OCRRequest, ParseRequest  # noqa: E402
from vvl.engines.openai_compat import OpenAICompatConfig, OpenAICompatEngine, StepOptions  # noqa: E402
from vvl.llama import inference as inference_mod  # noqa: E402
from vvl.llama.installer import engine_env  # noqa: E402
from vvl.llama.supervisor import build_command, device_env  # noqa: E402
from vvl.local_models import file_path  # noqa: E402
from vvl.prompting import PromptBuilder, ocr_prompt_text  # noqa: E402
from vvl.registry.loader import load_local_models  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
BASE_PORT = 8700


# --------------------------------------------------------------- helpers --

def find_build(build_id: str):
    registry = load_local_models()
    for family in registry.families:
        for build in family.builds:
            if build.id == build_id:
                return family, build
    raise SystemExit(f"Unknown build id: {build_id}")


def find_install(models_dir: Path) -> dict:
    manifests = sorted((models_dir / "engine").glob("*/*/vvl-install.json"))
    if not manifests:
        raise SystemExit(f"No engine is installed under {models_dir / 'engine'}")
    manifest = manifests[-1]
    data = json.loads(manifest.read_text())
    server = manifest.parent / data["server"]
    data["server_path"] = str(server)
    data["bin_dir_path"] = str(server.parent)
    return data


def gpu_memory_used() -> dict[int, int]:
    """MiB used per card, by nvidia-smi index ({} when nvidia-smi is missing)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout
    except Exception:  # noqa: BLE001
        return {}
    used = {}
    for line in out.strip().splitlines():
        index, value = [part.strip() for part in line.split(",")]
        used[int(index)] = int(value)
    return used


def _child_setup():
    """New session, and die with this script (Linux)."""
    os.setsid()
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001
        pass


class Instance:
    def __init__(self, index: int, port: int, gpu: int | None, slots: int):
        self.index = index
        self.port = port
        self.gpu = gpu
        self.slots = slots
        self.api_key = secrets.token_urlsafe(24)
        self.popen: subprocess.Popen | None = None
        self.log_path: Path | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"


def start_instances(layout, gpus, family, build, install, models_dir, report, log_dir, context_size):
    copies, slots = layout
    infer = inference_mod.settings_for(family, build)
    context = int(context_size or infer["context_size"])
    model_file = next(f for f in build.files if f.role == "model")
    mmproj_file = next((f for f in build.files if f.role == "mmproj"), None)
    backend = install.get("backend") or "cpu"
    instances = []
    for i in range(copies):
        gpu = gpus[i % len(gpus)] if gpus else None
        inst = Instance(i, BASE_PORT + i, gpu, slots)
        argv = build_command(
            install["server_path"], model=str(file_path(models_dir, build, model_file)),
            mmproj=str(file_path(models_dir, build, mmproj_file)) if mmproj_file else None,
            port=inst.port, alias=build.id, context_total=context * slots, parallel=slots,
            backend=backend, infer=infer)
        env = engine_env(Path(install["bin_dir_path"]))
        env["LLAMA_API_KEY"] = inst.api_key
        env.update(device_env(backend, [gpu] if gpu is not None else None, report=report,
                              devices=install.get("devices")))
        inst.log_path = log_dir / f"engine_{copies}x{slots}_{i}.log"
        log = open(inst.log_path, "w")
        inst.popen = subprocess.Popen(argv, env=env, cwd=install["bin_dir_path"], stdout=log,
                                      stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                      preexec_fn=_child_setup)
        instances.append(inst)
    return instances, infer


def wait_ready(instances, timeout=600.0) -> float:
    started = time.monotonic()
    pending = list(instances)
    while pending:
        for inst in list(pending):
            if inst.popen.poll() is not None:
                tail = inst.log_path.read_text(errors="replace")[-1500:]
                raise RuntimeError(f"engine {inst.index} exited with code {inst.popen.returncode}:\n{tail}")
            try:
                response = httpx.get(f"http://127.0.0.1:{inst.port}/health", timeout=2.0)
                if response.status_code == 200:
                    pending.remove(inst)
            except httpx.HTTPError:
                pass
        if time.monotonic() - started > timeout:
            raise RuntimeError("engines did not become ready in time")
        if pending:
            time.sleep(0.25)
    return time.monotonic() - started


def stop_instances(instances) -> None:
    for inst in instances:
        if inst.popen and inst.popen.poll() is None:
            try:
                os.killpg(inst.popen.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 15
    for inst in instances:
        if not inst.popen:
            continue
        try:
            inst.popen.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(inst.popen.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            inst.popen.wait(timeout=10)


def make_engine(inst: Instance, build, infer: dict) -> OpenAICompatEngine:
    from vvl.engines.local import CAPABILITIES

    return OpenAICompatEngine(OpenAICompatConfig(
        base_url=inst.base_url, api_key=inst.api_key, model=build.id,
        model_id=f"local:{build.id}", label=build.label, capabilities=CAPABILITIES,
        ocr=StepOptions(max_tokens=infer["ocr"].get("max_tokens"),
                        sampling=dict(infer["ocr"].get("sampling") or {}),
                        image_first=infer["ocr"].get("image_first", True) is not False),
        parse=StepOptions(max_tokens=infer["parse"].get("max_tokens"),
                          sampling=dict(infer["parse"].get("sampling") or {})),
        timeout_seconds=float(infer.get("request_timeout_seconds") or 900),
        max_retries=1, service_label="The local engine"))


def run_stage(instances, engines, items, call):
    """Run `call(engine, item)` for every item with one worker per slot.

    Workers are bound to an engine copy and pull from one shared queue, so
    faster copies simply take more items. Returns (results by item index,
    wall seconds, per-item seconds).
    """
    todo: queue.Queue = queue.Queue()
    for index, item in enumerate(items):
        todo.put((index, item))
    results: dict[int, object] = {}
    durations: list[float] = []
    lock = threading.Lock()

    def worker(engine):
        while True:
            try:
                index, item = todo.get_nowait()
            except queue.Empty:
                return
            began = time.monotonic()
            try:
                value = call(engine, item)
            except Exception as exc:  # noqa: BLE001
                value = exc
            took = time.monotonic() - began
            with lock:
                results[index] = value
                durations.append(took)

    threads = []
    began = time.monotonic()
    for inst, engine in zip(instances, engines):
        for _ in range(inst.slots):
            thread = threading.Thread(target=worker, args=(engine,), daemon=True)
            thread.start()
            threads.append(thread)
    for thread in threads:
        thread.join()
    return results, time.monotonic() - began, durations


# ------------------------------------------------------------------ main --

def build_collages(images: list[Path], out_dir: Path) -> tuple[list[Path], float]:
    from vvl.pipeline import VoucherVisionProcessor

    processor = VoucherVisionProcessor()
    if processor.collage_engine is None:
        raise SystemExit("TextCollage could not be loaded.")
    collages = []
    began = time.monotonic()
    for image in images:
        data, _ran = processor._run_collage(str(image), use_original=False)
        target = out_dir / f"collage_{image.stem}.jpg"
        target.write_bytes(base64.b64decode(data["base64image_text_collage"]))
        collages.append(target)
    return collages, time.monotonic() - began


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", required=True, help="build id from models_local.yaml")
    parser.add_argument("--layouts", required=True, help="comma list of COPIESxSLOTS, e.g. 1x1,1x4,2x2")
    parser.add_argument("--gpus", default="", help="comma list of card indexes; empty = engine default")
    parser.add_argument("--images", default=str(REPO / "bench" / "images"))
    parser.add_argument("--limit", type=int, default=0, help="use only the first N images")
    parser.add_argument("--prompt", default=str(REPO / "prompts" / "builtin" / "SLTPvM_default.yaml"))
    parser.add_argument("--models-dir", default="/datac/VoucherVisionLocal_Models")
    parser.add_argument("--context-size", type=int, default=0, help="per-slot context; 0 = the model's default")
    parser.add_argument("--out", default="")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    family, build = find_build(args.build)
    install = find_install(models_dir)
    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    layouts = [tuple(int(n) for n in item.lower().split("x")) for item in args.layouts.split(",") if item.strip()]
    images = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise SystemExit(f"No images in {args.images}")

    from vvl import system as vvl_system

    report = vvl_system.get_report(refresh=True, models_dir=str(models_dir))

    results_dir = REPO / "bench" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else results_dir / f"layouts_{args.build}_{args.label + '_' if args.label else ''}{stamp}.json"
    work = Path(tempfile.mkdtemp(prefix="vvl-bench-"))
    log_dir = results_dir / f"logs_{stamp}"
    log_dir.mkdir(exist_ok=True)

    print(f"Model {build.label}; {len(images)} images; engine {install['variant']} ({install['backend']}); cards {gpus or 'default'}")
    collages, collage_seconds = build_collages(images, work)
    print(f"Label collages: {collage_seconds:.1f} s total, {collage_seconds / len(images):.2f} s per image "
          f"({len(images) / collage_seconds * 60:.0f} images/min, one at a time on the CPU)")

    _key, ocr_prompt = ocr_prompt_text(None)
    builder = PromptBuilder(args.prompt)
    schema = builder.json_schema()
    template = dict(builder.JSON_dict_structure)

    summary = {"build": build.id, "images": len(images), "engine_variant": install["variant"], "gpus": gpus,
               "collage_seconds_per_image": collage_seconds / len(images), "layouts": []}
    reference_ocr: dict[int, str] | None = None

    for layout in layouts:
        copies, slots = layout
        name = f"{copies}x{slots}"
        baseline = gpu_memory_used()
        instances = []
        row = {"layout": name, "copies": copies, "slots": slots}
        try:
            instances, infer = start_instances(layout, gpus, family, build, install, models_dir, report, log_dir,
                                               args.context_size)
            load_seconds = wait_ready(instances)
            after = gpu_memory_used()
            vram = {g: after.get(g, 0) - baseline.get(g, 0) for g in after}
            engines = [make_engine(inst, build, infer) for inst in instances]

            # One untimed request per copy so first-use costs do not count.
            for engine in engines:
                engine.ocr(OCRRequest(image_path=str(collages[0]), prompt_text=ocr_prompt, prompt_key="warmup"))

            def do_ocr(engine, collage):
                return engine.ocr(OCRRequest(image_path=str(collage), prompt_text=ocr_prompt, prompt_key="bench"))

            ocr_results, ocr_wall, ocr_durations = run_stage(instances, engines, collages, do_ocr)
            ocr_errors = [r for r in ocr_results.values() if isinstance(r, Exception)]
            texts = {i: ("" if isinstance(r, Exception) else r.text) for i, r in ocr_results.items()}

            def do_parse(engine, text):
                return engine.parse(ParseRequest(prompt_text=builder.parse_prompt_text(text), template=dict(template),
                                                 json_schema=schema))

            ordered_texts = [texts[i] for i in range(len(collages))]
            parse_results, parse_wall, parse_durations = run_stage(instances, engines, ordered_texts, do_parse)
            parse_errors = [r for r in parse_results.values() if isinstance(r, Exception)]
            bad_records = sum(1 for r in parse_results.values() if not isinstance(r, Exception) and r.record is None)
            peak = gpu_memory_used()
            peak_vram = {g: peak.get(g, 0) - baseline.get(g, 0) for g in peak}

            if reference_ocr is None:
                reference_ocr = dict(texts)
            same_ocr = sum(1 for i in texts if texts[i] == reference_ocr.get(i))

            n = len(collages)
            ocr_tokens_out = [r.tokens_out for r in ocr_results.values() if not isinstance(r, Exception)]
            ocr_tokens_in = [r.tokens_in for r in ocr_results.values() if not isinstance(r, Exception)]
            parse_tokens_out = [r.tokens_out for r in parse_results.values() if not isinstance(r, Exception)]
            row.update({
                "load_seconds": round(load_seconds, 1),
                "vram_mib_after_load": vram, "vram_mib_peak": peak_vram,
                "ocr_wall_seconds": round(ocr_wall, 2), "ocr_images_per_min": round(n / ocr_wall * 60, 1),
                "ocr_seconds_per_request": round(statistics.mean(ocr_durations), 2),
                "parse_wall_seconds": round(parse_wall, 2), "parse_images_per_min": round(n / parse_wall * 60, 1),
                "parse_seconds_per_request": round(statistics.mean(parse_durations), 2),
                "total_images_per_min": round(n / (ocr_wall + parse_wall) * 60, 1),
                "ocr_errors": len(ocr_errors), "parse_errors": len(parse_errors), "empty_records": bad_records,
                "ocr_identical_to_first_layout": same_ocr,
                "ocr_tokens_in_mean": round(statistics.mean(ocr_tokens_in), 0) if ocr_tokens_in else None,
                "ocr_tokens_out_mean": round(statistics.mean(ocr_tokens_out), 0) if ocr_tokens_out else None,
                "parse_tokens_out_mean": round(statistics.mean(parse_tokens_out), 0) if parse_tokens_out else None,
                "first_error": str((ocr_errors + parse_errors)[0])[:300] if (ocr_errors or parse_errors) else None,
            })
            print(f"{name:>6}: load {load_seconds:5.1f} s | VRAM {sum(v for v in peak_vram.values() if v > 0) / 1024:5.1f} GB | "
                  f"OCR {row['ocr_images_per_min']:6.1f}/min ({row['ocr_seconds_per_request']:.1f} s each) | "
                  f"parse {row['parse_images_per_min']:6.1f}/min ({row['parse_seconds_per_request']:.1f} s each) | "
                  f"both {row['total_images_per_min']:6.1f}/min | errors {len(ocr_errors)}+{len(parse_errors)}, "
                  f"empty {bad_records}, same OCR {same_ocr}/{n}", flush=True)
        except Exception as exc:  # noqa: BLE001
            row["failed"] = f"{type(exc).__name__}: {exc}"[:600]
            print(f"{name:>6}: FAILED {row['failed'][:200]}", flush=True)
        finally:
            stop_instances(instances)
            time.sleep(2.0)
        summary["layouts"].append(row)
        out_path.write_text(json.dumps(summary, indent=2))

    print(f"Results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
