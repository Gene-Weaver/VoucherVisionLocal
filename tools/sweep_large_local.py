"""Download the large local builds and run the Jouvea test on each at three reasoning settings.

    uv run python tools/sweep_large_local.py            download what is missing, then test (resumable)
    uv run python tools/sweep_large_local.py --summary  rewrite RESULTS.md from what is on disk

Builds: Ling 3.0 Flash VL (IQ4_XS, Q4) and Nemotron 3.5 Super VL (Q3, Q4). Each is
downloaded through the app's own downloader (queue, resume, sha256 check) into the models
folder, then run through the app (label collage -> OCR -> parsing with the same model,
prompt SLTPvM_full.yaml, long output, high image detail, automatic context) with reasoning
off, minimal (the lowest level) and max (the highest level).

PASS = formatted_json.scientificName contains "Jouvea". "OCR pass" = the OCR text
contains "jouvea pilosa".
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import sweep_models as sm  # noqa: E402

BUILDS = ["ling-3.0-flash-vl-iq4xs", "ling-3.0-flash-vl-q4", "nemotron-3.5-super-vl-q3", "nemotron-3.5-super-vl-q4"]
SETTINGS = [("off", "off", None), ("minimal", "on", "low"), ("max", "on", "high")]
IMAGE = REPO / "bench" / "extra_images" / "MICH_16205594_Poaceae_Jouvea_pilosa_full.jpg"
OUT = REPO / "bench" / "sweep_large_local_2026-11-20"
MODELS_DIR = sm.DEFAULT_MODELS_DIR


def log(text: str) -> None:
    print(f"{dt.datetime.now().strftime('%H:%M:%S')} {text}", flush=True)


def find_build(server, build_id: str) -> tuple[dict, dict]:
    catalog = server.call("GET", "/v1/models")
    for family in catalog["local"]["families"]:
        for build in family["builds"]:
            if build["id"] == build_id:
                return family, build
    raise RuntimeError(f"{build_id} is not in the registry")


def download(server, build_id: str) -> bool:
    """Download one build through the app. True when it is installed."""
    _family, build = find_build(server, build_id)
    if build["install_state"] == "installed":
        log(f"{build_id}: already downloaded")
        return True
    total = build["total_size_bytes"]
    log(f"{build_id}: downloading {total / 1e9:.1f} GB")
    job = server.call("POST", f"/v1/models/local/{build_id}/download", json={"confirm_size_bytes": total})["job"]
    last = 0.0
    began = time.monotonic()
    while True:
        time.sleep(20)
        try:
            job = server.call("GET", f"/v1/jobs/{job['id']}")
            job = job.get("job", job)
        except Exception as exc:  # noqa: BLE001 - a busy server is not a failed download
            log(f"{build_id}: job status unavailable ({exc})")
            continue
        state = job.get("state")
        if state == "done":
            log(f"{build_id}: downloaded and verified in {(time.monotonic() - began) / 60:.0f} min")
            return True
        if state in ("error", "canceled"):
            log(f"{build_id}: download {state}: {json.dumps(job.get('error') or job.get('message'))[:400]}")
            return False
        if time.monotonic() - last > 300:
            last = time.monotonic()
            done = job.get("bytes_done") or 0
            rate = done / max(1.0, time.monotonic() - began) / 1e6
            log(f"{build_id}: {state} {done / 1e9:.1f} of {total / 1e9:.1f} GB (about {rate:.0f} MB/s)")


def warm(build: dict, hf_repo: str) -> None:
    """Read the model files once so the engine loads them from memory, not the hard drive."""
    folder = Path(MODELS_DIR) / "models" / Path(*hf_repo.split("/"))
    for item in build["files"]:
        path = folder / item["path"]
        if path.exists():
            subprocess.run(["cat", str(path)], stdout=subprocess.DEVNULL, check=False)


def summarize() -> str:
    rows = []
    for build_id in BUILDS:
        for name, _reasoning, _strength in SETTINGS:
            cdir = OUT / f"{build_id}__{name}"
            meta = json.loads((cdir / "config.json").read_text()) if (cdir / "config.json").exists() else {}
            path = cdir / f"{IMAGE.stem}.json"
            res = json.loads(path.read_text()) if path.exists() else None
            if res is None:
                state = "FAILED TO LOAD" if meta.get("failed") else "not run"
                rows.append((build_id, name, state, "", "", "", "", "", (meta.get("failed") or "")[:160]))
                continue
            ocr_pass = "jouvea pilosa" in " ".join(str(res.get("ocr_text") or "").lower().split())
            speed = (meta.get("engine_speed") or {}).get("gen_tokens_per_s_median")
            notes = [n for n, on in (("OCR hit the token limit", res.get("ocr_hit_limit")),
                                     ("parsing hit the token limit", res.get("parse_hit_limit")),
                                     ("empty record", res.get("record_empty"))) if on]
            if not res.get("ok"):
                notes.append(str(res.get("error"))[:160])
            rows.append((
                build_id, name, "PASS" if res.get("jouvea_pass") else ("FAIL" if res.get("ok") else "ERROR"),
                "PASS" if ocr_pass else "FAIL", res.get("jouvea_scientificName") or "",
                f"{res.get('ocr_seconds', '-')} + {res.get('parse_seconds', '-')} s",
                f"{res.get('ocr_tokens_out', '-')} / {res.get('parse_tokens_out', '-')} "
                f"(thinking {res.get('ocr_thinking_tokens', '-')} / {res.get('parse_thinking_tokens', '-')})",
                f"{speed} tok/s; load {meta.get('load_seconds', '-')} s; ctx "
                f"{', '.join(str(e.get('context_size')) for e in meta.get('engines') or [])}; GPU peak MiB {meta.get('gpu_peak_mib')}",
                "; ".join(notes)))
    head = ("| Build | Reasoning | Jouvea test | Jouvea OCR | scientificName | OCR + parse time | Tokens out, OCR / parse "
            "| Engine | Notes |\n|---|---|---|---|---|---|---|---|---|\n")
    text = (f"# Large local models: Jouvea test\n\nUpdated {dt.datetime.now().isoformat(timespec='seconds')}. "
            f"Image {IMAGE.name}; prompt {sm.PROMPT}; long output; high image detail; same model for OCR and parsing.\n\n"
            + head + "".join("| " + " | ".join(str(c).replace("|", "/") for c in row) + " |\n" for row in rows))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULTS.md").write_text(text)
    (OUT / "results.json").write_text(json.dumps([dict(zip(
        ("build", "reasoning", "jouvea_test", "jouvea_ocr", "scientificName", "time", "tokens", "engine", "notes"), r))
        for r in rows], indent=1))
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--only", default="", help="comma list of build ids")
    parser.add_argument("--home", default=str(Path.home() / ".local" / "share" / "VoucherVisionLocal-largelocal"))
    parser.add_argument("--port", type=int, default=8676)
    args = parser.parse_args()
    if args.summary:
        print(summarize())
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    builds = [b for b in BUILDS if not args.only or b in args.only.split(",")]
    server = sm.Server(Path(args.home), MODELS_DIR, args.port)
    log(f"Starting the sweep server on port {server.port}")
    server.start()
    try:
        index = 0
        for build_id in builds:
            try:
                ok = download(server, build_id)
            except Exception as exc:  # noqa: BLE001
                log(f"{build_id}: download failed to start: {exc}")
                ok = False
            if not ok:
                index += len(SETTINGS)
                continue
            family, build = find_build(server, build_id)
            log(f"{build_id}: reading the files into memory")
            warm(build, build["hf_repo"])
            for name, reasoning, strength in SETTINGS:
                index += 1
                cfg = {"build_id": build_id, "model_id": f"local:{build_id}", "label": build["label"],
                       "family": family["id"], "family_label": family["label"], "size_bytes": build["total_size_bytes"],
                       "reasoning_supported": True, "key": f"{build_id}__{name}", "reasoning": reasoning,
                       "strength": strength, "reasoning_note": name, "index": index}
                try:
                    sm.run_config(server, cfg, [IMAGE], OUT, retry_errors=True)
                except Exception as exc:  # noqa: BLE001
                    log(f"{cfg['key']}: crashed: {exc}")
                    try:
                        server.call("POST", "/v1/engine/stop")
                    except Exception:  # noqa: BLE001
                        pass
                summarize()
    finally:
        server.stop()
        summarize()
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
