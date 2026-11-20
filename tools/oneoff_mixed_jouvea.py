"""One-off Jouvea test with different models for the two steps.

    uv run python tools/oneoff_mixed_jouvea.py <ocr build id> <parse build id> [--parse-reasoning on]

OCR runs at max reasoning (the highest level); parsing runs with reasoning off unless asked.
The two models are loaded one after the other when they do not fit in memory together.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import sweep_models as sm  # noqa: E402

IMAGE = REPO / "bench" / "extra_images" / "MICH_16205594_Poaceae_Jouvea_pilosa_full.jpg"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ocr")
    parser.add_argument("parse")
    parser.add_argument("--parse-reasoning", default="off", choices=("off", "on"))
    args = parser.parse_args()
    ocr_id, parse_id = f"local:{args.ocr}", f"local:{args.parse}"
    out_dir = REPO / "bench" / "sweep_large_local_2026-11-20" / f"oneoff__ocr_{args.ocr}_max__parse_{args.parse}_{args.parse_reasoning}"
    out_dir.mkdir(parents=True, exist_ok=True)
    server = sm.Server(Path.home() / ".local" / "share" / "VoucherVisionLocal-largelocal", sm.DEFAULT_MODELS_DIR, 8676)
    server.start()
    out = {"ocr_model": ocr_id, "parse_model": parse_id, "parse_reasoning": args.parse_reasoning}
    try:
        server.call("PUT", "/v1/selection", json={"ocr": [ocr_id], "parse": parse_id})
        gen = server.call("PUT", "/v1/generation", json={
            "ocr": {"reasoning": "on", "strength": "high", "output_length": "long", "image_detail": "high"},
            "parse": {"reasoning": args.parse_reasoning, "strength": "high", "output_length": "long"},
            "context": {ocr_id: "auto", parse_id: "auto"}})
        out["sent"] = {step: [(m["model_id"], m.get("sent")) for m in (gen.get(step) or {}).get("models", [])] for step in ("ocr", "parse")}
        common = {"engines": ocr_id, "llm_model": parse_id, "prompt": sm.PROMPT, "include_wfo": "false", "include_cop90": "false"}
        t0 = time.monotonic()
        with open(IMAGE, "rb") as handle:
            staged = server.call("POST", "/process", data=dict(common, stage="ocr"),
                                 files={"file": (IMAGE.name, handle, "image/jpeg")})
        t1 = time.monotonic()
        out["ocr_seconds"] = round(t1 - t0, 1)
        out["ocr_text"] = staged.get("ocr", "")
        info = next(iter((staged.get("ocr_info") or {}).values()), {}) or {}
        out["ocr_tokens_out"] = info.get("tokens_out")
        result = server.call("POST", "/v1/process/finish", json={
            "stage_token": staged["stage_token"], "llm_model": parse_id, "prompt": sm.PROMPT,
            "include_wfo": False, "include_cop90": False})
        out["parse_seconds"] = round(time.monotonic() - t1, 1)
        out["parsing_info"] = result.get("parsing_info")
        out["record"] = result.get("formatted_json")
        name = str((out["record"] or {}).get("scientificName") or "") if isinstance(out["record"], dict) else ""
        out["scientificName"] = name
        out["jouvea_pass"] = "jouvea" in name.lower()
        out["jouvea_ocr_pass"] = "jouvea pilosa" in " ".join(out["ocr_text"].lower().split())
        out["ok"] = True
    except Exception as exc:  # noqa: BLE001
        out["ok"] = False
        out["error"] = str(exc)[:1500]
    finally:
        try:
            server.call("POST", "/v1/engine/stop")
        except Exception:  # noqa: BLE001
            pass
        server.stop()
    (out_dir / "result.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k not in ("record", "ocr_text", "sent")}, indent=1))
    print("--- OCR text ---\n" + str(out.get("ocr_text", ""))[:1800])
    if isinstance(out.get("record"), dict):
        print("--- record (filled fields) ---\n" + json.dumps({k: v for k, v in out["record"].items() if v}, indent=1)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
