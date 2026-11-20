"""Compare the local-model sweep with the Gemini (VoucherVisionGO) results on the same bench images.

Developer tool, not part of the app. Writes one self-contained page,
bench/local_vs_gemini.html (no network access), from:

  bench/sweep_2026-11-19/            local sweep (tools/sweep_models.py)
  bench/vvgo-20img-comparison/       VVGO batch downloads (zips are extracted to _extracted/)
  VoucherVisionGO/webpage/index.html "Thinking Budget Testing" table (the Gemini Jouvea test),
                                     cached to bench/vvgo_thinking_budget_jouvea.json

Gemini is the baseline. There is no hand-made ground truth for the 20 bench
sheets, so field values are scored by AGREEMENT with Gemini 3.1 Pro and
Gemini 3.8 Flash (the two baselines that cover all 20 sheets); each Gemini
model is scored against the other Gemini models, never against itself.
Two checks have a known answer and apply to every model, Gemini included:

  name check   the parsed genus / specific epithet equal the GBIF name in the
               image file name (<inst>_<gbifID>_<Family>_<Genus>_<epithet>)
  Jouvea test  VVGO's rule: formatted_json.scientificName contains "Jouvea"

Usage (from the repo root):  uv run python tools/compare_local_vs_gemini.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import html
import json
import re
import statistics
import unicodedata
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "bench"

KEY_FIELDS = ["catalogNumber", "scientificName", "genus", "specificEpithet", "collectedBy", "collectorNumber",
              "verbatimCollectionDate", "collectionDate", "country", "stateProvince", "county", "locality"]

GEMINI = {
    "gemini-3.1-pro": {"label": "Gemini 3.1 Pro", "level": "low"},
    "gemini-3.8-flash": {"label": "Gemini 3.8 Flash", "level": "low"},
    "gemini-3.1-flash-lite": {"label": "Gemini 3.1 Flash-Lite", "level": "high"},
}
FULL_BASELINES = ["gemini-3.1-pro", "gemini-3.8-flash"]
# Batch folder -> the model the batch was meant to use (a file's real model comes from its own JSON).
BATCH_INTENT = {"3-1-pro": "gemini-3.1-pro", "3-8-flash": "gemini-3.8-flash", "3-1-flash-lite": "gemini-3.1-flash-lite"}


# ---------------------------------------------------------------- strings --

def norm(value) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"[^0-9a-z]+", " ", text)
    return " ".join(text.split())


def similarity(a, b) -> float | None:
    """Agreement of two field values in [0, 1]; None when both are empty (not scored)."""
    x, y = norm(a), norm(b)
    if not x and not y:
        return None
    if not x or not y:
        return 0.0
    if x == y:
        return 1.0
    xs, ys = " ".join(sorted(x.split())), " ".join(sorted(y.split()))
    return max(difflib.SequenceMatcher(None, x, y, autojunk=False).ratio(),
               difflib.SequenceMatcher(None, xs, ys, autojunk=False).ratio())


def record_agreement(cand: dict | None, base: dict | None, fields: list[str]) -> dict:
    """Mean similarity over fields where either side has a value, plus per-field values."""
    cand, base = cand or {}, base or {}
    per = {}
    for f in fields:
        s = similarity(cand.get(f), base.get(f))
        if s is not None:
            per[f] = s
    return {"mean": statistics.mean(per.values()) if per else None, "per": per}


def ocr_word_f1(cand: str, base: str) -> float | None:
    a = [w for w in norm(cand).split() if len(w) > 1]
    b = [w for w in norm(base).split() if len(w) > 1]
    if not b:
        return None
    if not a:
        return 0.0
    from collections import Counter

    ca, cb = Counter(a), Counter(b)
    hit = sum((ca & cb).values())
    p, r = hit / sum(ca.values()), hit / sum(cb.values())
    return 0.0 if hit == 0 else 2 * p * r / (p + r)


def name_from_filename(name: str) -> tuple[str, str] | None:
    parts = Path(name).stem.split("_")
    if len(parts) < 5:
        return None
    return parts[3], parts[4]


def name_check(record: dict | None, image: str) -> float | None:
    truth = name_from_filename(image)
    if not truth:
        return None
    genus, epithet = (norm(t) for t in truth)
    record = record or {}
    sci = norm(record.get("scientificName")).split()
    g = norm(record.get("genus")) or (sci[0] if sci else "")
    e = norm(record.get("specificEpithet")) or (sci[1] if len(sci) > 1 else "")
    return ((g == genus) + (e == epithet)) / 2


# ------------------------------------------------------------------ Gemini --

def extract_gemini(root: Path) -> None:
    for batch in BATCH_INTENT:
        folder = root / batch
        for kind, pattern in (("full", "image_batch_full_json_results_*.zip"), ("json", "image_batch_json_results_*.zip")):
            target = root / "_extracted" / batch / kind
            if target.exists() and any(target.iterdir()):
                continue
            for z in folder.glob(pattern):
                target.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(z) as archive:
                    for member in archive.namelist():
                        safe = Path(member).name          # flat names only
                        if safe.endswith(".json"):
                            (target / safe).write_bytes(archive.read(member))


def load_gemini(root: Path) -> dict:
    """{model: {image: [run, ...]}}, primary run first (the batch meant for that model)."""
    extract_gemini(root)
    out: dict[str, dict[str, list]] = {m: {} for m in GEMINI}
    attribution = []
    for batch, intended in BATCH_INTENT.items():
        for path in sorted((root / "_extracted" / batch / "full").glob("*.json")):
            d = json.loads(path.read_text())
            ocr_models = list((d.get("ocr_info") or {}).keys())
            parse_model = (d.get("parsing_info") or {}).get("model")
            model = parse_model if parse_model in GEMINI else (ocr_models[0] if ocr_models else None)
            if model not in GEMINI:
                continue
            info = (d.get("ocr_info") or {}).get(model) or {}
            parsing = d.get("parsing_info") or {}
            run = {
                "batch": batch, "intended": intended, "model": model,
                "ocr_level": info.get("thinking_budget_requested"), "parse_level": parsing.get("thinking_budget_requested"),
                "record": d.get("formatted_json") or {}, "ocr": d.get("ocr") or "",
                "ocr_tokens_out": info.get("tokens_out"), "ocr_thinking": info.get("thinking_tokens"),
                "parse_tokens_out": parsing.get("output"), "parse_thinking": parsing.get("thinking_tokens"),
                "cost_usd": d.get("total_request_cost_usd"),
            }
            image = d.get("filename") or (path.stem + ".jpg")
            runs = out[model].setdefault(image, [])
            if intended == model:
                runs.insert(0, run)
            else:
                runs.append(run)
            attribution.append({"batch": batch, "image": image, "model": model, "intended": intended})
    return {"runs": out, "attribution": attribution}


def old_ocr_sweep(folder: Path, cache: Path) -> dict:
    """VoucherVision's earlier OCR-only Jouvea sweep (Temperature x Top K x Top P), summarized per CSV."""
    if cache.exists():
        return json.loads(cache.read_text())
    import csv

    rows = []
    for f in sorted(folder.glob("*_parameter_sweep_results_*.csv")):
        if f.stem.endswith("_fixed"):
            continue
        with open(f, encoding="utf-8", errors="replace") as handle:
            data = list(csv.DictReader(handle))
        ok = sum(int(float(r.get("Success Count") or 0)) for r in data)
        bad = sum(int(float(r.get("Fail Count") or 0)) for r in data)
        combos = {(r.get("Temperature"), r.get("Top K"), r.get("Top P")) for r in data}
        model, _, variant = f.stem.partition("_parameter_sweep_results_")
        model = model.replace("_OCR", "")
        rows.append({"model": model, "variant": variant, "combinations": len(combos), "calls": ok + bad,
                     "success": ok, "rate": ok / (ok + bad) if ok + bad else None})
    out = {"source": str(folder), "rule": "success = the OCR text contains 'jouvea pilosa' (OCR_Gemini.py)", "rows": rows}
    if rows:
        cache.write_text(json.dumps(out, indent=1))
    return out


def thinking_budget_table(index_html: Path, cache: Path) -> list[dict]:
    """Rows of VVGO's 'Thinking Budget Testing' table (cached as JSON next to the bench data)."""
    if cache.exists():
        return json.loads(cache.read_text())
    if not index_html.exists():
        return []
    text = index_html.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'<table class="thinking-budget-table">(.*?)</table>', text, re.S)
    if not m:
        return []
    heads = [html.unescape(re.sub(r"<[^>]+>", "", h)).strip() for h in re.findall(r"<th[^>]*>(.*?)</th>", m.group(1), re.S)]
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", m.group(1), re.S):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if len(cells) == len(heads):
            rows.append(dict(zip(heads, cells)))
    cache.write_text(json.dumps({"source": str(index_html), "copied": dt.date.today().isoformat(), "rows": rows}, indent=1))
    return json.loads(cache.read_text())


# ------------------------------------------------------------------- local --

def load_local(sweep: Path) -> tuple[list[dict], dict]:
    summary = json.loads((sweep / "summary.json").read_text()) if (sweep / "summary.json").exists() else {"rows": []}
    rows = summary.get("rows") or []
    records = {}
    for row in rows:
        cdir = sweep / row["key"]
        per = {}
        for p in sorted(cdir.glob("*.json")):
            if p.name == "config.json":
                continue
            try:
                d = json.loads(p.read_text())
            except ValueError:
                continue
            per[d.get("image") or p.stem + ".jpg"] = d
        records[row["key"]] = per
    return rows, {"header": summary.get("header"), "updated": summary.get("updated"), "records": records}


# ------------------------------------------------------------------ scoring --

def score_candidate(get_record, get_ocr, images: list[str], gem: dict, exclude: str | None) -> dict:
    """Agreement of one candidate (callable image -> record/OCR) with each Gemini baseline."""
    fields_all = None
    by_base = {}
    per_field_sum: dict[str, list[float]] = {}
    for base in GEMINI:
        if base == exclude:
            continue
        key_scores, all_scores, ocr_scores, n = [], [], [], 0
        for image in images:
            runs = gem["runs"][base].get(image)
            if not runs:
                continue
            brec = runs[0]["record"]
            if fields_all is None and brec:
                fields_all = list(brec.keys())
            crec, cocr = get_record(image), get_ocr(image)
            if crec is None and cocr is None:
                continue                       # not run yet
            n += 1
            ka = record_agreement(crec, brec, KEY_FIELDS)
            aa = record_agreement(crec, brec, fields_all or list(brec.keys()))
            if ka["mean"] is not None:
                key_scores.append(ka["mean"])
            if aa["mean"] is not None:
                all_scores.append(aa["mean"])
            o = ocr_word_f1(cocr or "", runs[0]["ocr"])
            if o is not None:
                ocr_scores.append(o)
            if base in FULL_BASELINES:
                for f, s in aa["per"].items():
                    per_field_sum.setdefault(f, []).append(s)
        by_base[base] = {"images": n,
                         "key": statistics.mean(key_scores) if key_scores else None,
                         "all": statistics.mean(all_scores) if all_scores else None,
                         "ocr_f1": statistics.mean(ocr_scores) if ocr_scores else None}
    full = [by_base[b] for b in FULL_BASELINES if b in by_base and by_base[b]["key"] is not None]
    if not full:      # flash-lite: score against the full baselines only
        full = [v for v in by_base.values() if v["key"] is not None]
    agg = {k: (statistics.mean(v[k] for v in full if v[k] is not None) if any(v[k] is not None for v in full) else None)
           for k in ("key", "all", "ocr_f1")}
    names = [name_check(get_record(im), im) for im in images if get_record(im) is not None]
    names = [x for x in names if x is not None]
    return {"by_base": by_base, "agree_key": agg["key"], "agree_all": agg["all"], "ocr_f1": agg["ocr_f1"],
            "name_check": statistics.mean(names) if names else None, "name_images": len(names),
            "per_field": {f: statistics.mean(v) for f, v in per_field_sum.items()}}


def overall(row: dict) -> float | None:
    parts = [row.get("agree_key"), row.get("agree_all"), row.get("name_check")]
    if any(p is None for p in parts):
        return None
    return statistics.mean(parts)


def build(args) -> dict:
    sweep = Path(args.sweep)
    gem = load_gemini(Path(args.gemini))
    budget = thinking_budget_table(Path(args.vvgo_index), BENCH / "vvgo_thinking_budget_jouvea.json")
    old_sweep = old_ocr_sweep(Path(args.old_sweep), BENCH / "vv_ocr_parameter_sweep_jouvea.json")
    local_rows, local = load_local(sweep)
    images = sorted({im for m in FULL_BASELINES for im in gem["runs"][m]})       # the 20 bench sheets
    field_order = None
    for runs in gem["runs"]["gemini-3.1-pro"].values():
        field_order = list(runs[0]["record"].keys())
        break

    ranking = []
    # Gemini models (each against the other Gemini models).
    for model, meta in GEMINI.items():
        runs = gem["runs"][model]
        if not runs:
            continue
        sc = score_candidate(lambda im, r=runs: (r.get(im) or [{}])[0].get("record") if r.get(im) else None,
                             lambda im, r=runs: (r.get(im) or [{}])[0].get("ocr") if r.get(im) else None,
                             images, gem, exclude=model)
        costs = [r[0]["cost_usd"] for r in runs.values() if r and isinstance(r[0].get("cost_usd"), (int, float))]
        empty = sum(1 for r in runs.values() if r and not any(norm(v) for v in r[0]["record"].values()))
        row = {"kind": "gemini", "key": model, "label": f"{meta['label']} ({meta['level']})", "model": meta["label"],
               "reasoning": meta["level"], "images_ok": sum(1 for im in images if runs.get(im)), "images_total": len(images),
               "empty_records": empty, "cost_per_image_usd": statistics.mean(costs) if costs else None, **sc}
        jt = [r for r in (budget.get("rows") if isinstance(budget, dict) else []) if r.get("Model") == model
              and r.get("Level", "").lower() == meta["level"]]
        row["jouvea"] = {"pass": jt[0].get("Jouvea Test") == "PASS", "source": "VVGO Thinking Budget Testing"} if jt else None
        row["overall"] = overall(row)
        ranking.append(row)
    # Repeat-run consistency for 3.8 Flash (the noise floor of the agreement measure).
    repeats = []
    for image, runs in gem["runs"]["gemini-3.8-flash"].items():
        if len(runs) > 1 and image in images:
            repeats.append({"key": record_agreement(runs[1]["record"], runs[0]["record"], KEY_FIELDS)["mean"],
                            "all": record_agreement(runs[1]["record"], runs[0]["record"], field_order or KEY_FIELDS)["mean"]})
    consistency = {"images": len(repeats),
                   "key": statistics.mean(r["key"] for r in repeats if r["key"] is not None) if repeats else None,
                   "all": statistics.mean(r["all"] for r in repeats if r["all"] is not None) if repeats else None}

    # Local configurations.
    for lr in local_rows:
        recs = local["records"].get(lr["key"]) or {}

        def get_record(im, recs=recs):
            r = recs.get(im)
            if r is None:
                return None
            return (r.get("record") or {}) if r.get("ok") else {}

        def get_ocr(im, recs=recs):
            r = recs.get(im)
            if r is None:
                return None
            return r.get("ocr_text") or ""

        sc = score_candidate(get_record, get_ocr, images, gem, exclude=None)
        jouvea_rec = next((r for im, r in recs.items() if "jouvea" in im.lower()), None)
        row = {"kind": "local", "key": lr["key"], "label": f"{lr['label']} · {'off' if lr['reasoning'] == 'off' else ('max' if not lr.get('strength') else lr['strength'])}",
               "model": lr["label"], "build_id": lr["build_id"], "family": lr.get("family_label"),
               "reasoning": lr["reasoning"], "strength": lr.get("strength"), "reasoning_note": lr.get("reasoning_note"),
               "state": lr.get("state"), "failed": lr.get("failed"),
               "images_ok": sum(1 for im in images if (recs.get(im) or {}).get("ok")),
               "images_done": sum(1 for im in images if im in recs), "images_total": len(images),
               "errors": sum(1 for im in images if im in recs and not recs[im].get("ok")),
               "empty_records": sum(1 for im in images if (recs.get(im) or {}).get("record_empty")),
               "ocr_hit_limit": sum(1 for im in images if (recs.get(im) or {}).get("ocr_hit_limit")),
               "parse_hit_limit": sum(1 for im in images if (recs.get(im) or {}).get("parse_hit_limit")),
               "images_per_min": lr.get("images_per_min"), "seconds_per_image": lr.get("total_seconds_mean"),
               "ocr_seconds": lr.get("ocr_seconds_mean"), "parse_seconds": lr.get("parse_seconds_mean"),
               "ocr_tokens_out": lr.get("ocr_tokens_out_mean"), "parse_tokens_out": lr.get("parse_tokens_out_mean"),
               "batch_minutes": (lr["batch_wall_seconds"] / 60) if lr.get("batch_wall_seconds") else None,
               "size_gb": lr["size_bytes"] / 1e9 if lr.get("size_bytes") else None,
               "gen_tps": (lr.get("engine_speed") or {}).get("gen_tokens_per_s_median"),
               "jouvea": ({"pass": bool(jouvea_rec.get("jouvea_pass")), "name": jouvea_rec.get("jouvea_scientificName"),
                           "ok": jouvea_rec.get("ok"), "ocr_tokens_out": jouvea_rec.get("ocr_tokens_out"),
                           "ocr_pass": "jouvea pilosa" in norm(jouvea_rec.get("ocr_text")),
                           "parse_tokens_out": jouvea_rec.get("parse_tokens_out"),
                           "seconds": jouvea_rec.get("total_seconds"), "error": jouvea_rec.get("error")}
                          if jouvea_rec else None),
               **sc}
        row["overall"] = overall(row) if row["images_done"] else None
        ranking.append(row)

    ranked = sorted([r for r in ranking if r.get("overall") is not None], key=lambda r: -r["overall"])
    for i, r in enumerate(ranked, 1):
        r["rank"] = i

    # Per-image records for the drill-down (Gemini primary runs + local records).
    per_image = {}
    for im in images + sorted({im for recs in local["records"].values() for im in recs if "jouvea" in im.lower()}):
        entry = {}
        for model in GEMINI:
            runs = gem["runs"][model].get(im)
            if runs:
                entry[model] = runs[0]["record"]
        for key, recs in local["records"].items():
            r = recs.get(im)
            if r is not None:
                entry[key] = (r.get("record") or {}) if r.get("ok") else {"_error": r.get("error", "error")}
        per_image[im] = entry

    return {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "sweep_header": local["header"], "sweep_updated": local["updated"],
        "configs_total": len(local_rows),
        "configs_done": sum(1 for r in local_rows if r.get("state") in ("done", "failed")),
        "images": images, "fields": field_order, "key_fields": KEY_FIELDS,
        "gemini_meta": GEMINI, "attribution": gem["attribution"], "consistency": consistency,
        "thinking_budget": budget, "old_ocr_sweep": old_sweep, "ranking": ranking, "per_image": per_image,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", default=str(BENCH / "sweep_2026-11-19"))
    parser.add_argument("--gemini", default=str(BENCH / "vvgo-20img-comparison"))
    parser.add_argument("--vvgo-index", default=str(Path.home() / "Dropbox" / "VoucherVisionGO" / "webpage" / "index.html"))
    parser.add_argument("--old-sweep", default=str(Path.home() / "Dropbox" / "VoucherVisionGO" / "vouchervision_main" / "OCR_vLM_Parameter_Sweep"))
    parser.add_argument("--out", default=str(BENCH / "local_vs_gemini.html"))
    args = parser.parse_args()
    data = build(args)
    template = (REPO / "tools" / "local_vs_gemini_template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    Path(args.out).write_text(template.replace("/*__DATA__*/null", payload), encoding="utf-8")
    done = data["configs_done"]
    print(f"Wrote {args.out} ({done}/{data['configs_total']} local configurations finished)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
