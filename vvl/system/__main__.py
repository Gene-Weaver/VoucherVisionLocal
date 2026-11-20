"""``python -m vvl.system [--summary] [--fit] [--models-dir PATH]``"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import fit, get_report
from ._util import fmt_gb

REGISTRY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "registry", "models_local.yaml")


def _yn(v):
    return "yes" if v is True else "no" if v is False else "unknown"


def format_summary(r):
    lines = [f"VoucherVisionLocal system check ({r.get('duration_ms')} ms, {r.get('generated_at')})"]
    o, c, m, d = r.get("os") or {}, r.get("cpu") or {}, r.get("memory") or {}, r.get("disk") or {}
    extra = []
    if o.get("is_wsl"):
        extra.append(f"WSL {o.get('wsl_version') or ''}".strip())
    if o.get("is_apple_silicon"):
        extra.append("Apple Silicon")
    if o.get("is_translated"):
        extra.append("translated Python")
    lines.append(f"OS:       {o.get('name')} ({o.get('system')} {o.get('kernel')}, {o.get('arch')}"
                 f"{', ' + ', '.join(extra) if extra else ''}), Python {o.get('python')}")
    f = c.get("features") or {}
    flags = ", ".join(f"{k.upper()} {_yn(f.get(k))}" for k in ("avx2", "avx512", "neon") if k in f)
    lines.append(f"CPU:      {c.get('model')}, {c.get('physical_cores')} cores / {c.get('logical_cores')} threads"
                 f"{', max ' + str(c.get('max_mhz')) + ' MHz' if c.get('max_mhz') else ''}; {flags}")
    lines.append(f"Memory:   {fmt_gb(m.get('total_bytes'))} total, {fmt_gb(m.get('available_bytes'))} available, "
                 f"swap {fmt_gb(m.get('swap_total_bytes'))}")
    gpus = r.get("gpus") or []
    if not gpus:
        lines.append("GPUs:     none found")
    for g in gpus:
        mem = fmt_gb(g.get("vram_total_bytes")) if g.get("vram_total_bytes") else "memory unknown"
        if g.get("vram_free_bytes") is not None and g.get("kind") != "unified":
            mem += f" ({fmt_gb(g['vram_free_bytes'])} free)"
        drv = f", driver {g['driver_version']}" if g.get("driver_version") else ""
        lines.append(f"GPU {g.get('index')}:    {g.get('name')} [{g.get('vendor')}, {g.get('kind') or 'kind unknown'}], "
                     f"{mem}{drv} (via {g.get('source')})")
    for name, label in (("cuda", "CUDA"), ("metal", "Metal"), ("vulkan", "Vulkan")):
        a = (r.get("accelerators") or {}).get(name) or {}
        lines.append(f"{label + ':':<10}{_yn(a.get('available'))} - {a.get('detail')}")
    lines.append(f"Models:   {d.get('models_dir')} ({'exists' if d.get('exists') else 'missing'}, from "
                 f"{d.get('models_dir_source')}); {fmt_gb(d.get('free_bytes'))} free of {fmt_gb(d.get('total_bytes'))}"
                 f" at {d.get('checked_path')}"
                 f"{'; synced: ' + str(d.get('synced_folder_hint')) if d.get('is_synced_folder') else ''}")
    cap = r.get("capability") or {}
    lines.append(f"Backend:  {cap.get('recommended_backend')} - {cap.get('backend_reason')}")
    lines.append(f"Tier:     {cap.get('tier')}; GPU budget {fmt_gb(cap.get('gpu_memory_budget_bytes'))}, "
                 f"CPU budget {fmt_gb(cap.get('cpu_memory_budget_bytes'))}")
    lines.append(f"Summary:  {cap.get('summary')}")
    warns = r.get("warnings") or []
    if not warns:
        lines.append("Warnings: none")
    for w in warns:
        lines.append(f"Warning:  [{w.get('severity', 'warning')}] {w.get('code')}: {w.get('message')}")
    probes = ", ".join(f"{p.get('name')} {p.get('duration_ms')} ms" + ("" if p.get("ok") else f" FAILED ({p.get('error')})")
                       for p in r.get("probes") or [])
    lines.append(f"Probes:   {probes}")
    return "\n".join(lines)


def _load_builds(path):
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    out = []
    for fam in data.get("families") or []:
        for b in fam.get("builds") or []:
            out.append(b)
    return out


def format_fit(r, builds):
    lines = []
    for b in builds:
        res = fit.assess(r, b)
        lines.append(f"{b.get('id', '?'):<22} {res['verdict']:<11} {fmt_gb(res['required_bytes']):>7}  {res['detail']}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m vvl.system", description="VoucherVisionLocal system report")
    ap.add_argument("--summary", action="store_true", help="print a short human-readable summary")
    ap.add_argument("--fit", action="store_true", help="print the fit verdict for every local model build")
    ap.add_argument("--models-dir", default=None, help="check this models folder instead of the configured one")
    args = ap.parse_args(argv)
    report = get_report(refresh=True, models_dir=args.models_dir)
    if args.summary or args.fit:
        if args.summary:
            print(format_summary(report))
        if args.fit:
            if args.summary:
                print()
            try:
                print(format_fit(report, _load_builds(REGISTRY)))
            except Exception as exc:
                print(f"Could not read {REGISTRY}: {exc}", file=sys.stderr)
                return 1
    else:
        print(json.dumps(report, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
