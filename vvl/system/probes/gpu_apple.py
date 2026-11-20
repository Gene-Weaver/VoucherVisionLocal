"""macOS GPUs through ``system_profiler SPDisplaysDataType -json``.

On Apple Silicon the GPU shares system memory (``kind: unified``):
``vram_total_bytes`` is the machine's RAM (``hw.memsize``) and the part the GPU
may use is worked out in ``capability.py`` (see ``policy.UNIFIED_*``). A
user-raised ``iogpu.wired_limit_mb`` is recorded in ``details`` so the budget
can follow it.
"""

from __future__ import annotations

import json
import re

from .. import _proc
from .._util import MiB, parse_size_text, to_int
from ..gpus import make_gpu

SYSTEM_PROFILER = "/usr/sbin/system_profiler"


def _vendor(entry):
    text = " ".join(str(entry.get(k) or "") for k in ("sppci_vendor", "spdisplays_vendor", "sppci_model", "_name"))
    low = text.lower()
    if "apple" in low:
        return "apple"
    if "nvidia" in low:
        return "nvidia"
    if "amd" in low or "ati" in re.findall(r"[a-z]+", low) or "radeon" in low:
        return "amd"
    if "intel" in low:
        return "intel"
    return "unknown"


def _metal(entry):
    for key in ("spdisplays_mtlgpufamilysupport", "spdisplays_metal", "spdisplays_metalfeatureset"):
        val = entry.get(key)
        if not val:
            continue
        m = re.search(r"metal(\d+)", str(val).lower())
        if m:
            return f"Metal {m.group(1)}"
        if "supported" in str(val).lower():
            return "Metal"
    return None


def parse_displays_json(text, *, apple_silicon, memsize=None, wired_limit_mb=None):
    """GPU entries from system_profiler JSON. Raises ValueError on bad JSON."""
    data = json.loads(text)
    items = data.get("SPDisplaysDataType") if isinstance(data, dict) else None
    gpus = []
    for entry in items or []:
        if not isinstance(entry, dict):
            continue
        dev_type = str(entry.get("sppci_device_type") or "")
        if dev_type and "gpu" not in dev_type.lower() and "display" not in dev_type.lower():
            continue
        name = entry.get("sppci_model") or entry.get("_name")
        vendor = _vendor(entry)
        bus = str(entry.get("sppci_bus") or "").lower()
        details = {
            "gpu_cores": to_int(entry.get("sppci_cores")),
            "metal_support": _metal(entry),
            "bus": entry.get("sppci_bus"),
            "pci_device_id": entry.get("spdisplays_device-id"),
        }
        if vendor == "apple" or (apple_silicon and "builtin" in bus):
            if wired_limit_mb:
                details["wired_limit_bytes"] = int(wired_limit_mb) * MiB
            details["unified_memory_bytes"] = memsize
            gpus.append(make_gpu("apple", name, kind="unified", vram_total=memsize, source="system_profiler",
                                 backend_hints=["metal"], details=details))
            continue
        vram = parse_size_text(entry.get("spdisplays_vram") or entry.get("_spdisplays_vram"))
        shared = parse_size_text(entry.get("spdisplays_vram_shared"))
        if "builtin" in bus or (vram is None and shared is not None):
            kind = "integrated"
        elif "pcie" in bus or "thunderbolt" in bus or "egpu" in bus:
            kind = "discrete"
        else:
            kind = "discrete" if vendor in ("amd", "nvidia") else None
        if shared is not None:
            details["shared_memory"] = True
        # llama.cpp's Metal backend is supported on Apple Silicon only; Intel
        # Macs (including their AMD GPUs) run models on the CPU.
        gpus.append(make_gpu(vendor, name, kind=kind, vram_total=vram if vram is not None else shared,
                             source="system_profiler", backend_hints=[], details=details))
    return gpus


def probe(ctx, log):
    from . import _macos

    out = {"gpus": [], "is_apple_silicon": None, "memsize": None}
    arm64, _translated = _macos.arch_info(ctx.run)
    memsize = _macos.sysctl_int("hw.memsize", ctx.run)
    wired = _macos.sysctl_int("iogpu.wired_limit_mb", ctx.run) or _macos.sysctl_int("debug.iogpu.wired_limit", ctx.run)
    out["is_apple_silicon"], out["memsize"] = arm64, memsize

    exe = _proc.find_tool("system_profiler", [SYSTEM_PROFILER]) or SYSTEM_PROFILER
    res = ctx.run([exe, "SPDisplaysDataType", "-json"], timeout=5.0)
    if res.ok and res.stdout.strip():
        try:
            out["gpus"] = parse_displays_json(res.stdout, apple_silicon=bool(arm64), memsize=memsize,
                                              wired_limit_mb=wired)
        except (ValueError, TypeError, AttributeError) as exc:
            log.fail(f"system_profiler output could not be parsed: {type(exc).__name__}")
    else:
        log.fail(res.describe_failure())

    if arm64 and not any(g["vendor"] == "apple" for g in out["gpus"]):
        # Every Apple Silicon Mac has a GPU; describe it from sysctl instead.
        brand = _macos.sysctl_str("machdep.cpu.brand_string", ctx.run)
        details = {"gpu_cores": None, "metal_support": None, "unified_memory_bytes": memsize}
        if wired:
            details["wired_limit_bytes"] = int(wired) * MiB
        out["gpus"].insert(0, make_gpu("apple", brand or "Apple Silicon GPU", kind="unified", vram_total=memsize,
                                       source="sysctl", backend_hints=["metal"], details=details))
    return out
