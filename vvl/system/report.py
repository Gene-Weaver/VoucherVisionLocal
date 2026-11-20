"""Build, cache and return the system report (schema 1, docs/API.md GET /v1/system)."""

from __future__ import annotations

import copy
import os
import platform
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from . import capability, gpus as gpus_mod, models_dir as models_dir_mod
from ._util import system_name
from .probes import cpu, disk, gpu_apple, gpu_linux, gpu_nvidia, gpu_windows, memory, os_info, vulkan
from .runner import ProbeContext, ProbeSpec, run_probes

SCHEMA_VERSION = 1


def default_probe_specs(system):
    """The probes for this OS, each with its own timeout (seconds).

    Subprocesses inside a probe are capped at 5 s each (``_proc``); the probe
    timeouts sit just above the worst case of the commands each one runs.
    """
    specs = [
        ProbeSpec("os", os_info.probe, 4.0),
        ProbeSpec("cpu", cpu.probe, 7.0),
        ProbeSpec("memory", memory.probe, 3.0),
    ]
    if system != "Darwin":
        specs.append(ProbeSpec("gpu_nvidia", gpu_nvidia.probe, 9.0))
    if system == "Darwin":
        specs.append(ProbeSpec("gpu_apple", gpu_apple.probe, 7.0))
    elif system == "Windows":
        specs.append(ProbeSpec("gpu_windows", gpu_windows.probe, 7.0))
    elif system == "Linux":
        specs.append(ProbeSpec("gpu_linux", gpu_linux.probe, 7.0))
    specs.append(ProbeSpec("vulkan", vulkan.probe, 7.0))
    specs.append(ProbeSpec("disk", disk.probe, 5.0))
    return specs


# ------------------------------------------------------------- sections ---

def _default_os():
    return {"system": system_name() or None, "name": None, "version": None, "kernel": None, "arch": None,
            "python_arch": None, "is_translated": None, "is_wsl": None, "wsl_version": None,
            "wsl_gpu_passthrough": None, "is_apple_silicon": None, "python": platform.python_version()}


def _default_cpu():
    return {"model": None, "vendor": None, "physical_cores": None, "logical_cores": None, "max_mhz": None,
            "features": cpu.empty_features(None), "features_source": None}


def _default_memory():
    return {"total_bytes": None, "available_bytes": None, "swap_total_bytes": None, "swap_free_bytes": None}


def _default_disk(ctx):
    return {"models_dir": ctx.models_dir, "models_dir_source": ctx.models_dir_source, "exists": None,
            "checked_path": None, "total_bytes": None, "free_bytes": None, "is_synced_folder": None,
            "synced_folder_hint": None, "writable": None, "filesystem": None, "is_network": None}


def _merge_section(default, value):
    out = dict(default)
    if isinstance(value, dict):
        for k, v in value.items():
            out[k] = v
    return out


def _value(outcomes, name):
    o = outcomes.get(name)
    return o.value if o is not None and isinstance(o.value, dict) else None


def _probe_failed(outcomes, name):
    o = outcomes.get(name)
    return o is not None and o.value is None


def _cuda_accelerator(system, outcomes, platform_gpus, platform_ok, os_section):
    base = {"available": None, "detail": None, "state": None, "driver_version": None, "cuda_version": None}
    if system == "Darwin":
        return dict(base, available=False, state="not_applicable", detail="CUDA is not available on macOS.")
    nv = _value(outcomes, "gpu_nvidia")
    if nv is None:
        return dict(base, state="unknown", detail="The NVIDIA check did not finish.")
    state = nv.get("state")
    res = dict(base, state=state, driver_version=nv.get("driver_version"), cuda_version=nv.get("cuda_version"))
    if state == "ok":
        bits = []
        if nv.get("driver_version"):
            bits.append(f"driver {nv['driver_version']}")
        if nv.get("cuda_version"):
            bits.append(f"CUDA {nv['cuda_version']}")
        return dict(res, available=True, detail=", ".join(bits) or "nvidia-smi reports a working driver")
    if state == "not_found":
        if any(g.get("vendor") == "nvidia" for g in platform_gpus):
            return dict(res, available=False, detail="nvidia-smi not found")
        if platform_ok or os_section.get("wsl_gpu_passthrough") is False:
            return dict(res, available=False, detail="No NVIDIA GPU found.")
        return dict(res, available=None, detail="nvidia-smi not found")
    if state == "timeout":
        return dict(res, available=None, detail=nv.get("message") or "nvidia-smi did not respond")
    if platform_ok and not any(g.get("vendor") == "nvidia" for g in platform_gpus) and state != "no_devices":
        # nvidia-smi is left over from an old driver, but no NVIDIA GPU is in the
        # machine: not a broken driver worth warning about.
        return dict(res, available=False, state="no_gpu",
                    detail=f"No NVIDIA GPU found (nvidia-smi: {nv.get('message') or 'failed'}).")
    return dict(res, available=False, detail=nv.get("message") or "nvidia-smi failed")


def _metal_accelerator(system, os_section, all_gpus):
    if system != "Darwin":
        return {"available": False, "detail": "macOS only"}
    apple = [g for g in all_gpus if g.get("vendor") == "apple"]
    if os_section.get("is_apple_silicon") is True or apple:
        g = apple[0] if apple else {}
        d = g.get("details") or {}
        bits = [g.get("name") or "Apple Silicon GPU"]
        if d.get("gpu_cores"):
            bits.append(f"{d['gpu_cores']}-core GPU")
        if d.get("metal_support"):
            bits.append(d["metal_support"])
        return {"available": True, "detail": ", ".join(bits)}
    if os_section.get("is_apple_silicon") is False:
        return {"available": False, "detail": "llama.cpp's Metal backend needs Apple Silicon; this is an Intel Mac."}
    return {"available": None, "detail": "Could not tell whether this Mac has Apple Silicon."}


def _vulkan_accelerator(outcomes):
    vk = _value(outcomes, "vulkan")
    if vk is None:
        return {"available": None, "detail": "The Vulkan check did not finish.", "devices": []}, []
    devices = vk.get("devices") or []
    public = [{"name": d.get("name"), "type": d.get("type"),
               "vendor_id": gpus_mod.hex_id(d.get("vendor_id")), "device_id": gpus_mod.hex_id(d.get("device_id")),
               "api_version": d.get("api_version"), "driver_name": d.get("driver_name")} for d in devices]
    return {"available": vk.get("available"), "detail": vk.get("detail"), "devices": public}, devices


def assemble(outcomes, ctx, *, generated_at, duration_ms):
    """Report dict from probe outcomes. Pure: no probing happens here."""
    system = ctx.system
    os_section = _merge_section(_default_os(), _value(outcomes, "os"))
    cpu_section = _merge_section(_default_cpu(), _value(outcomes, "cpu"))
    if not isinstance(cpu_section.get("features"), dict):
        cpu_section["features"] = cpu.empty_features(None)
    mem_section = _merge_section(_default_memory(), _value(outcomes, "memory"))
    disk_section = _merge_section(_default_disk(ctx), _value(outcomes, "disk"))

    nv = _value(outcomes, "gpu_nvidia") or {}
    apple = _value(outcomes, "gpu_apple") or {}
    plat_name = {"Windows": "gpu_windows", "Linux": "gpu_linux"}.get(system)
    plat = (_value(outcomes, plat_name) if plat_name else None) or {}
    platform_ok = bool(plat_name) and not _probe_failed(outcomes, plat_name) and not plat.get("no_pci_sysfs")
    vulkan_acc, vk_devices = _vulkan_accelerator(outcomes)

    if system == "Darwin" and apple.get("is_apple_silicon") is not None:
        # The GPU probe asked the same question; use it if the OS probe failed.
        if os_section.get("is_apple_silicon") is None:
            os_section["is_apple_silicon"] = bool(apple["is_apple_silicon"])

    merged = gpus_mod.merge(
        nv.get("gpus") or [], apple.get("gpus") or [], plat.get("gpus") or [], vk_devices,
        nvidia_authoritative=nv.get("state") == "ok",
        platform_ok=platform_ok or system == "Darwin",
    )
    accelerators = {
        "cuda": _cuda_accelerator(system, outcomes, plat.get("gpus") or [], platform_ok, os_section),
        "metal": _metal_accelerator(system, os_section, merged),
        "vulkan": vulkan_acc,
    }
    report = {
        "schema": SCHEMA_VERSION,
        "generated_at": generated_at,
        "duration_ms": int(duration_ms),
        "os": os_section,
        "cpu": cpu_section,
        "memory": mem_section,
        "gpus": merged,
        "accelerators": accelerators,
        "disk": disk_section,
        "capability": None,
        "warnings": [],
        "probes": [outcomes[s].record() for s in outcomes],
    }
    capability.apply(report)
    return report


# ---------------------------------------------------------------- cache ---

@dataclass
class _Cached:
    ctx: ProbeContext
    outcomes: dict
    report: dict
    generated_at: str
    duration_ms: int


_lock = threading.Lock()
_cache = None


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_ctx(models_dir):
    ctx = ProbeContext()
    if models_dir is not None and str(models_dir).strip():
        ctx.models_dir, ctx.models_dir_source = models_dir_mod.resolve(models_dir)
    return ctx


def _collect(models_dir, specs=None):
    ctx = _make_ctx(models_dir)
    generated_at = _now_iso()
    t0 = time.perf_counter()
    outcomes = run_probes(specs if specs is not None else default_probe_specs(ctx.system), ctx)
    duration_ms = int((time.perf_counter() - t0) * 1000)
    report = assemble(outcomes, ctx, generated_at=generated_at, duration_ms=duration_ms)
    return _Cached(ctx, outcomes, report, generated_at, duration_ms)


def _recollect_disk(cached, models_dir):
    ctx = _make_ctx(models_dir)
    ctx.system, ctx.env, ctx.run, ctx.sys_root = cached.ctx.system, cached.ctx.env, cached.ctx.run, cached.ctx.sys_root
    spec = next((s for s in default_probe_specs(ctx.system) if s.name == "disk"), None)
    outcomes = dict(cached.outcomes)
    if spec is not None:
        outcomes.update(run_probes([spec], ctx))
    report = assemble(outcomes, ctx, generated_at=cached.generated_at, duration_ms=cached.duration_ms)
    return _Cached(ctx, outcomes, report, cached.generated_at, cached.duration_ms)


def _same_dir(a, b):
    if not a or not b:
        return False
    try:
        return os.path.normcase(os.path.abspath(os.path.expanduser(str(a)))) == \
            os.path.normcase(os.path.abspath(os.path.expanduser(str(b))))
    except Exception:
        return False


def get_report(refresh: bool = False, models_dir: str | None = None) -> dict:
    """The system report (schema 1). Cached in-process; thread-safe.

    ``refresh=True`` re-probes everything. Passing a ``models_dir`` that differs
    from the cached one re-checks only the disk section. With ``models_dir=None``
    the folder is resolved from ``vvl.paths``, then the settings file, then the
    default user-data folder (see ``models_dir.resolve``).
    Returns a deep copy, so callers may modify it freely.
    """
    global _cache
    with _lock:
        if refresh or _cache is None:
            _cache = _collect(models_dir)
        elif models_dir is not None and not _same_dir(models_dir, _cache.report["disk"].get("models_dir")):
            _cache = _recollect_disk(_cache, models_dir)
        return copy.deepcopy(_cache.report)


def clear_cache():
    global _cache
    with _lock:
        _cache = None
