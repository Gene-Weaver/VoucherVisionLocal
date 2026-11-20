"""Turn the probed hardware into a recommendation: llama.cpp backend, memory
budgets, tier, a one-line summary, and plain-language warnings.

Everything here works on the report dict alone (no probing), tolerates
missing or odd fields, and never raises from ``apply``/``plan``. All numbers
come from ``policy.py``.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field

from . import policy
from ._util import fmt_gb, short_name, to_float

BACKENDS = ("cuda", "metal", "vulkan", "cpu")
TIERS = ("cpu_only", "entry", "mid", "high", "workstation")
NVIDIA_BROKEN_STATES = {"driver_not_loaded", "driver_mismatch", "nvml_error", "permission", "timeout", "error"}

_CAPABLE_IGPU = [re.compile(p, re.I) for p in policy.CAPABLE_IGPU_PATTERNS]


# ---------------------------------------------------------------- budgets ---

def usable_discrete(total):
    if not isinstance(total, int) or total <= 0:
        return None
    reserve = max(policy.DISCRETE_RESERVE_MIN_BYTES, int(policy.DISCRETE_RESERVE_FRACTION * total))
    return max(0, total - reserve)


def unified_budget(ram, wired_limit=None):
    if isinstance(wired_limit, int) and wired_limit > 0:
        return wired_limit if not isinstance(ram, int) else min(wired_limit, ram)
    if not isinstance(ram, int) or ram <= 0:
        return None
    frac = policy.UNIFIED_FRACTION_SMALL if ram <= policy.UNIFIED_SMALL_RAM_LIMIT_BYTES else policy.UNIFIED_FRACTION_LARGE
    return int(ram * frac)


def integrated_budget(dedicated, ram):
    shared = int(ram * policy.INTEGRATED_SHARED_FRACTION) if isinstance(ram, int) and ram > 0 else None
    vals = [v for v in (dedicated, shared) if isinstance(v, int) and v > 0]
    return max(vals) if vals else None


def cpu_budget(ram):
    if not isinstance(ram, int) or ram <= 0:
        return None
    reserve = max(policy.CPU_RESERVE_MIN_BYTES, int(policy.CPU_RESERVE_FRACTION * ram))
    return max(0, ram - reserve)


def tier_for(backend, gpu_budget):
    if backend == "cpu":
        return "cpu_only"
    if not isinstance(gpu_budget, int):
        return None
    for name, lower in policy.TIER_THRESHOLDS:
        if gpu_budget >= lower:
            return name
    return "entry"


# ------------------------------------------------------------------- plan ---

@dataclass
class Slot:
    index: int
    name: str
    usable: int
    free: int | None
    total: int | None


@dataclass
class Plan:
    backend: str = "cpu"
    reason: str = ""
    mode: str = "cpu"  # discrete | shared | cpu
    slots: list = field(default_factory=list)  # discrete mode: one per GPU with known memory
    gpus: list = field(default_factory=list)  # the GPU entries the backend would use
    gpu_budget: int | None = 0
    cpu_budget: int | None = None
    ram_total: int | None = None
    ram_available: int | None = None
    device_label: str = "the GPU"  # for sentences: "the GPU", "this Mac's GPU", ...
    nvidia_broken: bool = False
    nvidia_too_old: list = field(default_factory=list)


def _section(report, key):
    v = report.get(key) if isinstance(report, dict) else None
    return v if isinstance(v, dict) else {}


def _gpus(report):
    v = report.get("gpus") if isinstance(report, dict) else None
    return [g for g in v if isinstance(g, dict)] if isinstance(v, list) else []


def _details(g):
    d = g.get("details")
    return d if isinstance(d, dict) else {}


def _int(v):
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _vendor_label(vendor):
    return {"amd": "AMD", "intel": "Intel", "nvidia": "NVIDIA", "apple": "Apple"}.get(vendor, "A")


def _discrete_slots(gpus):
    slots = []
    for g in gpus:
        total = _int(g.get("vram_total_bytes"))
        usable = usable_discrete(total)
        if usable is not None:
            slots.append(Slot(_int(g.get("index")) or 0, g.get("name") or "GPU", usable,
                              _int(g.get("vram_free_bytes")), total))
    return slots


def plan(report) -> Plan:
    os_ = _section(report, "os")
    mem = _section(report, "memory")
    acc = _section(report, "accelerators")
    gpus = _gpus(report)
    ram = _int(mem.get("total_bytes"))
    p = Plan(cpu_budget=cpu_budget(ram), ram_total=ram, ram_available=_int(mem.get("available_bytes")))
    system = os_.get("system")
    is_mac = system == "Darwin"
    cuda = acc.get("cuda") if isinstance(acc.get("cuda"), dict) else {}
    vk_available = (acc.get("vulkan") or {}).get("available") if isinstance(acc.get("vulkan"), dict) else None

    nvidia_smi = [g for g in gpus if g.get("vendor") == "nvidia" and g.get("source") == "nvidia-smi"]
    nvidia_other = [g for g in gpus if g.get("vendor") == "nvidia" and g.get("source") != "nvidia-smi"]
    p.nvidia_broken = bool(nvidia_other) or (not nvidia_smi and cuda.get("state") in NVIDIA_BROKEN_STATES)

    def cc_ok(g):
        cc = to_float(_details(g).get("compute_capability"))
        return cc is None or cc >= policy.MIN_CUDA_COMPUTE_CAPABILITY

    supported = [g for g in nvidia_smi if cc_ok(g)]
    p.nvidia_too_old = [g for g in nvidia_smi if not cc_ok(g)]

    # 1. NVIDIA with a working driver -> CUDA.
    if supported:
        p.backend, p.gpus = "cuda", supported
        n = len(supported)
        p.reason = ("NVIDIA GPU with a working driver was found." if n == 1
                    else f"{n} NVIDIA GPUs with a working driver were found.")
        unified = [g for g in supported if g.get("kind") == "unified"]
        if unified:
            p.mode = "shared"
            p.gpu_budget = unified_budget(ram or _int(unified[0].get("vram_total_bytes")))
            p.device_label = "the GPU (which shares system memory)"
        else:
            p.mode = "discrete"
            p.slots = _discrete_slots(supported)
            p.gpu_budget = sum(s.usable for s in p.slots) if p.slots else None
        return p

    # 2. Apple Silicon -> Metal (unified memory).
    apple = [g for g in gpus if g.get("vendor") == "apple" and g.get("kind") == "unified"]
    if os_.get("is_apple_silicon") is True or apple:
        p.backend, p.mode, p.gpus = "metal", "shared", apple
        wired = _int(_details(apple[0]).get("wired_limit_bytes")) if apple else None
        p.gpu_budget = unified_budget(ram or (_int(apple[0].get("vram_total_bytes")) if apple else None), wired)
        p.reason = "Apple Silicon Mac: llama.cpp runs models on the GPU through Metal."
        p.device_label = "this Mac's GPU"
        return p

    others = [g for g in gpus if g.get("vendor") in ("amd", "intel")]
    discrete = [g for g in others if g.get("kind") == "discrete"]
    capable_igpu = [g for g in others if g.get("kind") == "integrated"
                    and any(rx.search(g.get("name") or "") for rx in _CAPABLE_IGPU)]

    # 3. AMD / Intel discrete GPU -> Vulkan.
    if discrete and not is_mac and vk_available is not False:
        p.backend, p.mode, p.gpus = "vulkan", "discrete", discrete
        p.slots = _discrete_slots(discrete)
        p.gpu_budget = sum(s.usable for s in p.slots) if p.slots else None
        label = _vendor_label(discrete[0].get("vendor"))
        p.reason = (f"{label} GPU found; llama.cpp uses it through Vulkan." if vk_available
                    else f"{label} GPU found; llama.cpp can use it through Vulkan (Vulkan support could not be confirmed).")
        return p

    # 4. A capable integrated GPU -> Vulkan with shared memory.
    if capable_igpu and not is_mac and vk_available is not False:
        g = capable_igpu[0]
        p.backend, p.mode, p.gpus = "vulkan", "shared", [g]
        p.gpu_budget = integrated_budget(_int(_details(g).get("dedicated_memory_bytes")) or _int(g.get("vram_total_bytes")), ram)
        p.reason = (f"Integrated {short_name(g.get('name') or 'GPU')} found; llama.cpp can use it through Vulkan "
                    f"with shared system memory.")
        p.device_label = "the integrated GPU"
        return p

    # 5. Processor only.
    p.backend, p.mode, p.gpu_budget, p.gpus = "cpu", "cpu", 0, []
    if is_mac:
        p.reason = "Intel Mac: llama.cpp's GPU support (Metal) needs Apple Silicon, so models run on the processor."
    elif p.nvidia_broken:
        p.reason = "An NVIDIA GPU was found but its driver is not working, so models run on the processor until the driver is fixed."
    elif p.nvidia_too_old:
        p.reason = "The NVIDIA GPU is too old for llama.cpp's CUDA support, so models run on the processor."
    elif os_.get("is_wsl"):
        p.reason = "No GPU is available inside WSL, so models run on the processor."
    elif discrete and vk_available is False:
        p.reason = f"{_vendor_label(discrete[0].get('vendor'))} GPU found but Vulkan is not working, so models run on the processor."
    elif any(g.get("kind") == "integrated" for g in gpus):
        p.reason = "Only an integrated GPU was found, which is not fast enough to help, so models run on the processor."
    else:
        p.reason = "No GPU that llama.cpp can use was found, so models run on the processor."
    return p


# ---------------------------------------------------------------- summary ---

def _gpu_group_text(gpus, mode):
    groups = OrderedDict()
    for g in gpus:
        name = short_name(g.get("name") or "GPU")
        groups.setdefault(name, []).append(g)
    texts = []
    for name, members in groups.items():
        total = _int(members[0].get("vram_total_bytes"))
        if mode == "shared":
            mem = "shared memory"
        else:
            mem = fmt_gb(total) if total else "memory unknown"
        if len(members) > 1:
            texts.append(f"{len(members)}x {name} ({mem} each)" if mode != "shared" else f"{len(members)}x {name} ({mem})")
        else:
            texts.append(f"{name} ({mem})")
    return " + ".join(texts)


def summary_text(report, p):
    cpu = _section(report, "cpu")
    ram = p.ram_total
    threads = _int(cpu.get("logical_cores"))
    parts = []
    if p.backend == "metal":
        g = p.gpus[0] if p.gpus else {}
        name = short_name(g.get("name") or cpu.get("model") or "Apple Silicon")
        cores = _int(_details(g).get("gpu_cores")) if g else None
        parts.append(name + (f" ({cores}-core GPU)" if cores else ""))
        if _int(cpu.get("physical_cores")):
            parts.append(f"{cpu['physical_cores']} CPU cores")
        if ram:
            parts.append(f"{fmt_gb(ram)} unified memory")
        return ", ".join(parts)
    if p.backend in ("cuda", "vulkan") and p.gpus:
        parts.append(_gpu_group_text(p.gpus, p.mode))
    elif cpu.get("model"):
        parts.append(short_name(cpu["model"]))
    if threads:
        parts.append(f"{threads} threads")
    if ram:
        parts.append(f"{fmt_gb(ram)} RAM")
    if p.backend == "cpu":
        parts.append("no usable GPU")
    return ", ".join(parts) if parts else "System details unavailable"


# --------------------------------------------------------------- warnings ---

def _w(code, message, severity="warning"):
    return {"code": code, "message": message, "severity": severity}


def warnings_for(report, p):
    out = []
    os_ = _section(report, "os")
    cpu = _section(report, "cpu")
    disk = _section(report, "disk")
    acc = _section(report, "accelerators")
    feats = cpu.get("features") if isinstance(cpu.get("features"), dict) else {}
    system = os_.get("system")

    if os_.get("is_translated") is True:
        if system == "Darwin":
            out.append(_w("RUNNING_UNDER_ROSETTA",
                          "VoucherVisionLocal is running under Rosetta (Intel emulation) on an Apple Silicon Mac; "
                          "install the Apple Silicon version for full speed."))
        else:
            out.append(_w("RUNNING_UNDER_EMULATION",
                          "VoucherVisionLocal is running as an emulated x64 program on an ARM computer; "
                          "install the ARM64 version for full speed."))
    arch = os_.get("arch")
    if arch and arch not in ("x86_64", "arm64"):
        out.append(_w("UNSUPPORTED_ARCH",
                      f"This computer's processor type ({arch}) is not supported by the local engine."))

    if arch == "x86_64" and os_.get("is_apple_silicon") is not True and feats.get("avx2") is False:
        if feats.get("avx") is False:
            msg = "This processor lacks AVX and AVX2, so anything that runs on the processor will be very slow."
        else:
            msg = "This processor lacks AVX2, so anything that runs on the processor will be slow."
        out.append(_w("CPU_NO_AVX2", msg))

    ram = p.ram_total
    if isinstance(ram, int) and ram < policy.LOW_RAM_BYTES:
        out.append(_w("LOW_RAM", f"This computer has only {fmt_gb(ram)} of memory, so only the smallest models "
                                 f"can run, and slowly."))

    if p.nvidia_broken:
        detail = (acc.get("cuda") or {}).get("detail") if isinstance(acc.get("cuda"), dict) else None
        msg = "An NVIDIA GPU was found, but its driver is not working"
        if detail and detail != "nvidia-smi not found":
            msg += f" ({detail.rstrip('.')})"
        elif detail == "nvidia-smi not found":
            msg += " (nvidia-smi is missing, so the driver is probably not installed)"
        msg += ". Install or repair the NVIDIA driver to use the GPU."
        out.append(_w("NVIDIA_DRIVER_NOT_WORKING", msg))
    for g in _gpus(report):
        if g.get("vendor") in ("amd", "intel") and _details(g).get("driver_missing"):
            label = _vendor_label(g.get("vendor"))
            out.append(_w("GPU_DRIVER_MISSING",
                          f"{g.get('name') or 'A GPU'} has no {label} driver installed; install the {label} graphics "
                          f"driver to use it."))
    for g in p.nvidia_too_old:
        cc = _details(g).get("compute_capability")
        out.append(_w("NVIDIA_GPU_TOO_OLD",
                      f"{g.get('name') or 'The NVIDIA GPU'} (compute capability {cc}) is too old for llama.cpp's "
                      f"CUDA support."))

    if os_.get("is_wsl") and p.backend == "cpu":
        if os_.get("wsl_version") == 1:
            msg = "WSL 1 cannot use the GPU; use WSL 2 or the Windows version of VoucherVisionLocal."
        elif os_.get("wsl_gpu_passthrough") is False:
            msg = ("This WSL installation has no GPU access; update WSL and the Windows GPU driver, "
                   "or use the Windows version of VoucherVisionLocal.")
        else:
            msg = ("No GPU that llama.cpp can use is visible inside WSL (only NVIDIA GPUs work there); "
                   "use the Windows version of VoucherVisionLocal to use this computer's GPU.")
        out.append(_w("WSL_NO_GPU", msg))

    vk = acc.get("vulkan") if isinstance(acc.get("vulkan"), dict) else {}
    amd_intel_dgpu = [g for g in _gpus(report) if g.get("vendor") in ("amd", "intel") and g.get("kind") == "discrete"]
    if amd_intel_dgpu and system != "Darwin" and vk.get("available") is False and p.backend != "cuda":
        out.append(_w("VULKAN_NOT_AVAILABLE",
                      f"{amd_intel_dgpu[0].get('name') or 'The GPU'} was found, but Vulkan is not working "
                      f"({vk.get('detail') or 'no driver'}); update the GPU driver to use it."))

    if p.backend in ("cuda", "vulkan") and p.mode == "discrete" and p.gpu_budget is None:
        out.append(_w("GPU_MEMORY_UNKNOWN", "The GPU's memory size could not be read, so model fit cannot be checked.",
                      "info"))
    if any(_details(g).get("vram_capped") for g in p.gpus):
        out.append(_w("GPU_MEMORY_CAPPED", "Windows reported the GPU memory as 4 GB, its 32-bit maximum; the real "
                                           "amount may be larger.", "info"))

    if disk.get("models_dir"):
        if disk.get("exists") is False:
            out.append(_w("MODELS_DIR_MISSING", "The models folder does not exist yet. It is created on first download.",
                          "info"))
        if disk.get("is_synced_folder"):
            hint = disk.get("synced_folder_hint") or "a cloud-synced folder"
            out.append(_w("MODELS_DIR_SYNCED",
                          f"The models folder is inside {hint}. Model files are tens of GB; choose a folder that is "
                          f"not synced."))
        fs = str(disk.get("filesystem") or "").lower()
        mdir = str(disk.get("models_dir"))
        if os_.get("is_wsl") and (fs in ("9p", "drvfs", "v9fs") or re.match(r"^/mnt/[a-z]/", mdir)):
            out.append(_w("MODELS_DIR_WSL_WINDOWS_DRIVE",
                          "The models folder is on a Windows drive (/mnt/...), which WSL reads slowly; choose a "
                          "folder inside the Linux file system."))
        elif disk.get("is_network"):
            out.append(_w("MODELS_DIR_NETWORK", "The models folder is on a network drive; models load much faster "
                                                "from a local disk."))
        if fs in ("vfat", "msdos", "fat", "fat16", "fat32"):
            out.append(_w("MODELS_DIR_FAT32", "The models folder is on a FAT32 drive, which cannot store files "
                                              "larger than 4 GB."))
        if disk.get("writable") is False:
            out.append(_w("MODELS_DIR_NOT_WRITABLE", "VoucherVisionLocal cannot write to the models folder; choose "
                                                     "another folder."))
        free = _int(disk.get("free_bytes"))
        if free is not None and free < policy.LOW_DISK_FREE_BYTES:
            out.append(_w("LOW_DISK_SPACE", f"Only {fmt_gb(free)} is free on the drive that holds the models folder; "
                                            f"model files need 2 to 36 GB each."))

    failed = [pr.get("name") for pr in (report.get("probes") or []) if isinstance(pr, dict) and pr.get("ok") is False]
    if failed:
        out.append(_w("PROBES_INCOMPLETE", f"Some system checks did not finish ({', '.join(map(str, failed))}); "
                                           f"parts of this report may be missing."))
    return out


# ------------------------------------------------------------------ apply ---

def compute(report):
    """(capability dict, warnings list) for a report. Never raises."""
    try:
        p = plan(report)
        cap = {
            "recommended_backend": p.backend,
            "backend_reason": p.reason,
            "gpu_memory_budget_bytes": p.gpu_budget,
            "cpu_memory_budget_bytes": p.cpu_budget,
            "tier": tier_for(p.backend, p.gpu_budget),
            "summary": summary_text(report, p),
            "memory_mode": p.mode,
            "gpu_indices": [_int(g.get("index")) for g in p.gpus if _int(g.get("index")) is not None],
        }
    except Exception as exc:  # pragma: no cover - defensive
        cap = {"recommended_backend": None, "backend_reason": None, "gpu_memory_budget_bytes": None,
               "cpu_memory_budget_bytes": None, "tier": None,
               "summary": f"Capability could not be computed ({type(exc).__name__}).",
               "memory_mode": None, "gpu_indices": []}
        return cap, [_w("PROBES_INCOMPLETE", "The capability summary could not be computed.")]
    try:
        warns = warnings_for(report, p)
    except Exception:  # pragma: no cover - defensive
        warns = []
    return cap, warns


def apply(report):
    """Fill ``report["capability"]`` and ``report["warnings"]`` in place; returns the report."""
    cap, warns = compute(report)
    report["capability"] = cap
    report["warnings"] = warns
    return report
