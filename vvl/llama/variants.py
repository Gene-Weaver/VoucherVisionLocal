"""Which llama.cpp build fits this computer.

`assess(lock, report, ...)` returns the variants for this OS and processor type
in the lock file's preference order, each with `compatible` and a `reason`.
The recommended variant is the first compatible one, so the choice falls back
down the list: CUDA 13 -> CUDA 12 -> Vulkan -> CPU on an NVIDIA PC, Vulkan ->
CPU on an AMD/Intel PC, Metal on Apple Silicon.

A variant that failed its self-test on this computer (recorded by the
installer, see `selftest_failures`) is not compatible until the NVIDIA driver
version changes.
"""

from __future__ import annotations

import platform
import re

from vvl.system import capability as sys_capability

OS_NAMES = {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}
ARCH_NAMES = {"x86_64": "x86_64", "amd64": "x86_64", "x64": "x86_64",
              "arm64": "arm64", "aarch64": "arm64", "armv8": "arm64", "armv8l": "arm64"}


def _section(report, key) -> dict:
    value = report.get(key) if isinstance(report, dict) else None
    return value if isinstance(value, dict) else {}


def platform_key(report: dict | None) -> tuple[str | None, str | None]:
    """(os, arch) in the lock file's vocabulary, from the report (else this process).

    `os.arch` is the hardware architecture, so an x64 Python running under
    Rosetta or Windows-on-ARM emulation still gets the native ARM64 engine.
    """
    os_ = _section(report or {}, "os")
    system = os_.get("system") or platform.system()
    arch = (os_.get("arch") or platform.machine() or "").lower()
    if os_.get("is_apple_silicon") is True:
        arch = "arm64"
    return OS_NAMES.get(system), ARCH_NAMES.get(arch)


def version_tuple(text) -> tuple[int, ...] | None:
    if not isinstance(text, str):
        return None
    parts = re.findall(r"\d+", text)
    return tuple(int(p) for p in parts) if parts else None


def driver_at_least(driver: str | None, minimum: str | None) -> bool | None:
    """True/False, or None when the driver version is unknown."""
    if not minimum:
        return True
    have, need = version_tuple(driver), version_tuple(minimum)
    if have is None or need is None:
        return None
    width = max(len(have), len(need))
    return have + (0,) * (width - len(have)) >= need + (0,) * (width - len(need))


def nvidia_driver_version(report) -> str | None:
    cuda = _section(_section(report or {}, "accelerators"), "cuda")
    if cuda.get("driver_version"):
        return str(cuda["driver_version"])
    for gpu in (report or {}).get("gpus") or []:
        if isinstance(gpu, dict) and gpu.get("vendor") == "nvidia" and gpu.get("driver_version"):
            return str(gpu["driver_version"])
    return None


def local_glibc() -> str | None:
    """This computer's glibc version on Linux ("2.35"), else None."""
    import os

    try:
        value = os.confstr("CS_GNU_LIBC_VERSION")
    except (AttributeError, ValueError, OSError):
        return None
    return value.split()[-1] if value else None


def host_glibc() -> str | None:
    """The glibc that `assess(glibc="auto")` checks: this process's, on Linux only.

    The single place the host is consulted, so tests can pin it.
    """
    return local_glibc() if platform.system() == "Linux" else None


def _glibc_check(variant, glibc: str | None) -> tuple[bool, str | None]:
    minimum = getattr(variant.requires, "glibc_min", None)
    if variant.os != "linux" or not minimum or not glibc:
        return True, None
    if driver_at_least(glibc, minimum) is False:
        return False, (f"Needs glibc {minimum} or newer (Ubuntu 24.04 or a similarly recent Linux); "
                       f"this computer has glibc {glibc}.")
    return True, None


def _backend_check(variant, report, plan) -> tuple[bool, str | None]:
    """(compatible, reason) from the hardware alone."""
    gpus = [g for g in (report or {}).get("gpus") or [] if isinstance(g, dict)]
    acc = _section(report or {}, "accelerators")
    backend = variant.backend
    if backend == "cpu":
        return True, None
    if backend == "metal":
        os_ = _section(report or {}, "os")
        if os_.get("is_apple_silicon") is False:
            return False, "Metal needs an Apple Silicon Mac."
        return True, None
    if backend == "cuda":
        working = [g for g in gpus if g.get("vendor") == "nvidia" and g.get("source") == "nvidia-smi"]
        if not working:
            if any(g.get("vendor") == "nvidia" for g in gpus) or plan.nvidia_broken:
                return False, "The NVIDIA driver is not working, so CUDA cannot be used."
            return False, "No NVIDIA GPU was found."
        if plan.nvidia_too_old and len(plan.nvidia_too_old) == len(working):
            return False, "The NVIDIA GPU is too old for llama.cpp's CUDA support."
        minimum = variant.requires.nvidia_driver_min
        driver = nvidia_driver_version(report)
        ok = driver_at_least(driver, minimum)
        if ok is False:
            return False, f"Needs NVIDIA driver {minimum} or newer; this computer has {driver}."
        if ok is None:
            return True, f"The NVIDIA driver version could not be read (this build needs {minimum} or newer)."
        return True, None
    if backend == "vulkan":
        vulkan = _section(acc, "vulkan")
        if vulkan.get("available") is False:
            return False, "Vulkan is not working on this computer."
        # The same judgement as the system checker: a working NVIDIA GPU, an
        # AMD/Intel discrete GPU, or an integrated GPU fast enough to help.
        nvidia = [g for g in gpus if g.get("vendor") == "nvidia" and g.get("source") == "nvidia-smi"]
        if not nvidia and plan.backend != "vulkan":
            return False, "No GPU that is worth using through Vulkan was found."
        if vulkan.get("available") is None:
            return True, "Vulkan support could not be confirmed (vulkaninfo is not installed)."
        return True, None
    return False, f"Unknown backend {backend!r}."


def assess(lock, report: dict | None, *, installed: set[str] | frozenset = frozenset(),
           failures: dict | None = None, glibc: str | None = "auto") -> list[dict]:
    """The variants for this OS + processor type, in preference order.

    `glibc` is checked against a variant's optional `requires.glibc_min`;
    "auto" uses `host_glibc()` when the report describes Linux; None means
    unknown and never blocks a build (its self-test still runs).
    """
    os_name, arch = platform_key(report)
    if glibc == "auto":
        glibc = host_glibc() if os_name == "linux" else None
    plan = sys_capability.plan(report or {})
    driver = nvidia_driver_version(report)
    out = []
    for variant in lock.variants:
        if variant.os != os_name or variant.arch != arch:
            continue
        compatible, reason = _backend_check(variant, report, plan)
        if compatible:
            ok, why = _glibc_check(variant, glibc)
            if not ok:
                compatible, reason = False, why
        failure = (failures or {}).get(variant.id)
        if compatible and failure and failure.get("driver_version") == driver:
            compatible = False
            reason = f"Failed its self-test on this computer: {failure.get('message') or 'no details'}"
        out.append({
            "id": variant.id,
            "label": variant.label,
            "backend": variant.backend,
            "download_size_bytes": variant.download_size_bytes,
            "compatible": compatible,
            "reason": reason,
            "installed": variant.id in installed,
        })
    return out


def recommend(assessed: list[dict]) -> dict | None:
    """The first compatible variant, with a reason that names what was skipped."""
    chosen = next((v for v in assessed if v["compatible"]), None)
    if chosen is None:
        return None
    skipped = []
    for v in assessed:
        if v is chosen:
            break
        skipped.append(f"{v['label']}: {v['reason']}")
    reason = f"{chosen['label']} is the best build for this computer."
    if chosen.get("reason"):
        reason += f" Note: {chosen['reason']}"
    if skipped:
        reason += " Not chosen: " + " ".join(skipped)
    return {"id": chosen["id"], "label": chosen["label"], "backend": chosen["backend"],
            "download_size_bytes": chosen["download_size_bytes"], "reason": reason}


def next_after(assessed: list[dict], variant_id: str) -> dict | None:
    """The next compatible variant after `variant_id` (for self-test failure messages)."""
    seen = False
    for v in assessed:
        if seen and v["compatible"] and v["id"] != variant_id:
            return v
        if v["id"] == variant_id:
            seen = True
    return None
