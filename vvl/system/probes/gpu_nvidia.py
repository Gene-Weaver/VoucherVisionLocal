"""NVIDIA GPUs through nvidia-smi (ships with the driver on Windows and Linux,
and inside WSL 2 at /usr/lib/wsl/lib)."""

from __future__ import annotations

import os
import re

from .. import _proc
from .._util import MiB, clean_na, first_line, norm_pci_bus_id, to_float, to_int
from ..gpus import NVIDIA_UNIFIED_RE, hex_id, make_gpu

FIELDS = ["index", "uuid", "pci.bus_id", "pci.device_id", "memory.total", "memory.free",
          "memory.used", "driver_version", "compute_cap", "name"]
# compute_cap only exists in newer drivers; retried without it on old ones.
FIELDS_LEGACY = [f for f in FIELDS if f != "compute_cap"]


def candidates(system, env):
    if system == "Windows":
        root = env.get("SystemRoot") or env.get("windir") or r"C:\Windows"
        pf = env.get("ProgramW6432") or env.get("ProgramFiles") or r"C:\Program Files"
        return [
            os.path.join(root, "System32", "nvidia-smi.exe"),
            os.path.join(root, "Sysnative", "nvidia-smi.exe"),  # 32-bit Python on 64-bit Windows
            os.path.join(pf, "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"),  # drivers before ~2019
        ]
    return ["/usr/bin/nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi", "/usr/local/bin/nvidia-smi",
            "/opt/bin/nvidia-smi", "/usr/local/nvidia/bin/nvidia-smi"]


def parse_query_csv(text, fields=FIELDS):
    """Rows of ``--format=csv,noheader,nounits`` output as dicts.

    ``name`` is the last field so a name containing a comma is rejoined.
    Placeholders such as [N/A] and [Not Supported] become None.
    """
    rows = []
    n = len(fields)
    for line in (text or "").splitlines():
        if not line.strip() or "," not in line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < n:
            continue
        if len(parts) > n:
            parts = parts[: n - 1] + [", ".join(parts[n - 1:])]
        rows.append({f: clean_na(v) for f, v in zip(fields, parts)})
    return rows


def _mib(value):
    v = to_float(value)
    return None if v is None or v < 0 else int(v * MiB)


def parse_cuda_version(text):
    m = re.search(r"CUDA Version\s*:\s*([0-9]+(?:\.[0-9]+)*)", text or "")
    return m.group(1) if m else None


def parse_driver_version_header(text):
    m = re.search(r"Driver Version\s*:\s*([0-9][0-9.]*)", text or "")
    return m.group(1) if m else None


def classify_error(returncode, output):
    """(state, message) for a failed nvidia-smi run."""
    text = output or ""
    low = text.lower()
    if "couldn't communicate with the nvidia driver" in low or "could not communicate with the nvidia driver" in low:
        return "driver_not_loaded", "nvidia-smi could not talk to the NVIDIA driver (the driver is not loaded or not installed)"
    if "driver/library version mismatch" in low:
        return "driver_mismatch", "the NVIDIA driver and its libraries do not match (usually fixed by restarting after a driver update)"
    if "no devices were found" in low:
        return "no_devices", "nvidia-smi found no NVIDIA GPU"
    if "insufficient permissions" in low:
        return "permission", "nvidia-smi was denied access to the GPU"
    if "failed to initialize nvml" in low or "nvml" in low:
        return "nvml_error", first_line(text) or "NVML failed to initialize"
    return "error", first_line(text) or f"nvidia-smi exited with code {returncode}"


def gpus_from_rows(rows, *, arch=None, system_ram=None):
    gpus = []
    for row in rows:
        name = row.get("name")
        total = _mib(row.get("memory.total"))
        free = _mib(row.get("memory.free"))
        used = _mib(row.get("memory.used"))
        dev = to_int(row.get("pci.device_id"))  # e.g. 0x26B110DE = device << 16 | vendor
        details = {
            "nvidia_index": to_int(row.get("index")),
            "uuid": row.get("uuid"),
            "pci_bus_id": norm_pci_bus_id(row.get("pci.bus_id")),
            "pci_vendor_id": hex_id(dev & 0xFFFF) if dev is not None else "0x10de",
            "pci_device_id": hex_id(dev >> 16) if dev is not None else None,
            "compute_capability": row.get("compute_cap"),
            "vram_used_bytes": used,
        }
        # GB10 / Grace / Jetson share system RAM and report memory as [N/A].
        unified = bool(NVIDIA_UNIFIED_RE.search(name or "")) or (total is None and arch == "arm64")
        kind = "unified" if unified else "discrete"
        if unified and total is None:
            total = system_ram
            details["memory_from"] = "system_ram"
        gpus.append(make_gpu(
            "nvidia", name, kind=kind, vram_total=total,
            vram_free=free, driver_version=row.get("driver_version"),
            source="nvidia-smi", backend_hints=["cuda", "vulkan"], details=details,
        ))
    return gpus


def _system_ram():
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        return None


def probe(ctx, log):
    out = {"state": "not_found", "message": None, "gpus": [], "driver_version": None,
           "cuda_version": None, "path": None}
    exe = _proc.find_tool("nvidia-smi", candidates(ctx.system, ctx.env))
    if not exe:
        out["message"] = "nvidia-smi not found"
        return out
    out["path"] = exe

    fields = FIELDS
    res = ctx.run([exe, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"], timeout=5.0)
    if not res.ok and not res.timed_out and res.found and "not a valid field" in res.output.lower():
        fields = FIELDS_LEGACY
        res = ctx.run([exe, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"], timeout=5.0)
    if res.timed_out:
        out["state"], out["message"] = "timeout", "nvidia-smi did not respond within 5 seconds"
        log.fail(out["message"])
        return out
    if not res.found:
        out["message"] = "nvidia-smi not found"
        return out
    if res.error or res.returncode != 0:
        out["state"], out["message"] = classify_error(res.returncode, res.output)
        return out

    from .._util import machine
    from .os_info import normalize_arch

    rows = parse_query_csv(res.stdout, fields)
    if not rows:
        out["state"], out["message"] = classify_error(res.returncode, res.output)
        if out["state"] == "error":
            out["message"] = "nvidia-smi returned no GPU rows"
        return out
    out["gpus"] = gpus_from_rows(rows, arch=normalize_arch(machine()), system_ram=_system_ram())
    out["state"] = "ok"
    out["driver_version"] = next((g["driver_version"] for g in out["gpus"] if g["driver_version"]), None)

    # The driver's CUDA version is not a --query-gpu field. "-q -d COMPUTE" prints
    # it in ~20 ms; the plain table (~350 ms) is the fallback for odd drivers.
    for argv in ([exe, "-q", "-d", "COMPUTE", "-i", "0"], [exe]):
        header = ctx.run(argv, timeout=2.5)
        if header.ok and parse_cuda_version(header.stdout):
            out["cuda_version"] = parse_cuda_version(header.stdout)
            out["driver_version"] = out["driver_version"] or parse_driver_version_header(header.stdout)
            break
    return out
