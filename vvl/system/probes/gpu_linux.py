"""Linux GPUs from sysfs PCI devices (works even when no DRM driver is bound),
with names from ``lspci -mm -nn`` when it is installed.

AMD cards report VRAM in ``mem_info_vram_total``/``_used`` (amdgpu). Intel and
NVIDIA do not expose VRAM in sysfs; NVIDIA sizes come from nvidia-smi.
Display-only server chips (ASPEED, Matrox) and virtual adapters are ignored.
"""

from __future__ import annotations

import os
import re
import shlex

from .. import _proc
from .._util import to_int
from ..gpus import BACKEND_HINTS, VENDOR_IDS, classify_kind, hex_id, make_gpu

VENDOR_SHORT = {"nvidia": "NVIDIA", "amd": "AMD", "intel": "Intel"}


def _read(path, limit=4096):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(limit).strip()
    except OSError:
        return None


def scan_pci_display(sys_root="/"):
    """Raw dicts for PCI devices of class 0x03xxxx (display controllers)."""
    base = os.path.join(sys_root, "sys/bus/pci/devices")
    try:
        entries = sorted(os.listdir(base))
    except OSError:
        return None
    raw = []
    for addr in entries[:4096]:
        d = os.path.join(base, addr)
        cls = to_int(_read(os.path.join(d, "class")))
        if cls is None or (cls >> 16) != 0x03:
            continue
        driver = None
        link = os.path.join(d, "driver")
        if os.path.islink(link):
            try:
                driver = os.path.basename(os.readlink(link))
            except OSError:
                driver = None
        raw.append({
            "addr": addr,
            "class": cls,
            "vendor_id": to_int(_read(os.path.join(d, "vendor"))),
            "device_id": to_int(_read(os.path.join(d, "device"))),
            "driver": driver,
            "boot_vga": _read(os.path.join(d, "boot_vga")),
            "vram_total": to_int(_read(os.path.join(d, "mem_info_vram_total"))),
            "vram_used": to_int(_read(os.path.join(d, "mem_info_vram_used"))),
            "gtt_total": to_int(_read(os.path.join(d, "mem_info_gtt_total"))),
        })
    return raw


_ID_SUFFIX = re.compile(r"\s*\[[0-9a-fA-F]{4}\]$")


def parse_lspci_mm(text):
    """{bus_id: {"vendor": str, "device": str}} from ``lspci -mm -nn -D``."""
    out = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parts = shlex.split(line)
        except ValueError:
            continue
        positional = [p for p in parts if not re.fullmatch(r"-[rp]\S*", p)]
        if len(positional) < 4:
            continue
        slot = positional[0]
        if re.fullmatch(r"[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", slot):
            slot = "0000:" + slot
        out[slot.lower()] = {
            "class": _ID_SUFFIX.sub("", positional[1]),
            "vendor": _ID_SUFFIX.sub("", positional[2]),
            "device": _ID_SUFFIX.sub("", positional[3]),
        }
    return out


def display_name(vendor, device_text, vendor_id, device_id):
    """Marketing name from an lspci device string such as
    "Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M]" -> "AMD Radeon RX 7900 XT/7900 XTX/7900M"."""
    short = VENDOR_SHORT.get(vendor, "")
    text = (device_text or "").strip()
    if not text or text.lower() == "device":
        ids = f"{vendor_id:04x}:{device_id:04x}" if vendor_id is not None and device_id is not None else "unknown id"
        return f"{short} GPU (PCI {ids})".strip()
    m = re.search(r"\[([^\]]+)\]\s*$", text)
    label = m.group(1) if m else text
    if short and not label.lower().startswith(short.lower()):
        label = f"{short} {label}"
    return label


def gpus_from_scan(raw, names=None):
    names = names or {}
    gpus = []
    for r in raw or []:
        vendor = VENDOR_IDS.get(r.get("vendor_id"))
        if vendor not in ("nvidia", "amd", "intel"):
            continue
        bus = r["addr"].lower()
        device_text = (names.get(bus) or {}).get("device")
        name = display_name(vendor, device_text, r.get("vendor_id"), r.get("device_id"))
        classify_text = f"{device_text or ''} {name}"
        total = r.get("vram_total")
        kind = classify_kind(vendor, classify_text, total, bus, r.get("device_id"))
        free = None
        if total is not None and r.get("vram_used") is not None:
            free = max(0, total - r["vram_used"])
        details = {
            "pci_bus_id": bus,
            "pci_vendor_id": hex_id(r.get("vendor_id")),
            "pci_device_id": hex_id(r.get("device_id")),
            "kernel_driver": r.get("driver"),
            "boot_vga": r.get("boot_vga") == "1",
        }
        if r.get("gtt_total") is not None:
            details["gtt_total_bytes"] = r["gtt_total"]
        if not r.get("driver") or r.get("driver") in ("vfio-pci", "pci-stub"):
            details["driver_missing"] = True  # no kernel driver (or reserved for a VM)
        if kind == "integrated" and total is not None:
            details["dedicated_memory_bytes"] = total
        gpus.append(make_gpu(vendor, name, kind=kind, vram_total=total, vram_free=free, source="sysfs",
                             backend_hints=BACKEND_HINTS.get(vendor, []), details=details))
    return gpus


def probe(ctx, log):
    out = {"gpus": [], "lspci": False}
    raw = scan_pci_display(ctx.sys_root)
    if raw is None:
        # No PCI bus in sysfs (WSL 2, some containers): nothing to report here.
        out["no_pci_sysfs"] = True
        return out
    names = {}
    if raw:
        exe = _proc.find_tool("lspci", ["/usr/bin/lspci", "/usr/sbin/lspci", "/sbin/lspci", "/bin/lspci"])
        if exe:
            res = ctx.run([exe, "-mm", "-nn", "-D"], timeout=5.0)
            if res.ok:
                names = parse_lspci_mm(res.stdout)
                out["lspci"] = True
    out["gpus"] = gpus_from_scan(raw, names)
    return out
