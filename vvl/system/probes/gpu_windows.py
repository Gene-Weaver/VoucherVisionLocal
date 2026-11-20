"""Windows display adapters from the registry (no PowerShell in the normal path).

Win32_VideoController.AdapterRAM (WMI/CIM) is a 32-bit value that tops out at
4 GB, so the real size comes from the display-class driver key:
``HKLM\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-...}\\NNNN``
value ``HardwareInformation.qwMemorySize`` (64-bit). Older drivers only write
``HardwareInformation.MemorySize`` (32-bit, capped); those sizes are flagged.

Driver keys can outlive a removed card, so the list is limited to adapters that
are present right now (cfgmgr32), when that call works. PowerShell CIM is used
only if the registry yields nothing.
"""

from __future__ import annotations

import json
import os
import re

from .._util import clean_na, norm_pci_bus_id, to_int
from ..gpus import BACKEND_HINTS, VENDOR_IDS, classify_kind, hex_id, make_gpu
from . import _windows

CLASS_PATHS = [
    "\\".join(["SYSTEM", "CurrentControlSet", "Control", "Class", _windows.DISPLAY_CLASS_GUID]),
    "\\".join(["SYSTEM", "ControlSet001", "Control", "Class", _windows.DISPLAY_CLASS_GUID]),
]
ENUM_PATH = r"SYSTEM\CurrentControlSet\Enum"
CAP_32BIT = 0xFFF00000  # values at or above this are the 32-bit ceiling, not a real size

_BASIC_DISPLAY = re.compile(r"Microsoft Basic Display Adapter", re.I)
_VENDOR_LABEL = {"nvidia": "NVIDIA", "amd": "AMD", "intel": "Intel"}
_SKIP_NAMES = re.compile(
    r"Microsoft Basic (?:Display|Render)|Microsoft Remote Display|Microsoft Hyper-V Video|"
    r"Remote Desktop|Citrix|Parsec|Virtual Display|Indirect Display|DisplayLink|Mirage|"
    r"VMware SVGA|VirtualBox|Spacedesk|Splashtop|Meta Virtual Monitor|IddSampleDriver",
    re.I,
)


def _as_text(value):
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        try:
            return raw.decode("utf-16-le").split("\x00", 1)[0].strip() or None
        except UnicodeDecodeError:
            return raw.decode("latin-1").split("\x00", 1)[0].strip() or None
    if isinstance(value, (list, tuple)):
        return str(value[0]).strip() if value else None
    return clean_na(value)


def _as_int(value):
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        if len(raw) in (4, 8):
            return int.from_bytes(raw, "little")
        return None
    return to_int(value)


def adapter_memory(values):
    """(bytes or None, value name used, capped_at_4_gb)."""
    q = _as_int(values.get("HardwareInformation.qwMemorySize"))
    if q:
        return q, "qwMemorySize", False
    m = _as_int(values.get("HardwareInformation.MemorySize"))
    if m:
        m &= 0xFFFFFFFF
        return m, "MemorySize", m >= CAP_32BIT
    return None, None, False


def parse_pci_ids(text):
    """(vendor_id, device_id) from "PCI\\VEN_10DE&DEV_2684&..."; Nones otherwise."""
    m = re.search(r"ven_([0-9a-f]{4}).*?dev_([0-9a-f]{4})", text or "", re.I)
    if not m:
        return None, None
    return int(m.group(1), 16), int(m.group(2), 16)


def parse_location(text):
    """PCI bus id from LocationInformation in either of its two forms."""
    t = text or ""
    m = re.search(r"PCI bus (\d+), device (\d+), function (\d+)", t, re.I) or re.search(r"\((\d+),(\d+),(\d+)\)\s*$", t)
    if not m:
        return None
    return f"0000:{int(m.group(1)):02x}:{int(m.group(2)):02x}.{int(m.group(3))}"


def _vendor_from_provider(provider, name):
    text = f"{provider or ''} {name or ''}".lower()
    if "nvidia" in text:
        return "nvidia"
    if "advanced micro devices" in text or re.search(r"\bamd\b|\bati\b|radeon", text):
        return "amd"
    if "intel" in text:
        return "intel"
    return "unknown"


def parse_adapters(adapters, presence_checked=False):
    """GPU entries from raw registry value dicts (one per driver key).

    Each dict holds the driver key's values plus optional ``_instance_id``,
    ``_location`` and ``_subkey`` added by the reader.
    """
    gpus = []
    for values in adapters:
        name = _as_text(values.get("DriverDesc")) or _as_text(values.get("HardwareInformation.AdapterString")) \
            or _as_text(values.get("DeviceDesc"))
        if name and ";" in name:
            name = name.rsplit(";", 1)[-1].strip()
        match_id = _as_text(values.get("MatchingDeviceId")) or ""
        inst = _as_text(values.get("_instance_id")) or ""
        provider = _as_text(values.get("ProviderName"))
        vid, did = parse_pci_ids(match_id)
        if vid is None:
            vid, did = parse_pci_ids(inst)
        is_pci = match_id.lower().startswith("pci\\") or inst.lower().startswith("pci\\")
        is_qcom = "qcom" in (match_id + inst).lower() or "qualcomm" in (provider or "").lower()
        driver_missing = False
        if name and _BASIC_DISPLAY.search(name) and VENDOR_IDS.get(vid) in ("nvidia", "amd", "intel"):
            # A real GPU running on Windows' fallback driver: its vendor driver is not installed.
            driver_missing = True
            name = f"{_VENDOR_LABEL[VENDOR_IDS[vid]]} GPU (PCI {vid:04x}:{did:04x}, driver not installed)"
        elif name and _SKIP_NAMES.search(name):
            continue
        if not is_pci and not is_qcom:
            continue  # software / remote / virtual display drivers
        vendor = VENDOR_IDS.get(vid) if vid is not None else None
        if vendor is None:
            if vid is not None and not is_qcom:
                continue  # VMware, VirtualBox, Matrox, ASPEED, ...
            vendor = _vendor_from_provider(provider, name)
        vram, vram_src, capped = adapter_memory(values)
        if driver_missing:
            vram, vram_src, capped = None, None, False  # the fallback driver's numbers are not the card's
        bus = parse_location(_as_text(values.get("_location")))
        kind = "integrated" if is_qcom else classify_kind(vendor, name, vram, bus, did)
        details = {
            "pci_vendor_id": hex_id(vid),
            "pci_device_id": hex_id(did),
            "pci_bus_id": bus,
            "provider": provider,
            "registry_key": _as_text(values.get("_subkey")),
            "vram_source": vram_src,
            "vram_capped": capped,
            "presence_checked": presence_checked,
        }
        if driver_missing:
            details["driver_missing"] = True
        if kind == "integrated" and vram is not None:
            details["dedicated_memory_bytes"] = vram
        gpus.append(make_gpu(
            vendor, name, kind=kind, vram_total=vram, vram_free=None,
            driver_version=None if driver_missing else _as_text(values.get("DriverVersion")), source="registry",
            backend_hints=BACKEND_HINTS.get(vendor, []), details=details,
        ))
    return gpus


def parse_wmi_json(text):
    """GPU entries from ``Get-CimInstance Win32_VideoController | ConvertTo-Json``."""
    data = json.loads(text) if text and text.strip() else []
    if isinstance(data, dict):
        data = [data]
    adapters = []
    for item in data or []:
        if not isinstance(item, dict):
            continue
        ram = to_int(item.get("AdapterRAM"))
        adapters.append({
            "DriverDesc": item.get("Name"),
            "MatchingDeviceId": item.get("PNPDeviceID"),
            "DriverVersion": item.get("DriverVersion"),
            "ProviderName": item.get("AdapterCompatibility"),
            "HardwareInformation.MemorySize": ram & 0xFFFFFFFF if ram is not None else None,
        })
    gpus = parse_adapters(adapters, presence_checked=True)
    for g in gpus:
        g["source"] = "wmi"
    return gpus


def read_registry_adapters():
    """(adapters, presence_checked). Windows only; never raises."""
    present = _windows.present_display_devices()
    adapters = []
    if present is not None:
        for inst in present:
            enum_vals = _windows.reg_values("HKEY_LOCAL_MACHINE", ENUM_PATH + "\\" + inst) or {}
            driver = _as_text(enum_vals.get("Driver")) or ""
            sub = driver.rsplit("\\", 1)[-1] if "\\" in driver else None
            values = None
            if sub:
                for base in CLASS_PATHS:
                    values = _windows.reg_values("HKEY_LOCAL_MACHINE", base + "\\" + sub)
                    if values is not None:
                        break
            values = dict(values or {})
            values.setdefault("DeviceDesc", enum_vals.get("DeviceDesc"))
            values["_instance_id"] = inst
            values["_location"] = enum_vals.get("LocationInformation")
            values["_subkey"] = sub
            adapters.append(values)
        return adapters, True
    for base in CLASS_PATHS:
        subs = _windows.reg_subkeys("HKEY_LOCAL_MACHINE", base)
        if subs is None:
            continue
        for sub in subs:
            if not re.fullmatch(r"\d{4}", sub):
                continue
            values = _windows.reg_values("HKEY_LOCAL_MACHINE", base + "\\" + sub)
            if values:
                values = dict(values)
                values["_subkey"] = sub
                adapters.append(values)
        break
    return adapters, False


POWERSHELL_CMD = (
    "[Console]::OutputEncoding=[Text.Encoding]::UTF8; "
    "Get-CimInstance -ClassName Win32_VideoController | "
    "Select-Object Name,AdapterRAM,DriverVersion,PNPDeviceID,AdapterCompatibility | ConvertTo-Json -Compress"
)


def probe(ctx, log):
    out = {"gpus": [], "source": None}
    adapters, presence = read_registry_adapters()
    gpus = parse_adapters(adapters, presence_checked=presence) if adapters else []
    if gpus or adapters:
        out["gpus"], out["source"] = gpus, "registry"
        return out

    root =ctx.env.get("SystemRoot") or r"C:\Windows"
    ps = os.path.join(root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
    res = ctx.run([ps, "-NoProfile", "-NonInteractive", "-Command", POWERSHELL_CMD], timeout=5.0)
    if res.ok:
        try:
            out["gpus"], out["source"] = parse_wmi_json(res.stdout), "wmi"
        except (ValueError, TypeError) as exc:
            log.fail(f"could not parse the WMI adapter list: {type(exc).__name__}")
    else:
        log.fail("display adapters could not be read from the registry or WMI: " + res.describe_failure())
    return out


# Re-exported for tests.
__all__ = ["adapter_memory", "parse_adapters", "parse_location", "parse_pci_ids", "parse_wmi_json",
           "norm_pci_bus_id", "probe"]
