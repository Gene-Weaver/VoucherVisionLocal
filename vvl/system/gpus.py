"""GPU entries shared by every GPU probe: construction, integrated/discrete
classification, and de-duplication across sources (the vendor tool wins)."""

from __future__ import annotations

import re

from ._util import to_int

VENDOR_IDS = {0x10DE: "nvidia", 0x1002: "amd", 0x8086: "intel", 0x106B: "apple"}
BACKEND_HINTS = {"nvidia": ["cuda", "vulkan"], "amd": ["vulkan"], "intel": ["vulkan"]}

# Names that mean an NVIDIA part shares system memory (DGX Spark GB10, Grace,
# Jetson). nvidia-smi reports their memory as [N/A].
NVIDIA_UNIFIED_RE = re.compile(r"\b(GB10|GH200|GB200|Grace|Jetson|Orin|Thor|Xavier|Tegra)\b", re.I)

# Order matters: these are tested before the generic patterns below.
_AMD_IGPU_FIRST = re.compile(r"\bVega\s+\d{1,2}\s+Graphics\b|\bRadeon(?:\s*\(TM\))?\s+R[2-8]\s+Graphics\b", re.I)
_AMD_DGPU = re.compile(
    r"\bRX\s?\d|\bRX\s?Vega\b|\bR[579]\s\d{3}\b|\bRadeon\s+(?:\(TM\)\s+)?Pro\b|\bPRO\s+W\d|\bRadeon\s+VII\b|"
    r"\bFirePro\b|\bInstinct\b|\bNavi\b|\bVega\s+(?:10|20|56|64)\b|\bPolaris\b|\bEllesmere\b|\bBaffin\b|"
    r"\bLexa\b|\bFiji\b|\bTonga\b|\bHawaii\b|\bTahiti\b|\bW\d{4}\b|\bMI\d{2,3}\b",
    re.I,
)
_AMD_IGPU = re.compile(
    r"\b\d{3}M\b|\b80[4-6]0S\b|Radeon(?:\s*\(TM\))?\s+Graphics\b|\bRenoir\b|\bCezanne\b|\bLucienne\b|"
    r"\bBarcelo\b|\bRembrandt\b|\bPhoenix\d?\b|\bHawk Point\b|\bStrix\b|\bKrackan\b|\bRaphael\b|"
    r"\bGranite Ridge\b|\bMendocino\b|\bVan ?Gogh\b|\bPicasso\b|\bRaven\d?\b|\bDali\b|\bPollock\b|"
    r"\bKaveri\b|\bCarrizo\b|\bStoney\b|\bKabini\b|\bMullins\b|\bBeema\b|\bGodavari\b|\bBristol Ridge\b",
    re.I,
)
_INTEL_DGPU = re.compile(r"\bArc(?:\(TM\))?\s+(?:Pro\s+)?[AB]\d{2,3}\b|\bIris\s*(?:\(R\))?\s*Xe\s+MAX\b|"
                         r"\bData Center GPU\b|\bDG[12]\b|\bBattlemage\b|\bAlchemist\b", re.I)


def classify_kind(vendor, name, vram_total=None, pci_bus_id=None, device_id=None):
    """Best guess at "discrete" | "integrated"; None when it cannot be told."""
    n = name or ""
    if vendor == "nvidia":
        return "unified" if NVIDIA_UNIFIED_RE.search(n) else "discrete"
    if vendor == "apple":
        return "unified"
    if vendor == "intel":
        if device_id is not None and ((device_id & 0xFF00) == 0x5600 or (device_id & 0xFFE0) == 0xE200):
            return "discrete"  # Arc Alchemist (0x56xx) / Battlemage (0xe20x-0xe21x)
        if pci_bus_id and pci_bus_id.endswith(":00:02.0"):
            return "integrated"  # Intel's iGPU always sits at 00:02.0
        if _INTEL_DGPU.search(n):
            return "discrete"
        return "integrated"
    if vendor == "amd":
        if _AMD_IGPU_FIRST.search(n):
            return "integrated"
        if _AMD_DGPU.search(n):
            return "discrete"
        if _AMD_IGPU.search(n):
            return "integrated"
        if vram_total is not None:
            if vram_total <= 1024**3:
                return "integrated"
            if vram_total >= 3 * 1024**3:
                return "discrete"
        return None
    return None


def _clean_vulkan_name(name):
    """"AMD Radeon RX 7900 XTX (RADV NAVI31)" -> "AMD Radeon RX 7900 XTX"."""
    if not name:
        return None
    text = re.sub(r"\s*\((?:RADV|ANV|Intel open-source Mesa driver|NVK|Mesa)[^)]*\)\s*$", "", str(name)).strip()
    return text or None


def make_gpu(vendor, name, *, kind=None, vram_total=None, vram_free=None, driver_version=None,
             source, backend_hints=None, details=None):
    vendor = vendor if vendor in ("nvidia", "amd", "intel", "apple") else "unknown"
    return {
        "index": 0,
        "vendor": vendor,
        "name": name or None,
        "kind": kind,
        "vram_total_bytes": vram_total,
        "vram_free_bytes": vram_free,
        "driver_version": driver_version or None,
        "backend_hints": list(backend_hints if backend_hints is not None else BACKEND_HINTS.get(vendor, [])),
        "source": source,
        "details": dict(details or {}),
    }


def _pci_key(gpu):
    return (gpu.get("details") or {}).get("pci_bus_id")


def hex_id(value):
    """PCI ids are stored in ``details`` as readable strings like "0x10de"."""
    n = to_int(value)
    return None if n is None else f"0x{n:04x}"


def _ids(gpu):
    d = gpu.get("details") or {}
    return to_int(d.get("pci_vendor_id")), to_int(d.get("pci_device_id"))


def merge(nvidia_gpus, apple_gpus, platform_gpus, vulkan_devices=(), nvidia_authoritative=False,
          platform_ok=True):
    """Combine GPU lists from every source into one list with stable indexes.

    - nvidia-smi is authoritative for NVIDIA cards: when it worked, NVIDIA
      entries from the registry/sysfs are dropped (matched by PCI bus id when
      both sides have one; otherwise all of them).
    - Entries seen twice (same PCI bus id) are kept once.
    - Vulkan devices only fill in ``kind`` and a ``details.vulkan`` flag; they
      add new entries only when the platform GPU probe produced nothing.
    """
    result = []
    nv = sorted(nvidia_gpus or [], key=lambda g: (g.get("details") or {}).get("nvidia_index", 0))
    result.extend(nv)
    result.extend(apple_gpus or [])
    nv_bus = {_pci_key(g) for g in nv if _pci_key(g)}
    seen_bus = set(nv_bus)
    for g in platform_gpus or []:
        if g.get("vendor") == "nvidia" and nvidia_authoritative and nv:
            bus = _pci_key(g)
            if not bus or not nv_bus or bus in nv_bus:
                continue
            g = dict(g)
            g["details"] = dict(g.get("details") or {}, note="not listed by nvidia-smi")
        bus = _pci_key(g)
        if bus and bus in seen_bus:
            continue
        if bus:
            seen_bus.add(bus)
        result.append(g)

    vk = [d for d in (vulkan_devices or []) if d.get("type") not in ("cpu",)]
    for g in result:
        vid, did = _ids(g)
        for d in vk:
            if vid is not None and did is not None and d.get("vendor_id") == vid and d.get("device_id") == did:
                g["details"]["vulkan"] = True
                if g.get("kind") is None and d.get("type") in ("discrete", "integrated"):
                    g["kind"] = d["type"]
                vk_name = _clean_vulkan_name(d.get("name"))
                if g.get("source") == "sysfs" and vk_name:
                    # pci.ids names are often family-wide ("RX 7900 XT/7900 XTX/7900M")
                    # or codenames; the Vulkan driver reports the exact product.
                    g["details"]["pci_name"] = g.get("name")
                    g["name"] = vk_name
                break
    if not result and not platform_ok:
        for d in vk:
            vendor = VENDOR_IDS.get(d.get("vendor_id"), "unknown")
            result.append(make_gpu(
                vendor, d.get("name"),
                kind=d.get("type") if d.get("type") in ("discrete", "integrated") else None,
                source="vulkaninfo",
                details={"pci_vendor_id": hex_id(d.get("vendor_id")), "pci_device_id": hex_id(d.get("device_id")),
                         "vulkan": True},
            ))
    for i, g in enumerate(result):
        g["index"] = i
    return result
