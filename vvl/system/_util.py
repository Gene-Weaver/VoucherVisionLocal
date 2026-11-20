"""Small, dependency-free helpers shared by the probes, capability, and fit code."""

from __future__ import annotations

import math
import os
import platform
import re
import sys

KiB = 1024
MiB = 1024**2
GiB = 1024**3
TiB = 1024**4

# Values that tools print when a field is unknown or unsupported.
_NA_VALUES = {
    "",
    "n/a",
    "na",
    "none",
    "null",
    "not supported",
    "not available",
    "unknown",
    "unknown error",
    "insufficient permissions",
    "requested functionality has been deprecated",
}


def system_name():
    """"Windows" | "Darwin" | "Linux" | other, without platform.uname().

    On Windows, Python 3.12's platform.system()/machine()/release() run WMI
    queries, which can be slow or hang on a damaged WMI repository.
    """
    p = sys.platform
    if p.startswith("win"):
        return "Windows"
    if p == "darwin":
        return "Darwin"
    if p.startswith("linux"):
        return "Linux"
    return platform.system()


def machine():
    """The process's machine string (platform.machine() without WMI on Windows)."""
    if os.name == "nt":
        return os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get("PROCESSOR_ARCHITECTURE") or ""
    try:
        return os.uname().machine
    except Exception:
        return platform.machine()


def clean_na(value):
    """Return a stripped string, or None for empty / N/A-style placeholders."""
    if value is None:
        return None
    text = str(value).strip()
    if text.lower().strip("[]() ") in _NA_VALUES:
        return None
    return text


def to_int(value):
    """Best-effort integer conversion. Never raises; returns None when unsure.

    Accepts ints (but not bools), finite floats, decimal strings ("49140",
    "49140 MiB", "3,600"), and hexadecimal strings ("0x10de").
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) else None
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("ascii", "replace")
        except Exception:
            return None
    text = clean_na(value)
    if text is None:
        return None
    text = text.replace(",", "").replace("_", "")
    m = re.fullmatch(r"\s*0[xX]([0-9a-fA-F]+)\s*", text)
    if m:
        return int(m.group(1), 16)
    m = re.match(r"\s*([-+]?\d+(?:\.\d+)?)", text)
    if not m:
        return None
    try:
        return int(float(m.group(1)))
    except (ValueError, OverflowError):
        return None


def to_float(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    text = clean_na(value)
    if text is None:
        return None
    m = re.match(r"\s*([-+]?\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else None


def parse_size_text(text):
    """Parse sizes such as "4 GB", "1536 MB", "8192MB", "16 GiB" into bytes.

    macOS and Windows tools use binary multiples while labeling them "MB"/"GB",
    so this treats KB/MB/GB/TB as powers of 1024.
    """
    t = clean_na(text)
    if t is None:
        return None
    m = re.match(r"\s*([\d.,]+)\s*([KMGT])i?B?\b", t, re.IGNORECASE)
    if not m:
        n = to_int(t)
        return n
    try:
        number = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    mult = {"K": KiB, "M": MiB, "G": GiB, "T": TiB}[m.group(2).upper()]
    return int(number * mult)


def fmt_gb(n, *, decimals_below=10):
    """Human size in binary GB (what operating systems show as "GB").

    Under 10 GB keeps one decimal ("7.4 GB"), otherwise whole numbers ("48 GB");
    1000 GB and above switch to TB.
    """
    if n is None:
        return "unknown"
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "unknown"
    gb = n / GiB
    if gb >= 1000:
        return f"{n / TiB:.1f} TB"
    if gb < decimals_below:
        text = f"{gb:.1f}"
        if text.endswith(".0"):
            text = text[:-2]
        return f"{text} GB"
    return f"{round(gb):d} GB"


def truncate(text, limit=300):
    if text is None:
        return None
    text = str(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def first_line(text, limit=200):
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return truncate(line, limit)
    return None


def norm_pci_bus_id(value):
    """Normalize "00000000:01:00.0", "0000:01:00.0" or "01:00.0" to "0000:01:00.0"."""
    text = clean_na(value)
    if text is None:
        return None
    m = re.search(
        r"(?:([0-9a-fA-F]{1,8}):)?([0-9a-fA-F]{1,2}):([0-9a-fA-F]{1,2})\.([0-7])\b", text
    )
    if not m:
        return None
    domain = int(m.group(1), 16) if m.group(1) else 0
    return f"{domain:04x}:{int(m.group(2), 16):02x}:{int(m.group(3), 16):02x}.{int(m.group(4))}"


def short_name(name):
    """Shorten a CPU/GPU marketing name for one-line summaries."""
    if not name:
        return name
    s = re.sub(r"\((?:R|TM|tm|r)\)|®|™", "", str(name))
    s = re.sub(r"\s+CPU\s*@\s*[\d.]+\s*GHz", "", s)
    s = re.sub(r"\s*@\s*[\d.]+\s*GHz", "", s)
    s = re.sub(r"\s+\d+-Core(?:s)?(?: Processor)?\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+(?:with Radeon Graphics|Processor)\b", "", s)
    s = re.sub(r"\s+Generation$", "", s)
    s = re.sub(r"^\d+(?:st|nd|rd|th) Gen\s+", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s or str(name)
