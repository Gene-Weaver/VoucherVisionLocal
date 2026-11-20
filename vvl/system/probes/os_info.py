"""Operating system: friendly name, version, kernel, architecture, WSL,
Apple Silicon (including a Rosetta-translated Python), Python version."""

from __future__ import annotations

import os
import platform
import re
import sys
import sysconfig

from .._util import machine as _machine
from .._util import to_int

MAC_CODENAMES = {
    "10.13": "High Sierra",
    "10.14": "Mojave",
    "10.15": "Catalina",
    "11": "Big Sur",
    "12": "Monterey",
    "13": "Ventura",
    "14": "Sonoma",
    "15": "Sequoia",
    "26": "Tahoe",
}


def normalize_arch(machine):
    """Map platform.machine()-style strings to "x86_64" | "arm64" (others pass through)."""
    if not machine:
        return None
    m = str(machine).strip().lower()
    if m in ("x86_64", "amd64", "x64", "em64t", "intel64"):
        return "x86_64"
    if m in ("arm64", "aarch64", "arm64e") or m.startswith("armv8") or m.startswith("aarch64"):
        return "arm64"
    if m in ("i386", "i486", "i586", "i686", "x86"):
        return "x86"
    return m


def python_arch_from_platform_tag(tag, machine):
    """Architecture the Python interpreter itself was built for."""
    t = (tag or "").lower()
    if t.startswith("win"):
        if "arm64" in t:
            return "arm64"
        if "amd64" in t:
            return "x86_64"
        if t == "win32":
            return "x86"
    return normalize_arch(machine)


def parse_os_release(text):
    """Parse /etc/os-release (KEY=VALUE, optional quotes and backslash escapes)."""
    out = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z0-9_]+", key):
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            quote, val = val[0], val[1:-1]
            if quote == '"':
                val = re.sub(r'\\(["\\$`])', r"\1", val)
        out[key] = val
    return out


def linux_name_version(os_release):
    name = os_release.get("PRETTY_NAME") or None
    if not name:
        base = os_release.get("NAME")
        ver = os_release.get("VERSION") or os_release.get("VERSION_ID")
        name = f"{base} {ver}".strip() if base else None
    version = os_release.get("VERSION_ID") or os_release.get("BUILD_ID") or None
    return name or "Linux", version


def wsl_info(kernel_release, proc_version, env):
    """(is_wsl, wsl_version) from the kernel strings and environment."""
    k = (kernel_release or "").lower()
    v = (proc_version or "").lower()
    env = env or {}
    marked = "microsoft" in k or "microsoft" in v or "wsl" in k
    has_env = bool(env.get("WSL_DISTRO_NAME") or env.get("WSL_INTEROP"))
    if not marked and not has_env:
        return False, None
    if "wsl2" in k or "microsoft-standard" in k:
        return True, 2
    if re.search(r"\d+-microsoft\b", k):  # WSL 1 kernels look like 4.4.0-19041-Microsoft
        return True, 1
    if env.get("WSL_INTEROP"):  # only WSL 2 sets WSL_INTEROP
        return True, 2
    return True, None


def windows_name_version(values, release="", version=""):
    """Friendly Windows name from HKLM\\...\\Windows NT\\CurrentVersion values.

    ProductName still says "Windows 10" on Windows 11, so builds >= 22000 are
    renamed. Returns (name, version).
    """
    values = values or {}
    product = (values.get("ProductName") or "").strip() or None
    build = to_int(values.get("CurrentBuildNumber") or values.get("CurrentBuild"))
    if build is None:
        m = re.search(r"\d+\.\d+\.(\d+)", version or "")
        build = int(m.group(1)) if m else None
    if not product:
        product = f"Windows {release}".strip() if release else "Windows"
    if build is not None and build >= 22000 and "Windows 10" in product:
        product = product.replace("Windows 10", "Windows 11")
    display = (values.get("DisplayVersion") or values.get("ReleaseId") or "").strip()
    name = f"{product} {display}".strip() if display else product
    major = to_int(values.get("CurrentMajorVersionNumber"))
    minor = to_int(values.get("CurrentMinorVersionNumber"))
    if build is not None:
        ver = f"{major if major is not None else 10}.{minor if minor is not None else 0}.{build}"
        ubr = to_int(values.get("UBR"))
        if ubr is not None:
            ver += f".{ubr}"
    else:
        ver = version or None
    return name, ver


def mac_name(product_version):
    if not product_version:
        return "macOS"
    parts = product_version.split(".")
    key = ".".join(parts[:2]) if parts[0] == "10" else parts[0]
    code = MAC_CODENAMES.get(key)
    return f"macOS {product_version}" + (f" {code}" if code else "")


def _read(path, limit=65536):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(limit)
    except OSError:
        return None


def _windows_version_string():
    """"10.0.22631" from sys.getwindowsversion() (no WMI, unlike platform.version())."""
    try:
        v = sys.getwindowsversion()
        return f"{v.major}.{v.minor}.{v.build}", str(v.major)
    except Exception:
        return "", ""


def probe(ctx, log):
    system = ctx.system
    machine = _machine()
    out = {
        "system": system or None,
        "name": None,
        "version": None,
        "kernel": None,
        "arch": normalize_arch(machine),
        "python_arch": python_arch_from_platform_tag(sysconfig.get_platform(), machine),
        "is_translated": False,
        "is_wsl": False,
        "wsl_version": None,
        "wsl_gpu_passthrough": None,
        "is_apple_silicon": False,
        "python": platform.python_version(),
    }

    if system == "Linux":
        root = ctx.sys_root
        text = _read(os.path.join(root, "etc/os-release")) or _read(os.path.join(root, "usr/lib/os-release"))
        if text is None:
            log.fail("os-release not found")
        out["name"], out["version"] = linux_name_version(parse_os_release(text or ""))
        out["kernel"] = platform.release() or None
        osrel = _read(os.path.join(root, "proc/sys/kernel/osrelease")) or out["kernel"]
        is_wsl, wsl_ver = wsl_info(osrel, _read(os.path.join(root, "proc/version")), ctx.env)
        out["is_wsl"], out["wsl_version"] = is_wsl, wsl_ver
        if is_wsl:
            out["wsl_gpu_passthrough"] = os.path.exists(os.path.join(root, "dev/dxg"))
    elif system == "Windows":
        from . import _windows

        values = _windows.reg_values("HKEY_LOCAL_MACHINE", r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
        if values is None:
            log.fail("could not read the Windows version from the registry")
        winver, major = _windows_version_string()
        out["name"], out["version"] = windows_name_version(values, major, winver)
        out["kernel"] = winver or None
        native = normalize_arch(_windows.native_machine())
        if native:
            out["arch"] = native
        out["is_translated"] = bool(native and out["python_arch"] and native != out["python_arch"])
    elif system == "Darwin":
        from . import _macos

        ver = platform.mac_ver()[0] or ""
        if not ver or ver.startswith("10.16"):
            # Old SDK compatibility mode reports 10.16; ask sw_vers instead.
            res = ctx.run(["/usr/bin/sw_vers", "-productVersion"], timeout=3.0)
            if res.ok and res.stdout.strip():
                ver = res.stdout.strip()
        out["version"] = ver or None
        out["name"] = mac_name(ver)
        out["kernel"] = platform.release() or None
        native_arm64, translated = _macos.arch_info(ctx.run)
        if native_arm64 is None:
            native_arm64 = out["arch"] == "arm64"
        out["is_apple_silicon"] = bool(native_arm64)
        out["is_translated"] = bool(translated)
        out["arch"] = "arm64" if native_arm64 else out["arch"]
    else:
        out["name"] = f"{system} {platform.release()}".strip() or None
        out["version"] = platform.release() or None
        out["kernel"] = platform.version() or None
    return out
