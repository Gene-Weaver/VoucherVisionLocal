"""Vulkan through ``vulkaninfo --summary`` (from the Vulkan SDK / vulkan-tools).

``available`` is True when a hardware Vulkan device is listed, False when
vulkaninfo ran and found none (or no driver), and None when it could not be
determined (vulkaninfo missing, hung, or failed in an unrecognized way).
"""

from __future__ import annotations

import os
import re

from .. import _proc
from .._util import first_line, to_int

DEVICE_TYPES = {
    "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU": "discrete",
    "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU": "integrated",
    "PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU": "virtual",
    "PHYSICAL_DEVICE_TYPE_CPU": "cpu",
    "PHYSICAL_DEVICE_TYPE_OTHER": "other",
}
_KEYS = {"devicename": "name", "devicetype": "type", "vendorid": "vendor_id", "deviceid": "device_id",
         "apiversion": "api_version", "drivername": "driver_name", "driverinfo": "driver_info"}

LINUX_LOADERS = [
    "usr/lib/x86_64-linux-gnu/libvulkan.so.1", "usr/lib/aarch64-linux-gnu/libvulkan.so.1",
    "usr/lib64/libvulkan.so.1", "usr/lib/libvulkan.so.1",
]


def candidates(system, env):
    out = []
    sdk = env.get("VULKAN_SDK")
    if system == "Windows":
        root = env.get("SystemRoot") or r"C:\Windows"
        out += [os.path.join(root, "System32", "vulkaninfo.exe")]
        if sdk:
            out += [os.path.join(sdk, "Bin", "vulkaninfoSDK.exe"), os.path.join(sdk, "Bin", "vulkaninfo.exe")]
    else:
        out += ["/usr/bin/vulkaninfo", "/usr/local/bin/vulkaninfo", "/opt/homebrew/bin/vulkaninfo"]
        if sdk:
            out += [os.path.join(sdk, "bin", "vulkaninfo")]
    return out


def parse_summary(text):
    """{"instance_version": str|None, "devices": [...]} from ``vulkaninfo --summary``."""
    devices, cur = [], None
    in_devices = False
    for line in (text or "").splitlines():
        s = line.strip()
        if re.fullmatch(r"Devices:?", s):
            in_devices = True
            continue
        m = re.fullmatch(r"GPU\s*(\d+)\s*:?", s)
        if m:
            cur = {"index": int(m.group(1))}
            devices.append(cur)
            in_devices = True
            continue
        if cur is not None and in_devices and "=" in s:
            key, _, val = s.partition("=")
            k = _KEYS.get(key.strip().lower())
            if k:
                cur[k] = val.strip()
    for d in devices:
        d["type"] = DEVICE_TYPES.get(str(d.get("type", "")).strip(), "other" if d.get("type") else None)
        d["vendor_id"] = to_int(d.get("vendor_id"))
        d["device_id"] = to_int(d.get("device_id"))
    m = re.search(r"Vulkan Instance Version:\s*([0-9.]+)", text or "")
    return {"instance_version": m.group(1) if m else None, "devices": devices}


_NO_DRIVER = re.compile(r"ERROR_INCOMPATIBLE_DRIVER|vkCreateInstance failed|Cannot create Vulkan instance|"
                        r"failed to find a usable vulkan|No Vulkan devices|ERROR_INITIALIZATION_FAILED", re.I)


def assess(res):
    """(available, detail, devices) from a CmdResult of vulkaninfo --summary."""
    parsed = parse_summary(res.stdout)
    devices = parsed["devices"]
    hardware = [d for d in devices if d.get("type") not in ("cpu", None)]
    if hardware:
        names = ", ".join(d.get("name") or "unknown device" for d in hardware)
        ver = parsed["instance_version"]
        return True, f"{names}" + (f" (Vulkan {ver})" if ver else ""), devices
    if devices and res.returncode == 0:
        return False, "Only a software (CPU) Vulkan driver was found.", devices
    if _NO_DRIVER.search(res.output):
        return False, "No working Vulkan driver was found.", devices
    if res.ok:
        return False, "vulkaninfo listed no Vulkan devices.", devices
    return None, f"vulkaninfo failed: {first_line(res.output) or res.describe_failure()}", devices


def loader_present(ctx):
    if ctx.system == "Windows":
        root = ctx.env.get("SystemRoot") or r"C:\Windows"
        return os.path.isfile(os.path.join(root, "System32", "vulkan-1.dll"))
    if ctx.system == "Linux":
        return any(os.path.exists(os.path.join(ctx.sys_root, p)) for p in LINUX_LOADERS)
    return False


def probe(ctx, log):
    exe = _proc.find_tool("vulkaninfo", candidates(ctx.system, ctx.env))
    if not exe:
        detail = "vulkaninfo not found"
        if loader_present(ctx):
            detail += " (a Vulkan loader is installed, so Vulkan may work)"
        return {"available": None, "detail": detail, "devices": [], "tool": False}
    res = ctx.run([exe, "--summary"], timeout=5.0)
    if res.timed_out:
        log.fail("vulkaninfo did not finish within 5 seconds")
        return {"available": None, "detail": "vulkaninfo did not respond", "devices": [], "tool": True}
    available, detail, devices = assess(res)
    return {"available": available, "detail": detail, "devices": devices, "tool": True}
