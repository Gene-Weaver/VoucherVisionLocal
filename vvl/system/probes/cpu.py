"""CPU: model, vendor, cores, max clock, and the instruction sets llama.cpp cares about.

Cheap native sources first: /proc/cpuinfo on Linux, sysctl on macOS, the
registry plus IsProcessorFeaturePresent on Windows. py-cpuinfo is only a
fallback, and it always runs in a separate, time-limited Python process,
because in-process it is slow and on Windows it starts helper processes of its
own.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

from .._util import to_float, to_int

# Contract keys first, then extras that help choose a llama.cpp CPU build.
FEATURE_KEYS = ("avx", "avx2", "avx512", "fma", "f16c", "neon",
                "avx_vnni", "avx512_vnni", "amx", "sve", "dotprod", "i8mm")
X86_KEYS = ("avx", "avx2", "avx512", "fma", "f16c", "avx_vnni", "avx512_vnni", "amx")
ARM_KEYS = ("neon", "sve", "dotprod", "i8mm")

# Above this, a reported "max MHz" is a firmware/driver artifact (for example
# AMD CPPC kernels reporting ~7000 MHz for 4.5 GHz parts).
PLAUSIBLE_MAX_MHZ = 6600

ARM_IMPLEMENTERS = {
    0x41: "ARM", 0x42: "Broadcom", 0x43: "Cavium", 0x46: "Fujitsu", 0x48: "HiSilicon",
    0x4E: "NVIDIA", 0x50: "APM", 0x51: "Qualcomm", 0x53: "Samsung", 0x61: "Apple",
    0x6D: "Microsoft", 0xC0: "Ampere",
}
ARM_PARTS = {
    (0x41, 0xD03): "Cortex-A53", (0x41, 0xD04): "Cortex-A35", (0x41, 0xD05): "Cortex-A55",
    (0x41, 0xD07): "Cortex-A57", (0x41, 0xD08): "Cortex-A72", (0x41, 0xD09): "Cortex-A73",
    (0x41, 0xD0A): "Cortex-A75", (0x41, 0xD0B): "Cortex-A76", (0x41, 0xD0C): "Neoverse-N1",
    (0x41, 0xD0D): "Cortex-A77", (0x41, 0xD40): "Neoverse-V1", (0x41, 0xD41): "Cortex-A78",
    (0x41, 0xD44): "Cortex-X1", (0x41, 0xD46): "Cortex-A510", (0x41, 0xD47): "Cortex-A710",
    (0x41, 0xD48): "Cortex-X2", (0x41, 0xD49): "Neoverse-N2", (0x41, 0xD4B): "Cortex-A78C",
    (0x41, 0xD4D): "Cortex-A715", (0x41, 0xD4E): "Cortex-X3", (0x41, 0xD4F): "Neoverse-V2",
    (0x41, 0xD80): "Cortex-A520", (0x41, 0xD81): "Cortex-A720", (0x41, 0xD82): "Cortex-X4",
    (0x41, 0xD84): "Neoverse-V3", (0x41, 0xD85): "Cortex-X925", (0x41, 0xD87): "Cortex-A725",
    (0xC0, 0xAC3): "Ampere-1", (0xC0, 0xAC4): "Ampere-1a",
}


def empty_features(value=None):
    return {k: value for k in FEATURE_KEYS}


def features_from_x86_flags(flags):
    f = {t.lower() for t in flags}
    out = empty_features(False)
    out.update(
        avx="avx" in f,
        avx2="avx2" in f,
        avx512="avx512f" in f,
        fma="fma" in f,
        f16c="f16c" in f,
        avx_vnni="avx_vnni" in f,
        avx512_vnni="avx512_vnni" in f or "avx512vnni" in f,
        amx="amx_tile" in f or "amx_int8" in f,
    )
    return out


def features_from_arm_flags(flags):
    """ARM feature tokens from /proc/cpuinfo "Features" (aarch64 or armv7).

    The x86 keys are False on ARM: those instruction sets do not exist there.
    """
    f = {t.lower() for t in flags}
    out = empty_features(False)
    out.update(
        neon="asimd" in f or "neon" in f,
        sve="sve" in f,
        dotprod="asimddp" in f,
        i8mm="i8mm" in f,
    )
    return out


def parse_proc_cpuinfo(text):
    """Parse /proc/cpuinfo for x86 and ARM kernels."""
    blocks = [b for b in re.split(r"\n\s*\n", text or "") if b.strip()]
    procs, extra = [], {}
    legacy_model = None
    for block in blocks:
        fields = {}
        for line in block.splitlines():
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key, val = key.strip(), val.strip()
            if key == "Processor" and not val.isdigit():
                legacy_model = val  # old ARM kernels: "Processor : AArch64 Processor rev 4"
                continue
            fields[key.lower()] = val
        if "processor" in fields and fields["processor"].isdigit():
            procs.append(fields)
        else:
            extra.update(fields)
    first = procs[0] if procs else extra

    flags_text = first.get("flags") or first.get("features") or extra.get("features") or ""
    is_arm = ("cpu implementer" in first or "features" in first or "features" in extra) and "flags" not in first

    model = first.get("model name") or extra.get("model name") or first.get("cpu model")
    vendor = first.get("vendor_id") or None
    implementer = to_int(first.get("cpu implementer"))
    parts = []
    for p in procs:
        key = (to_int(p.get("cpu implementer")), to_int(p.get("cpu part")))
        if key[1] is not None and key not in parts:
            parts.append(key)
    if is_arm:
        if implementer is not None:
            vendor = ARM_IMPLEMENTERS.get(implementer, f"ARM implementer 0x{implementer:02x}")
        if not model:
            model = extra.get("model") or None  # Raspberry Pi: "Raspberry Pi 4 Model B Rev 1.4"
        if not model and parts:
            names = [ARM_PARTS.get(k) for k in parts]
            if all(names):
                model = " + ".join(dict.fromkeys(names))
        if not model:
            model = extra.get("hardware") or legacy_model

    cores = set()
    for p in procs:
        if "physical id" in p and "core id" in p:
            cores.add((p["physical id"], p["core id"]))

    return {
        "model": model or None,
        "vendor": vendor,
        "flags": flags_text.split(),
        "is_arm": is_arm,
        "logical": len(procs) or None,
        "physical": len(cores) or None,
        "mhz": to_float(first.get("cpu mhz")),
    }


def linux_max_mhz(sys_root="/"):
    """Highest cpuinfo_max_freq across CPUs (kHz in sysfs), if plausible."""
    best = None
    pattern = os.path.join(sys_root, "sys/devices/system/cpu/cpu[0-9]*/cpufreq/cpuinfo_max_freq")
    for path in glob.glob(pattern)[:1024]:
        try:
            with open(path, "r") as fh:
                khz = to_int(fh.read())
        except OSError:
            continue
        if khz:
            mhz = khz / 1000.0
            best = mhz if best is None or mhz > best else best
    if best is not None and best > PLAUSIBLE_MAX_MHZ:
        return None
    return best


def mac_features(values):
    """Features from macOS sysctl values (dict of name -> str/int)."""
    if values.get("hw.optional.arm64") in (1, "1"):
        out = empty_features(False)
        out["neon"] = True
        dp = values.get("hw.optional.arm.FEAT_DotProd")
        i8 = values.get("hw.optional.arm.FEAT_I8MM")
        sve = values.get("hw.optional.arm.FEAT_SVE")
        out["dotprod"] = None if dp is None else bool(to_int(dp))
        out["i8mm"] = None if i8 is None else bool(to_int(i8))
        out["sve"] = False if sve is None else bool(to_int(sve))
        return out
    tokens = set()
    for key in ("machdep.cpu.features", "machdep.cpu.leaf7_features"):
        tokens.update(t.upper() for t in str(values.get(key) or "").split())
    if not tokens and all(values.get(k) is None for k in ("hw.optional.avx2_0", "hw.optional.avx1_0")):
        return empty_features(None)
    out = empty_features(False)

    def opt(name):
        return bool(to_int(values.get(name)))

    out["avx"] = "AVX1.0" in tokens or opt("hw.optional.avx1_0")
    out["avx2"] = "AVX2" in tokens or opt("hw.optional.avx2_0")
    out["avx512"] = "AVX512F" in tokens or opt("hw.optional.avx512f")
    out["fma"] = "FMA" in tokens or opt("hw.optional.fma")
    out["f16c"] = "F16C" in tokens
    out["avx512_vnni"] = "AVX512VNNI" in tokens
    return out


def windows_features(pf, arch):
    """Features from IsProcessorFeaturePresent results.

    Windows offers no FMA/F16C query, so those stay None (unknown). If every
    SSSE3..AVX2 query is False on x86-64, this Windows build predates those
    constants and nothing can be concluded: everything is None.
    """
    out = empty_features(None)
    if not pf:
        return out
    if arch == "arm64":
        out.update({k: False for k in X86_KEYS})
        out["neon"] = bool(pf.get("arm_neon") or pf.get("arm_v8"))
        out["dotprod"] = bool(pf.get("arm_v82_dp"))
        out["sve"] = bool(pf.get("arm_sve"))
        return out
    probe_keys = ("ssse3", "sse4_1", "sse4_2", "avx", "avx2")
    if not any(pf.get(k) for k in probe_keys):
        return out
    out.update(avx=bool(pf.get("avx")), avx2=bool(pf.get("avx2")),
               avx512=bool(pf.get("avx512f")), neon=False, sve=False, dotprod=False, i8mm=False)
    return out


# Runs in a separate, time-limited Python process. That process is already the
# isolation py-cpuinfo wants, so it is told not to start helper processes of its
# own (its get_cpu_info() would launch two more, with visible console windows on
# Windows).
PY_CPUINFO_CHILD = r"""
import io, json, sys
real_stdout = sys.stdout
try:
    from cpuinfo import cpuinfo as m
    m.CAN_CALL_CPUID_IN_SUBPROCESS = False
    sys.stdout = io.StringIO()
    info = {}
    try:
        info = m._get_cpu_info_from_cpuid() or {}
        if isinstance(info, dict) and "info" in info and isinstance(info["info"], dict):
            info = info["info"]
    except Exception:
        info = {}
    if not info.get("flags"):
        try:
            info = m._get_cpu_info_internal() or {}
        except Exception:
            pass
    arch = info.get("arch")
    if not arch:
        try:
            arch = m._parse_arch(m.DataSource.arch_string_raw)[0]
        except Exception:
            arch = None
    sys.stdout = real_stdout
    print(json.dumps({"brand_raw": info.get("brand_raw"), "vendor_id_raw": info.get("vendor_id_raw"),
                      "flags": info.get("flags") or [], "arch": arch}))
except Exception as exc:
    sys.stdout = real_stdout
    print(json.dumps({"error": repr(exc)}))
"""


def py_cpuinfo_fallback(run):
    """Run py-cpuinfo in a separate, time-limited process. Returns dict or None."""
    if getattr(sys, "frozen", False) or not sys.executable:
        return None
    res = run([sys.executable, "-I", "-c", PY_CPUINFO_CHILD], timeout=5.0)
    if not res.ok:
        return None
    try:
        data = json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        return None
    return None if not isinstance(data, dict) or "error" in data else data


def _psutil_counts():
    try:
        import psutil

        return psutil.cpu_count(logical=False), psutil.cpu_count(logical=True)
    except Exception:
        return None, os.cpu_count()


def _psutil_max_mhz():
    try:
        import psutil

        freq = psutil.cpu_freq()
        if freq and freq.max and 0 < freq.max <= PLAUSIBLE_MAX_MHZ:
            return float(freq.max)
    except Exception:
        pass
    return None


def _round_mhz(v):
    return None if v is None else int(round(v))


def probe(ctx, log):
    physical, logical = _psutil_counts()
    out = {
        "model": None,
        "vendor": None,
        "physical_cores": physical,
        "logical_cores": logical,
        "max_mhz": None,
        "features": empty_features(None),
        "features_source": None,
    }
    system = ctx.system

    if system == "Linux":
        text = None
        try:
            with open(os.path.join(ctx.sys_root, "proc/cpuinfo"), "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read(4_000_000)
        except OSError as exc:
            log.fail(f"could not read /proc/cpuinfo: {exc}")
        if text:
            info = parse_proc_cpuinfo(text)
            out["model"], out["vendor"] = info["model"], info["vendor"]
            out["logical_cores"] = out["logical_cores"] or info["logical"]
            out["physical_cores"] = out["physical_cores"] or info["physical"]
            if info["flags"]:
                out["features"] = (features_from_arm_flags if info["is_arm"] else features_from_x86_flags)(info["flags"])
                out["features_source"] = "proc_cpuinfo"
            if not out["model"] and info["is_arm"]:
                dt = None
                try:
                    with open(os.path.join(ctx.sys_root, "sys/firmware/devicetree/base/model"), "rb") as fh:
                        dt = fh.read(256).split(b"\0", 1)[0].decode("utf-8", "replace").strip()
                except OSError:
                    pass
                out["model"] = dt or None
        out["max_mhz"] = _round_mhz(linux_max_mhz(ctx.sys_root) or _psutil_max_mhz())

    elif system == "Darwin":
        from . import _macos

        names = ["machdep.cpu.brand_string", "machdep.cpu.vendor", "hw.optional.arm64",
                 "hw.physicalcpu", "hw.logicalcpu", "hw.cpufrequency_max",
                 "machdep.cpu.features", "machdep.cpu.leaf7_features",
                 "hw.optional.avx1_0", "hw.optional.avx2_0", "hw.optional.avx512f", "hw.optional.fma",
                 "hw.optional.arm.FEAT_DotProd", "hw.optional.arm.FEAT_I8MM", "hw.optional.arm.FEAT_SVE"]
        string_names = {"machdep.cpu.brand_string", "machdep.cpu.vendor", "machdep.cpu.features",
                        "machdep.cpu.leaf7_features"}
        values = {}
        for n in names:
            values[n] = (_macos.sysctl_str if n in string_names else _macos.sysctl_int)(n, ctx.run)
        apple = values.get("hw.optional.arm64") == 1
        out["model"] = values.get("machdep.cpu.brand_string")
        out["vendor"] = "Apple" if apple else values.get("machdep.cpu.vendor")
        out["physical_cores"] = values.get("hw.physicalcpu") or out["physical_cores"]
        out["logical_cores"] = values.get("hw.logicalcpu") or out["logical_cores"]
        hz = values.get("hw.cpufrequency_max")
        out["max_mhz"] = _round_mhz(hz / 1e6) if hz and not apple else None
        out["features"] = mac_features(values)
        out["features_source"] = "sysctl"

    elif system == "Windows":
        from . import _windows
        from .os_info import normalize_arch

        reg = _windows.reg_values("HKEY_LOCAL_MACHINE", r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") or {}
        name = str(reg.get("ProcessorNameString") or "").strip()
        out["model"] = re.sub(r"\s{2,}", " ", name) or ctx.env.get("PROCESSOR_IDENTIFIER") or None
        out["vendor"] = (str(reg.get("VendorIdentifier") or "").strip() or None)
        arch = normalize_arch(_windows.native_machine())
        pf = _windows.processor_features()
        out["features"] = windows_features(pf, arch)
        if pf is not None:
            out["features_source"] = "windows_api"
        out["max_mhz"] = _round_mhz(_psutil_max_mhz() or to_float(reg.get("~MHz")))

    if out["features"].get("avx2") is None and out["features"].get("neon") is None:
        data = py_cpuinfo_fallback(ctx.run)
        if data and data.get("flags"):
            arch = str(data.get("arch") or "").upper()
            is_arm = arch.startswith("ARM")
            out["features"] = (features_from_arm_flags if is_arm else features_from_x86_flags)(data["flags"])
            out["features_source"] = "py-cpuinfo"
            out["model"] = out["model"] or data.get("brand_raw")
            out["vendor"] = out["vendor"] or data.get("vendor_id_raw")
        else:
            log.fail("CPU instruction-set flags could not be read")

    if out["model"]:
        out["model"] = re.sub(r"\s{2,}", " ", str(out["model"])).strip()
    return out
