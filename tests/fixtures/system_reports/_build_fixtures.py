"""Regenerate the fake full system reports in this folder.

    uv run python tests/fixtures/system_reports/_build_fixtures.py

Each machine is described the way its probes would see it: raw tool output
(nvidia-smi CSV, system_profiler JSON, registry values, sysfs + lspci,
vulkaninfo) goes through the real parsers, the OS/CPU/memory/disk sections are
written by hand, and ``vvl.system.report.assemble`` produces the report. The
tests check that the stored JSON still matches what the code produces, so run
this after an intentional change to ``policy.py`` or ``capability.py``.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SAMPLES = os.path.join(os.path.dirname(HERE), "system_samples")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from vvl.system import report as report_mod  # noqa: E402
from vvl.system.probes import cpu as cpu_mod  # noqa: E402
from vvl.system.probes import gpu_apple, gpu_linux, gpu_nvidia, gpu_windows, vulkan  # noqa: E402
from vvl.system.runner import ProbeContext, ProbeOutcome  # noqa: E402

GiB = 1024**3
GENERATED_AT = "2026-11-19T18:04:11Z"


def sample(name):
    with open(os.path.join(SAMPLES, name), "r", encoding="utf-8") as fh:
        return fh.read()


def features(**on):
    f = cpu_mod.empty_features(False)
    f.update(on)
    return f


def nvidia_value(csv_name, driver, cuda, *, arch="x86_64", system_ram=None):
    rows = gpu_nvidia.parse_query_csv(sample(csv_name))
    return {"state": "ok", "message": None, "gpus": gpu_nvidia.gpus_from_rows(rows, arch=arch, system_ram=system_ram),
            "driver_version": driver, "cuda_version": cuda, "path": "nvidia-smi"}


NVIDIA_NOT_FOUND = {"state": "not_found", "message": "nvidia-smi not found", "gpus": [], "driver_version": None,
                    "cuda_version": None, "path": None}


def vulkan_value(text_name=None, *, returncode=0, missing_detail=None):
    if text_name is None:
        return {"available": None, "detail": missing_detail or "vulkaninfo not found", "devices": [], "tool": False}
    from vvl.system._proc import CmdResult

    res = CmdResult(argv=["vulkaninfo", "--summary"], returncode=returncode, stdout=sample(text_name))
    available, detail, devices = vulkan.assess(res)
    return {"available": available, "detail": detail, "devices": devices, "tool": True}


def os_section(system, name, version, kernel, arch, *, python_arch=None, translated=False, wsl=None,
               apple_silicon=False):
    return {
        "system": system, "name": name, "version": version, "kernel": kernel, "arch": arch,
        "python_arch": python_arch or arch, "is_translated": translated,
        "is_wsl": wsl is not None, "wsl_version": wsl, "wsl_gpu_passthrough": True if wsl == 2 else None,
        "is_apple_silicon": apple_silicon, "python": "3.12.15",
    }


def disk_section(models_dir, source, exists, checked, total, free, *, synced=None, fs="ext4", network=False):
    return {
        "models_dir": models_dir, "models_dir_source": source, "exists": exists, "checked_path": checked,
        "total_bytes": total, "free_bytes": free, "is_synced_folder": synced is not None,
        "synced_folder_hint": synced, "writable": True, "filesystem": fs, "is_network": network,
    }


def win_reg(name, provider, version, pci, qw=None, mem32=None, location=None):
    v = {"DriverDesc": name, "ProviderName": provider, "DriverVersion": version, "MatchingDeviceId": pci,
         "_instance_id": pci.upper() + "\\4&2d78ab8f&0&0008", "_location": location, "_subkey": "0000"}
    if qw is not None:
        v["HardwareInformation.qwMemorySize"] = qw
    if mem32 is not None:
        v["HardwareInformation.MemorySize"] = mem32
    return v


# ------------------------------------------------------------------ machines

def linux_workstation():
    return "Linux", {
        "os": os_section("Linux", "Ubuntu 22.04.5 LTS", "22.04", "6.8.0-138-generic", "x86_64"),
        "cpu": {"model": "AMD Ryzen Threadripper PRO 5975WX 32-Cores", "vendor": "AuthenticAMD",
                "physical_cores": 32, "logical_cores": 64, "max_mhz": 3600,
                "features": features(avx=True, avx2=True, fma=True, f16c=True), "features_source": "proc_cpuinfo"},
        "memory": {"total_bytes": 540653092864, "available_bytes": 464625541120,
                   "swap_total_bytes": 34359734272, "swap_free_bytes": 26517172224},
        "gpu_nvidia": nvidia_value("nvidia_smi_query_2x_rtx6000ada.csv", "560.35.05", "12.6"),
        "gpu_linux": {"gpus": gpu_linux.gpus_from_scan([
            {"addr": "0000:01:00.0", "class": 0x030000, "vendor_id": 0x10DE, "device_id": 0x26B1, "driver": "nvidia",
             "boot_vga": "1", "vram_total": None, "vram_used": None, "gtt_total": None},
            {"addr": "0000:29:00.0", "class": 0x030000, "vendor_id": 0x1A03, "device_id": 0x2000, "driver": "ast",
             "boot_vga": "0", "vram_total": None, "vram_used": None, "gtt_total": None},
            {"addr": "0000:61:00.0", "class": 0x030000, "vendor_id": 0x10DE, "device_id": 0x26B1, "driver": "nvidia",
             "boot_vga": "0", "vram_total": None, "vram_used": None, "gtt_total": None},
        ], {}), "lspci": True},
        "vulkan": vulkan_value(missing_detail="vulkaninfo not found (a Vulkan loader is installed, so Vulkan may work)"),
        "disk": disk_section("/datac/VoucherVisionLocal_Models", "settings", True, "/datac/VoucherVisionLocal_Models",
                             11904377847808, 4321652998144),
    }


def windows_gaming_laptop():
    reg = [
        win_reg("NVIDIA GeForce RTX 4060 Laptop GPU", "NVIDIA", "32.0.15.6094",
                "pci\\ven_10de&dev_28e0&subsys_0b2a1028", qw=8589934592, location="PCI bus 1, device 0, function 0"),
        win_reg("Intel(R) UHD Graphics", "Intel Corporation", "31.0.101.4502",
                "pci\\ven_8086&dev_a7a8&subsys_0b2a1028", qw=134217728,
                location="@System32\\drivers\\pci.sys,#65536;PCI bus %1, device %2, function %3;(0,2,0)"),
    ]
    return "Windows", {
        "os": os_section("Windows", "Windows 11 Home 23H2", "10.0.22631.4460", "10.0.22631", "x86_64"),
        "cpu": {"model": "13th Gen Intel(R) Core(TM) i7-13620H", "vendor": "GenuineIntel",
                "physical_cores": 10, "logical_cores": 16, "max_mhz": 2400,
                "features": cpu_mod.windows_features({"ssse3": True, "sse4_1": True, "sse4_2": True, "avx": True,
                                                       "avx2": True, "avx512f": False}, "x86_64"),
                "features_source": "windows_api"},
        "memory": {"total_bytes": 17005965312, "available_bytes": 9126805504,
                   "swap_total_bytes": 2550136832, "swap_free_bytes": 2400000000},
        "gpu_nvidia": nvidia_value("nvidia_smi_query_rtx4060_laptop_windows.csv", "560.94", "12.6"),
        "gpu_windows": {"gpus": gpu_windows.parse_adapters(reg, presence_checked=True), "source": "registry"},
        "vulkan": vulkan_value("vulkaninfo_summary_windows_nvidia_intel.txt"),
        "disk": disk_section("C:\\Users\\Sam\\AppData\\Local\\VoucherVisionLocal\\models", "default", False,
                             "C:\\Users\\Sam\\AppData\\Local", 1022870155264, 412316860416, fs="NTFS"),
    }


def macbook_air_m2():
    mem = 16 * GiB
    return "Darwin", {
        "os": os_section("Darwin", "macOS 14.5 Sonoma", "14.5", "23.5.0", "arm64", apple_silicon=True),
        "cpu": {"model": "Apple M2", "vendor": "Apple", "physical_cores": 8, "logical_cores": 8, "max_mhz": None,
                "features": cpu_mod.mac_features({"hw.optional.arm64": 1, "hw.optional.arm.FEAT_DotProd": 1,
                                                  "hw.optional.arm.FEAT_I8MM": 1}),
                "features_source": "sysctl"},
        "memory": {"total_bytes": mem, "available_bytes": 7516192768, "swap_total_bytes": 1073741824,
                   "swap_free_bytes": 900000000},
        "gpu_apple": {"gpus": gpu_apple.parse_displays_json(sample("system_profiler_m2_air.json"),
                                                            apple_silicon=True, memsize=mem),
                      "is_apple_silicon": True, "memsize": mem},
        "vulkan": vulkan_value(),
        "disk": disk_section("/Users/sam/Library/Application Support/VoucherVisionLocal/models", "default", False,
                             "/Users/sam/Library/Application Support", 494384795648, 128849018880, fs="apfs"),
    }


def macbook_pro_m3_max_rosetta_dropbox():
    mem = 64 * GiB
    return "Darwin", {
        "os": os_section("Darwin", "macOS 15.1 Sequoia", "15.1", "24.1.0", "arm64", python_arch="x86_64",
                         translated=True, apple_silicon=True),
        "cpu": {"model": "Apple M3 Max", "vendor": "Apple", "physical_cores": 16, "logical_cores": 16,
                "max_mhz": None,
                "features": cpu_mod.mac_features({"hw.optional.arm64": 1, "hw.optional.arm.FEAT_DotProd": 1,
                                                  "hw.optional.arm.FEAT_I8MM": 1}),
                "features_source": "sysctl"},
        "memory": {"total_bytes": mem, "available_bytes": 41875931136, "swap_total_bytes": 0, "swap_free_bytes": 0},
        "gpu_apple": {"gpus": gpu_apple.parse_displays_json(sample("system_profiler_m3_max.json"),
                                                            apple_silicon=True, memsize=mem),
                      "is_apple_silicon": True, "memsize": mem},
        "vulkan": vulkan_value(),
        "disk": disk_section("/Users/ana/Library/CloudStorage/Dropbox/VVL_Models", "settings", True,
                             "/Users/ana/Library/CloudStorage/Dropbox/VVL_Models", 994662584320, 412316860416,
                             synced="Dropbox", fs="apfs"),
    }


def office_pc_cpu_only():
    reg = [win_reg("Intel(R) HD Graphics", "Intel Corporation", "10.18.10.5161",
                   "pci\\ven_8086&dev_0152&subsys_309717aa", mem32=b"\x00\x00\x00\x80",
                   location="PCI bus 0, device 2, function 0")]
    return "Windows", {
        "os": os_section("Windows", "Windows 10 Pro 22H2", "10.0.19045.5131", "10.0.19045", "x86_64"),
        "cpu": {"model": "Intel(R) Core(TM) i5-3470 CPU @ 3.20GHz", "vendor": "GenuineIntel",
                "physical_cores": 4, "logical_cores": 4, "max_mhz": 3201,
                "features": cpu_mod.windows_features({"ssse3": True, "sse4_1": True, "sse4_2": True, "avx": True,
                                                       "avx2": False, "avx512f": False}, "x86_64"),
                "features_source": "windows_api"},
        "memory": {"total_bytes": 8484540416, "available_bytes": 3758096384,
                   "swap_total_bytes": 1342177280, "swap_free_bytes": 1000000000},
        "gpu_nvidia": dict(NVIDIA_NOT_FOUND),
        "gpu_windows": {"gpus": gpu_windows.parse_adapters(reg, presence_checked=True), "source": "registry"},
        "vulkan": vulkan_value(missing_detail="vulkaninfo not found (a Vulkan loader is installed, so Vulkan may work)"),
        "disk": disk_section("C:\\Users\\Pat\\OneDrive - State University\\Documents\\VoucherVisionLocal\\models",
                             "settings", False, "C:\\Users\\Pat\\OneDrive - State University\\Documents",
                             255505461248, 61203283968, synced="OneDrive", fs="NTFS"),
    }


def linux_amd_rx7900xtx():
    scan = [
        {"addr": "0000:03:00.0", "class": 0x030000, "vendor_id": 0x1002, "device_id": 0x744C, "driver": "amdgpu",
         "boot_vga": "1", "vram_total": 25753026560, "vram_used": 1073741824, "gtt_total": 33285996544},
        {"addr": "0000:13:00.0", "class": 0x030000, "vendor_id": 0x1002, "device_id": 0x164E, "driver": "amdgpu",
         "boot_vga": "0", "vram_total": 536870912, "vram_used": 61865984, "gtt_total": 33285996544},
    ]
    names = gpu_linux.parse_lspci_mm(sample("lspci_mm_nn_amd_rx7900xtx_raphael.txt"))
    return "Linux", {
        "os": os_section("Linux", "Fedora Linux 40 (Workstation Edition)", "40", "6.10.6-200.fc40.x86_64", "x86_64"),
        "cpu": {"model": "AMD Ryzen 9 7950X 16-Core Processor", "vendor": "AuthenticAMD", "physical_cores": 16,
                "logical_cores": 32, "max_mhz": 5881,
                "features": features(avx=True, avx2=True, avx512=True, fma=True, f16c=True, avx512_vnni=True),
                "features_source": "proc_cpuinfo"},
        "memory": {"total_bytes": 66571993088, "available_bytes": 52613349376,
                   "swap_total_bytes": 8589930496, "swap_free_bytes": 8589930496},
        "gpu_nvidia": dict(NVIDIA_NOT_FOUND),
        "gpu_linux": {"gpus": gpu_linux.gpus_from_scan(scan, names), "lspci": True},
        "vulkan": vulkan_value("vulkaninfo_summary_amd_rx7900xtx_raphael.txt"),
        "disk": disk_section("/home/alex/.local/share/VoucherVisionLocal/models", "default", False,
                             "/home/alex/.local/share", 1998678130688, 858993459200, fs="btrfs"),
    }


def wsl2_rtx3090():
    return "Linux", {
        "os": os_section("Linux", "Ubuntu 22.04.4 LTS", "22.04", "5.15.153.1-microsoft-standard-WSL2", "x86_64",
                         wsl=2),
        "cpu": {"model": "AMD Ryzen 9 5900X 12-Core Processor", "vendor": "AuthenticAMD", "physical_cores": 12,
                "logical_cores": 24, "max_mhz": None,
                "features": features(avx=True, avx2=True, fma=True, f16c=True), "features_source": "proc_cpuinfo"},
        "memory": {"total_bytes": 33613971456, "available_bytes": 31138512896,
                   "swap_total_bytes": 8589934592, "swap_free_bytes": 8589934592},
        "gpu_nvidia": nvidia_value("nvidia_smi_query_rtx3090_wsl2.csv", "566.03", "12.7"),
        "gpu_linux": {"gpus": [], "lspci": False, "no_pci_sysfs": True},
        "vulkan": vulkan_value(),
        "disk": disk_section("/mnt/d/VVL_Models", "settings", True, "/mnt/d/VVL_Models", 2000396742656,
                             1099511627776, fs="9p"),
    }


def intel_mac_2019():
    mem = 32 * GiB
    return "Darwin", {
        "os": os_section("Darwin", "macOS 13.6.7 Ventura", "13.6.7", "22.6.0", "x86_64"),
        "cpu": {"model": "Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz", "vendor": "GenuineIntel", "physical_cores": 8,
                "logical_cores": 16, "max_mhz": 2300,
                "features": cpu_mod.mac_features({
                    "machdep.cpu.features": "FPU VME DE PSE TSC MSR PAE MCE CX8 APIC SEP MTRR PGE MCA CMOV PAT PSE36 "
                                            "CLFSH DS ACPI MMX FXSR SSE SSE2 SS HTT TM PBE SSE3 PCLMULQDQ DTES64 MON "
                                            "DSCPL VMX EST TM2 SSSE3 FMA CX16 TPR PDCM SSE4.1 SSE4.2 x2APIC MOVBE "
                                            "POPCNT AES PCID XSAVE OSXSAVE SEGLIM64 TSCTMR AVX1.0 RDRAND F16C",
                    "machdep.cpu.leaf7_features": "RDWRFSGS TSC_THREAD_OFFSET SGX BMI1 HLE AVX2 SMEP BMI2 ERMS INVPCID "
                                                  "RTM FPU_CSDS MPX RDSEED ADX SMAP CLFSOPT IPT MDCLEAR TSXFA IBRS "
                                                  "STIBP L1DF SSBD",
                    "hw.optional.arm64": None}),
                "features_source": "sysctl"},
        "memory": {"total_bytes": mem, "available_bytes": 15032385536, "swap_total_bytes": 2147483648,
                   "swap_free_bytes": 1073741824},
        "gpu_apple": {"gpus": gpu_apple.parse_displays_json(sample("system_profiler_intel_mbp16_2019.json"),
                                                            apple_silicon=False, memsize=mem),
                      "is_apple_silicon": False, "memsize": mem},
        "vulkan": vulkan_value(),
        "disk": disk_section("/Users/lee/Library/Application Support/VoucherVisionLocal/models", "default", False,
                             "/Users/lee/Library/Application Support", 1000240963584, 240518168576, fs="apfs"),
    }


MACHINES = {
    "linux_workstation_2x_rtx6000ada": linux_workstation,
    "windows_laptop_rtx4060_8gb": windows_gaming_laptop,
    "macbook_air_m2_16gb": macbook_air_m2,
    "macbook_pro_m3_max_64gb_rosetta_dropbox": macbook_pro_m3_max_rosetta_dropbox,
    "office_pc_cpu_only_8gb_no_avx2": office_pc_cpu_only,
    "linux_amd_rx7900xtx": linux_amd_rx7900xtx,
    "wsl2_ubuntu_rtx3090": wsl2_rtx3090,
    "intel_mac_2019_radeon_pro": intel_mac_2019,
}

PROBE_MS = {"os": 2, "cpu": 6, "memory": 1, "gpu_nvidia": 48, "gpu_apple": 410, "gpu_windows": 12,
            "gpu_linux": 20, "vulkan": 160, "disk": 3}


def build(name):
    system, values = MACHINES[name]()
    outcomes = {k: ProbeOutcome(k, True, PROBE_MS.get(k, 5), None, v) for k, v in values.items()}
    disk = values["disk"]
    ctx = ProbeContext(system=system, env={}, models_dir=disk["models_dir"], models_dir_source=disk["models_dir_source"])
    duration = max(o.duration_ms for o in outcomes.values()) + 4
    return report_mod.assemble(outcomes, ctx, generated_at=GENERATED_AT, duration_ms=duration)


def main():
    for name in MACHINES:
        path = os.path.join(HERE, f"{name}.json")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(build(name), fh, indent=2)
            fh.write("\n")
        print("wrote", os.path.relpath(path, REPO))


if __name__ == "__main__":
    main()
