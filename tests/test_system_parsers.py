"""Parsers for every platform path, fed with captured or realistic tool output.

Command execution is separate from parsing, so none of these tests needs the
tool (or the OS) it describes.
"""

from __future__ import annotations

import json
import os

import pytest

from vvl.system import gpus as gpus_mod
from vvl.system._proc import CmdResult
from vvl.system._util import GiB, MiB, norm_pci_bus_id, parse_size_text, short_name, to_int
from vvl.system.probes import cpu, gpu_apple, gpu_linux, gpu_nvidia, gpu_windows, os_info, vulkan

SAMPLES = os.path.join(os.path.dirname(__file__), "fixtures", "system_samples")


def sample(name):
    with open(os.path.join(SAMPLES, name), "r", encoding="utf-8") as fh:
        return fh.read()


# ------------------------------------------------------------------ util ---

def test_small_converters():
    assert to_int("49140") == 49140
    assert to_int("49140 MiB") == 49140
    assert to_int("0x10de") == 0x10DE
    assert to_int("[N/A]") is None
    assert to_int(True) is None
    assert to_int(float("nan")) is None
    assert parse_size_text("4 GB") == 4 * GiB
    assert parse_size_text("1536 MB") == 1536 * MiB
    assert parse_size_text("[Not Supported]") is None
    assert norm_pci_bus_id("00000000:01:00.0") == "0000:01:00.0"
    assert norm_pci_bus_id("0000000F:01:00.0") == "000f:01:00.0"
    assert norm_pci_bus_id("2b:00.0") == "0000:2b:00.0"
    assert short_name("NVIDIA RTX 6000 Ada Generation") == "NVIDIA RTX 6000 Ada"
    assert short_name("13th Gen Intel(R) Core(TM) i7-13620H") == "Intel Core i7-13620H"
    assert short_name("Intel(R) Core(TM) i5-3470 CPU @ 3.20GHz") == "Intel Core i5-3470"
    assert short_name("AMD Ryzen 9 7950X 16-Core Processor") == "AMD Ryzen 9 7950X"


# ------------------------------------------------------------- nvidia-smi ---

def test_nvidia_csv_multi_gpu():
    rows = gpu_nvidia.parse_query_csv(sample("nvidia_smi_query_2x_rtx6000ada.csv"))
    gpus = gpu_nvidia.gpus_from_rows(rows, arch="x86_64")
    assert [g["name"] for g in gpus] == ["NVIDIA RTX 6000 Ada Generation"] * 2
    g0 = gpus[0]
    assert g0["vram_total_bytes"] == 49140 * MiB == 51527024640
    assert g0["vram_free_bytes"] == 45801 * MiB
    assert g0["driver_version"] == "560.35.05"
    assert g0["kind"] == "discrete" and g0["vendor"] == "nvidia" and g0["source"] == "nvidia-smi"
    assert g0["backend_hints"] == ["cuda", "vulkan"]
    d = g0["details"]
    assert d["pci_bus_id"] == "0000:01:00.0" and d["pci_vendor_id"] == "0x10de" and d["pci_device_id"] == "0x26b1"
    assert d["compute_capability"] == "8.9" and d["nvidia_index"] == 0 and d["vram_used_bytes"] == 2736 * MiB
    assert gpus[1]["details"]["pci_bus_id"] == "0000:61:00.0"


def test_nvidia_csv_na_fields_and_comma_in_name():
    rows = gpu_nvidia.parse_query_csv(sample("nvidia_smi_query_mixed_na.csv"))
    assert len(rows) == 3
    gpus = gpu_nvidia.gpus_from_rows(rows, arch="x86_64")
    assert gpus[1]["name"] == "NVIDIA GeForce GTX 1080"
    assert gpus[1]["vram_total_bytes"] == 8192 * MiB
    assert gpus[1]["vram_free_bytes"] is None and gpus[1]["details"]["vram_used_bytes"] is None
    k40 = gpus[2]
    assert k40["name"] == "NVIDIA Tesla K40m, Rev 2"  # comma rejoined
    assert k40["vram_total_bytes"] is None and k40["details"]["uuid"] is None
    assert k40["details"]["pci_device_id"] is None and k40["details"]["compute_capability"] == "3.5"
    assert k40["kind"] == "discrete"  # unknown memory on x86 stays discrete


def test_nvidia_unified_gb10():
    rows = gpu_nvidia.parse_query_csv(sample("nvidia_smi_query_gb10_spark.csv"))
    (g,) = gpu_nvidia.gpus_from_rows(rows, arch="arm64", system_ram=128 * GiB)
    assert g["kind"] == "unified"
    assert g["vram_total_bytes"] == 128 * GiB and g["details"]["memory_from"] == "system_ram"
    assert g["details"]["pci_bus_id"] == "000f:01:00.0"


def test_nvidia_legacy_fields_parse():
    text = "0, GPU-x, 00000000:01:00.0, 0x1B8010DE, 8192, 8000, 192, 470.256.02, NVIDIA GeForce GTX 1080\n"
    rows = gpu_nvidia.parse_query_csv(text, gpu_nvidia.FIELDS_LEGACY)
    assert rows[0]["name"] == "NVIDIA GeForce GTX 1080" and "compute_cap" not in rows[0]


@pytest.mark.parametrize("key", ["driver_not_loaded", "driver_mismatch", "no_devices", "nvml_unknown", "legacy_field"])
def test_nvidia_error_classification(key):
    case = json.loads(sample("nvidia_smi_errors.json"))[key]
    state, message = gpu_nvidia.classify_error(case["returncode"], case["stdout"] + case["stderr"])
    assert state == case["state"]
    assert message and len(message) < 220


def test_nvidia_cuda_version_both_formats():
    assert gpu_nvidia.parse_cuda_version(sample("nvidia_smi_q_compute_rtx6000ada.txt")) == "12.6"
    assert gpu_nvidia.parse_cuda_version(sample("nvidia_smi_header_rtx6000ada.txt")) == "12.6"
    assert gpu_nvidia.parse_driver_version_header(sample("nvidia_smi_header_rtx6000ada.txt")) == "560.35.05"
    assert gpu_nvidia.parse_cuda_version("no version here") is None


def test_nvidia_windows_candidates_cover_system32_and_sysnative():
    env = {"SystemRoot": r"C:\Windows", "ProgramFiles": r"C:\Program Files"}
    c = gpu_nvidia.candidates("Windows", env)
    assert any(p.endswith(os.path.join("System32", "nvidia-smi.exe")) for p in c)
    assert any("Sysnative" in p for p in c)
    assert "/usr/lib/wsl/lib/nvidia-smi" in gpu_nvidia.candidates("Linux", {})


class FakeRun:
    """Stands in for _proc.run: returns canned CmdResults by argv pattern."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, argv, timeout=5.0):
        self.calls.append(list(argv))
        joined = " ".join(argv)
        for pattern, res in self.responses:
            if pattern in joined:
                return res() if callable(res) else res
        return CmdResult(argv=list(argv), found=False)


def _ctx(run, system="Linux"):
    from vvl.system.runner import ProbeContext

    return ProbeContext(system=system, env={}, run=run)


def test_nvidia_probe_flow_success(monkeypatch):
    monkeypatch.setattr(gpu_nvidia._proc, "find_tool", lambda name, cands=(): "/fake/nvidia-smi")
    run = FakeRun([
        ("--query-gpu", CmdResult(argv=[], returncode=0, stdout=sample("nvidia_smi_query_2x_rtx6000ada.csv"))),
        ("-q -d COMPUTE", CmdResult(argv=[], returncode=0, stdout=sample("nvidia_smi_q_compute_rtx6000ada.txt"))),
    ])
    from vvl.system.runner import ProbeLog

    out = gpu_nvidia.probe(_ctx(run), ProbeLog())
    assert out["state"] == "ok" and len(out["gpus"]) == 2
    assert out["cuda_version"] == "12.6" and out["driver_version"] == "560.35.05"
    assert all(isinstance(c, list) for c in run.calls)  # explicit argument lists


def test_nvidia_probe_retries_without_compute_cap(monkeypatch):
    monkeypatch.setattr(gpu_nvidia._proc, "find_tool", lambda name, cands=(): "/fake/nvidia-smi")
    legacy_err = json.loads(sample("nvidia_smi_errors.json"))["legacy_field"]
    legacy_ok = "0, GPU-x, 00000000:01:00.0, 0x1B8010DE, 8192, 8000, 192, 470.256.02, NVIDIA GeForce GTX 1080\n"

    def query(argv, timeout=5.0):
        if "compute_cap" in argv[1]:
            return CmdResult(argv=argv, returncode=legacy_err["returncode"], stdout=legacy_err["stdout"])
        if argv[1].startswith("--query-gpu"):
            return CmdResult(argv=argv, returncode=0, stdout=legacy_ok)
        return CmdResult(argv=argv, returncode=0, stdout="CUDA Version : 11.4\n")

    from vvl.system.runner import ProbeLog

    out = gpu_nvidia.probe(_ctx(query), ProbeLog())
    assert out["state"] == "ok"
    assert out["gpus"][0]["name"] == "NVIDIA GeForce GTX 1080" and out["cuda_version"] == "11.4"


def test_nvidia_probe_driver_not_loaded(monkeypatch):
    monkeypatch.setattr(gpu_nvidia._proc, "find_tool", lambda name, cands=(): "/fake/nvidia-smi")
    case = json.loads(sample("nvidia_smi_errors.json"))["driver_not_loaded"]
    run = FakeRun([("--query-gpu", CmdResult(argv=[], returncode=case["returncode"], stdout=case["stdout"]))])
    from vvl.system.runner import ProbeLog

    out = gpu_nvidia.probe(_ctx(run), ProbeLog())
    assert out["state"] == "driver_not_loaded" and out["gpus"] == []
    assert "driver" in out["message"]


def test_nvidia_probe_not_found(monkeypatch):
    monkeypatch.setattr(gpu_nvidia._proc, "find_tool", lambda name, cands=(): None)
    from vvl.system.runner import ProbeLog

    out = gpu_nvidia.probe(_ctx(FakeRun([])), ProbeLog())
    assert out["state"] == "not_found" and out["gpus"] == []


def test_nvidia_probe_timeout_is_soft_failure(monkeypatch):
    monkeypatch.setattr(gpu_nvidia._proc, "find_tool", lambda name, cands=(): "/fake/nvidia-smi")
    run = FakeRun([("--query-gpu", CmdResult(argv=[], timed_out=True))])
    from vvl.system.runner import ProbeLog

    log = ProbeLog()
    out = gpu_nvidia.probe(_ctx(run), log)
    assert out["state"] == "timeout" and log.errors


# ------------------------------------------------------- system_profiler ---

def test_system_profiler_m2_air():
    (g,) = gpu_apple.parse_displays_json(sample("system_profiler_m2_air.json"), apple_silicon=True, memsize=16 * GiB)
    assert g["vendor"] == "apple" and g["kind"] == "unified" and g["name"] == "Apple M2"
    assert g["vram_total_bytes"] == 16 * GiB and g["vram_free_bytes"] is None
    assert g["details"]["gpu_cores"] == 10 and g["details"]["metal_support"] == "Metal 3"
    assert g["backend_hints"] == ["metal"]


def test_system_profiler_m3_max_with_wired_limit():
    (g,) = gpu_apple.parse_displays_json(sample("system_profiler_m3_max.json"), apple_silicon=True,
                                         memsize=64 * GiB, wired_limit_mb=56 * 1024)
    assert g["name"] == "Apple M3 Max" and g["details"]["gpu_cores"] == 40
    assert g["details"]["wired_limit_bytes"] == 56 * GiB


def test_system_profiler_intel_mac():
    gpus = gpu_apple.parse_displays_json(sample("system_profiler_intel_mbp16_2019.json"), apple_silicon=False,
                                         memsize=32 * GiB)
    intel, amd = gpus
    assert intel["vendor"] == "intel" and intel["name"] == "Intel UHD Graphics 630"
    assert intel["kind"] == "integrated" and intel["vram_total_bytes"] == 1536 * MiB
    assert amd["vendor"] == "amd" and amd["name"] == "AMD Radeon Pro 5500M"
    assert amd["kind"] == "discrete" and amd["vram_total_bytes"] == 8 * GiB
    assert amd["backend_hints"] == [] and intel["backend_hints"] == []  # no llama.cpp Metal on Intel Macs


def test_system_profiler_bad_json_raises_valueerror():
    with pytest.raises(ValueError):
        gpu_apple.parse_displays_json("not json", apple_silicon=True)
    assert gpu_apple.parse_displays_json("{}", apple_silicon=True) == []


def test_mac_features_and_names():
    f = cpu.mac_features({"hw.optional.arm64": 1, "hw.optional.arm.FEAT_DotProd": 1, "hw.optional.arm.FEAT_I8MM": 0})
    assert f["neon"] is True and f["dotprod"] is True and f["i8mm"] is False and f["avx2"] is False
    intel = cpu.mac_features({"machdep.cpu.features": "FPU SSE3 FMA AVX1.0 F16C", "machdep.cpu.leaf7_features": "BMI1 AVX2"})
    assert intel["avx"] and intel["avx2"] and intel["fma"] and intel["f16c"] and not intel["avx512"] and not intel["neon"]
    assert cpu.mac_features({})["avx2"] is None
    assert os_info.mac_name("14.5") == "macOS 14.5 Sonoma"
    assert os_info.mac_name("10.15.7") == "macOS 10.15.7 Catalina"
    assert os_info.mac_name("26.1") == "macOS 26.1 Tahoe"
    assert os_info.mac_name("27.0") == "macOS 27.0"


# --------------------------------------------------------------- Windows ---

def _nv_values(**extra):
    v = {"DriverDesc": "NVIDIA GeForce RTX 4070", "ProviderName": "NVIDIA", "DriverVersion": "32.0.15.6094",
         "MatchingDeviceId": "pci\\ven_10de&dev_2786", "HardwareInformation.qwMemorySize": 12 * GiB,
         "_location": "PCI bus 1, device 0, function 0", "_subkey": "0001"}
    v.update(extra)
    return v


def test_windows_qword_memory_and_location():
    (g,) = gpu_windows.parse_adapters([_nv_values()], presence_checked=True)
    assert g["vendor"] == "nvidia" and g["vram_total_bytes"] == 12 * GiB and g["kind"] == "discrete"
    assert g["details"]["vram_source"] == "qwMemorySize" and g["details"]["vram_capped"] is False
    assert g["details"]["pci_bus_id"] == "0000:01:00.0" and g["details"]["pci_device_id"] == "0x2786"
    assert g["source"] == "registry" and g["driver_version"] == "32.0.15.6094"


def test_windows_32bit_memory_cap_is_flagged():
    amd = {"DriverDesc": "AMD Radeon RX 6800 XT", "ProviderName": "Advanced Micro Devices, Inc.",
           "MatchingDeviceId": "PCI\\VEN_1002&DEV_73BF&SUBSYS_0E3A1002",
           "HardwareInformation.MemorySize": b"\x00\x00\xf0\xff"}  # REG_BINARY 0xFFF00000
    (g,) = gpu_windows.parse_adapters([amd])
    assert g["vram_total_bytes"] == 0xFFF00000 and g["details"]["vram_capped"] is True
    assert g["details"]["vram_source"] == "MemorySize" and g["kind"] == "discrete"
    small = dict(amd, **{"HardwareInformation.MemorySize": 2 * GiB})  # REG_DWORD 2 GB
    (g2,) = gpu_windows.parse_adapters([small])
    assert g2["vram_total_bytes"] == 2 * GiB and g2["details"]["vram_capped"] is False


def test_windows_qword_as_bytes_and_adapter_string():
    v = {"HardwareInformation.AdapterString": "AMD Radeon RX 7600".encode("utf-16-le") + b"\x00\x00",
         "MatchingDeviceId": "PCI\\VEN_1002&DEV_7480",
         "HardwareInformation.qwMemorySize": (8 * GiB).to_bytes(8, "little")}
    (g,) = gpu_windows.parse_adapters([v])
    assert g["name"] == "AMD Radeon RX 7600" and g["vram_total_bytes"] == 8 * GiB


def test_windows_wmi_4gb_cap():
    item = {"Name": "NVIDIA GeForce RTX 3080", "AdapterRAM": 4293918720, "DriverVersion": "31.0.15.3623",
            "PNPDeviceID": "PCI\\VEN_10DE&DEV_2206&SUBSYS_38971462&REV_A1\\4&1F2B3C4D&0&0008",
            "AdapterCompatibility": "NVIDIA"}
    gpus = gpu_windows.parse_wmi_json(json.dumps(item))  # one adapter: ConvertTo-Json emits an object
    assert len(gpus) == 1 and gpus[0]["details"]["vram_capped"] is True and gpus[0]["source"] == "wmi"
    two = gpu_windows.parse_wmi_json(json.dumps([item, dict(item, Name="Intel(R) UHD Graphics 770", AdapterRAM=1073741824,
                                                            PNPDeviceID="PCI\\VEN_8086&DEV_4680")]))
    assert [g["vendor"] for g in two] == ["nvidia", "intel"] and two[1]["kind"] == "integrated"


def test_windows_skips_virtual_and_unknown_vendors():
    adapters = [
        {"DriverDesc": "Microsoft Basic Display Adapter", "ProviderName": "Microsoft", "MatchingDeviceId": "root\\basicdisplay"},
        {"DriverDesc": "Parsec Virtual Display Adapter", "MatchingDeviceId": "root\\parsecvda"},
        {"DriverDesc": "VMware SVGA 3D", "MatchingDeviceId": "PCI\\VEN_15AD&DEV_0405"},
        {"DriverDesc": "Microsoft Remote Display Adapter", "MatchingDeviceId": "swd\\remotedisplayenum"},
        _nv_values(),
    ]
    gpus = gpu_windows.parse_adapters(adapters)
    assert [g["name"] for g in gpus] == ["NVIDIA GeForce RTX 4070"]


def test_windows_gpu_on_basic_display_driver_is_kept_as_driver_missing():
    basic = {"DriverDesc": "Microsoft Basic Display Adapter", "ProviderName": "Microsoft",
             "MatchingDeviceId": "PCI\\CC_0300", "HardwareInformation.qwMemorySize": 0,
             "_instance_id": "PCI\\VEN_10DE&DEV_2684&SUBSYS_16F310DE&REV_A1\\4&2d78ab8f&0&0008"}
    (g,) = gpu_windows.parse_adapters([basic], presence_checked=True)
    assert g["vendor"] == "nvidia" and g["details"]["driver_missing"] is True
    assert g["name"] == "NVIDIA GPU (PCI 10de:2684, driver not installed)" and g["vram_total_bytes"] is None
    no_vendor = dict(basic, _instance_id="ROOT\\BasicDisplay\\0000")
    assert gpu_windows.parse_adapters([no_vendor]) == []


def test_windows_localized_device_desc_and_qualcomm():
    v = {"DeviceDesc": "@oem12.inf,%nvidia_dev.2684%;NVIDIA GeForce RTX 4090",
         "_instance_id": "PCI\\VEN_10DE&DEV_2684&SUBSYS_16F310DE&REV_A1\\4&2d78ab8f&0&0008",
         "_location": "@System32\\drivers\\pci.sys,#65536;PCI bus %1, device %2, function %3;(65,0,0)"}
    (g,) = gpu_windows.parse_adapters([v])
    assert g["name"] == "NVIDIA GeForce RTX 4090" and g["details"]["pci_bus_id"] == "0000:41:00.0"
    q = {"DriverDesc": "Qualcomm(R) Adreno(TM) X1-85 GPU", "ProviderName": "Qualcomm Technologies Inc",
         "MatchingDeviceId": "ACPI\\VEN_QCOM&DEV_0C36", "HardwareInformation.qwMemorySize": 0}
    (qg,) = gpu_windows.parse_adapters([q])
    assert qg["vendor"] == "unknown" and qg["kind"] == "integrated" and qg["backend_hints"] == []


def test_windows_location_forms():
    assert gpu_windows.parse_location("PCI bus 3, device 0, function 0") == "0000:03:00.0"
    assert gpu_windows.parse_location("@System32\\drivers\\pci.sys,#65536;PCI bus %1, device %2, function %3;(0,2,0)") \
        == "0000:00:02.0"
    assert gpu_windows.parse_location(None) is None


def test_windows_name_version():
    v11 = {"ProductName": "Windows 10 Pro", "CurrentBuildNumber": "22631", "DisplayVersion": "23H2", "UBR": 4460,
           "CurrentMajorVersionNumber": 10, "CurrentMinorVersionNumber": 0}
    assert os_info.windows_name_version(v11) == ("Windows 11 Pro 23H2", "10.0.22631.4460")
    v10 = {"ProductName": "Windows 10 Home", "CurrentBuildNumber": "19045", "DisplayVersion": "22H2", "UBR": 5131}
    assert os_info.windows_name_version(v10) == ("Windows 10 Home 22H2", "10.0.19045.5131")
    srv = {"ProductName": "Windows Server 2022 Standard", "CurrentBuildNumber": "20348", "DisplayVersion": "21H2"}
    assert os_info.windows_name_version(srv)[0] == "Windows Server 2022 Standard 21H2"
    name, ver = os_info.windows_name_version(None, "11", "10.0.26100")
    assert name == "Windows 11" and ver == "10.0.26100"


def test_windows_cpu_features():
    modern = cpu.windows_features({"ssse3": True, "sse4_1": True, "sse4_2": True, "avx": True, "avx2": True,
                                   "avx512f": False}, "x86_64")
    assert modern["avx2"] is True and modern["avx512"] is False and modern["neon"] is False
    assert modern["fma"] is None and modern["f16c"] is None  # Windows cannot be asked
    old_build = cpu.windows_features({k: False for k in ("ssse3", "sse4_1", "sse4_2", "avx", "avx2", "avx512f")}, "x86_64")
    assert all(v is None for v in old_build.values())
    arm = cpu.windows_features({"arm_neon": True, "arm_v82_dp": True}, "arm64")
    assert arm["neon"] is True and arm["dotprod"] is True and arm["avx2"] is False
    assert all(v is None for v in cpu.windows_features(None, "x86_64").values())


# ----------------------------------------------------------------- Linux ---

def test_proc_cpuinfo_intel():
    info = cpu.parse_proc_cpuinfo(sample("proc_cpuinfo_x86_intel_i7_12700.txt"))
    assert info["model"] == "12th Gen Intel(R) Core(TM) i7-12700" and info["vendor"] == "GenuineIntel"
    assert info["logical"] == 4 and info["physical"] == 2 and not info["is_arm"]
    f = cpu.features_from_x86_flags(info["flags"])
    assert f["avx"] and f["avx2"] and f["fma"] and f["f16c"] and f["avx_vnni"]
    assert not f["avx512"] and not f["neon"]


def test_proc_cpuinfo_no_avx2():
    info = cpu.parse_proc_cpuinfo(sample("proc_cpuinfo_x86_ivybridge_no_avx2.txt"))
    f = cpu.features_from_x86_flags(info["flags"])
    assert f["avx"] is True and f["avx2"] is False and f["fma"] is False and f["f16c"] is True
    assert info["physical"] == 4


def test_proc_cpuinfo_raspberry_pi():
    info = cpu.parse_proc_cpuinfo(sample("proc_cpuinfo_arm64_raspberrypi4.txt"))
    assert info["is_arm"] and info["model"] == "Raspberry Pi 4 Model B Rev 1.4" and info["vendor"] == "ARM"
    f = cpu.features_from_arm_flags(info["flags"])
    assert f["neon"] is True and f["dotprod"] is False and f["avx2"] is False
    assert info["logical"] == 4


def test_proc_cpuinfo_arm_big_little_without_model_name():
    info = cpu.parse_proc_cpuinfo(sample("proc_cpuinfo_arm64_gb10_bigLITTLE.txt"))
    assert info["model"] == "Cortex-X925 + Cortex-A725"
    f = cpu.features_from_arm_flags(info["flags"])
    assert f["neon"] and f["sve"] and f["dotprod"] and f["i8mm"]


def test_proc_cpuinfo_armv7_legacy():
    info = cpu.parse_proc_cpuinfo(sample("proc_cpuinfo_armv7_legacy.txt"))
    assert info["is_arm"] and info["model"] == "Cortex-A53" and info["logical"] == 2
    assert cpu.features_from_arm_flags(info["flags"])["neon"] is True


def test_linux_max_mhz(tmp_path):
    for i, khz in enumerate((4500000, 3600000)):
        d = tmp_path / f"sys/devices/system/cpu/cpu{i}/cpufreq"
        d.mkdir(parents=True)
        (d / "cpuinfo_max_freq").write_text(f"{khz}\n")
    assert cpu.linux_max_mhz(str(tmp_path)) == 4500
    (tmp_path / "sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq").write_text("7006640\n")
    assert cpu.linux_max_mhz(str(tmp_path)) is None  # implausible firmware value


OS_RELEASES = {
    "ubuntu": ('PRETTY_NAME="Ubuntu 22.04.5 LTS"\nNAME="Ubuntu"\nVERSION_ID="22.04"\nVERSION="22.04.5 LTS (Jammy Jellyfish)"\n'
               "ID=ubuntu\n", ("Ubuntu 22.04.5 LTS", "22.04")),
    "fedora": ('NAME="Fedora Linux"\nVERSION="40 (Workstation Edition)"\nID=fedora\nVERSION_ID=40\n'
               "PRETTY_NAME='Fedora Linux 40 (Workstation Edition)'\n", ("Fedora Linux 40 (Workstation Edition)", "40")),
    "arch": ('NAME="Arch Linux"\nPRETTY_NAME="Arch Linux"\nID=arch\nBUILD_ID=rolling\n', ("Arch Linux", "rolling")),
    "escaped": ('# comment\n\nPRETTY_NAME="Some \\"Quoted\\" OS"\nVERSION_ID=1\nbad line\n', ('Some "Quoted" OS', "1")),
    "no_pretty": ('NAME=Debian\nVERSION="12 (bookworm)"\nVERSION_ID="12"\n', ("Debian 12 (bookworm)", "12")),
    "empty": ("", ("Linux", None)),
}


@pytest.mark.parametrize("key", sorted(OS_RELEASES))
def test_os_release(key):
    text, expected = OS_RELEASES[key]
    assert os_info.linux_name_version(os_info.parse_os_release(text)) == expected


@pytest.mark.parametrize("kernel,proc_version,env,expected", [
    ("5.15.153.1-microsoft-standard-WSL2", "", {}, (True, 2)),
    ("6.6.36.6-microsoft-standard-WSL2+", "", {}, (True, 2)),
    ("4.4.0-19041-Microsoft", "Linux version 4.4.0-19041-Microsoft (Microsoft@Microsoft.com)", {}, (True, 1)),
    ("6.8.0-138-generic", "Linux version 6.8.0-138-generic (buildd@lcy02-amd64-008)", {}, (False, None)),
    ("6.1.21-custom", "", {"WSL_DISTRO_NAME": "Ubuntu", "WSL_INTEROP": "/run/WSL/1_interop"}, (True, 2)),
    ("6.1.21-custom", "", {"WSL_DISTRO_NAME": "Ubuntu"}, (True, None)),
])
def test_wsl_detection(kernel, proc_version, env, expected):
    assert os_info.wsl_info(kernel, proc_version, env) == expected


def test_arch_normalization():
    assert os_info.normalize_arch("AMD64") == "x86_64"
    assert os_info.normalize_arch("aarch64") == "arm64"
    assert os_info.normalize_arch("ARM64") == "arm64"
    assert os_info.normalize_arch("i686") == "x86"
    assert os_info.normalize_arch("armv7l") == "armv7l"
    assert os_info.normalize_arch("") is None
    assert os_info.python_arch_from_platform_tag("win-amd64", "ARM64") == "x86_64"  # emulated x64 Python
    assert os_info.python_arch_from_platform_tag("win-arm64", "ARM64") == "arm64"
    assert os_info.python_arch_from_platform_tag("macosx-11.0-universal2", "x86_64") == "x86_64"  # Rosetta


def _make_pci(root, addr, cls, vendor, device, driver=None, **files):
    d = root / "sys/bus/pci/devices" / addr
    d.mkdir(parents=True)
    (d / "class").write_text(f"0x{cls:06x}\n")
    (d / "vendor").write_text(f"0x{vendor:04x}\n")
    (d / "device").write_text(f"0x{device:04x}\n")
    for k, v in files.items():
        (d / k).write_text(f"{v}\n")
    if driver:
        drv = root / "sys/bus/pci/drivers" / driver
        drv.mkdir(parents=True, exist_ok=True)
        os.symlink(drv, d / "driver")


def test_linux_sysfs_scan(tmp_path):
    _make_pci(tmp_path, "0000:00:02.0", 0x030000, 0x8086, 0xA780, "i915", boot_vga=1)
    _make_pci(tmp_path, "0000:03:00.0", 0x030000, 0x1002, 0x744C, "amdgpu", mem_info_vram_total=25753026560,
              mem_info_vram_used=1073741824, mem_info_gtt_total=33285996544)
    _make_pci(tmp_path, "0000:03:00.1", 0x040300, 0x1002, 0xAB30, "snd_hda_intel")  # HDMI audio: not a GPU
    _make_pci(tmp_path, "0000:29:00.0", 0x030000, 0x1A03, 0x2000, "ast")  # BMC display: ignored
    _make_pci(tmp_path, "0000:65:00.0", 0x030200, 0x10DE, 0x2204, "nouveau")  # NVIDIA without its driver
    raw = gpu_linux.scan_pci_display(str(tmp_path))
    assert [r["addr"] for r in raw] == ["0000:00:02.0", "0000:03:00.0", "0000:29:00.0", "0000:65:00.0"]
    names = {"0000:03:00.0": {"device": "Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M]"},
             "0000:00:02.0": {"device": "Raptor Lake-S GT1 [UHD Graphics 770]"}}
    gpus = gpu_linux.gpus_from_scan(raw, names)
    by_bus = {g["details"]["pci_bus_id"]: g for g in gpus}
    assert set(by_bus) == {"0000:00:02.0", "0000:03:00.0", "0000:65:00.0"}
    intel, amd, nv = by_bus["0000:00:02.0"], by_bus["0000:03:00.0"], by_bus["0000:65:00.0"]
    assert intel["kind"] == "integrated" and intel["name"] == "Intel UHD Graphics 770"
    assert amd["kind"] == "discrete" and amd["vram_total_bytes"] == 25753026560
    assert amd["vram_free_bytes"] == 25753026560 - 1073741824 and amd["details"]["kernel_driver"] == "amdgpu"
    assert nv["vendor"] == "nvidia" and nv["details"]["kernel_driver"] == "nouveau" and nv["vram_total_bytes"] is None
    assert nv["name"] == "NVIDIA GPU (PCI 10de:2204)"
    assert gpu_linux.scan_pci_display(str(tmp_path / "missing")) is None


def test_lspci_mm_parse_and_names():
    names = gpu_linux.parse_lspci_mm(sample("lspci_mm_nn_amd_rx7900xtx_raphael.txt"))
    assert names["0000:03:00.0"]["device"] == "Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M]"
    assert names["0000:03:00.0"]["vendor"] == "Advanced Micro Devices, Inc. [AMD/ATI]"
    assert names["0000:13:00.0"]["device"] == "Raphael"
    assert names["0000:00:01.1"]["device"] == "Device"
    assert gpu_linux.display_name("amd", names["0000:03:00.0"]["device"], 0x1002, 0x744C) == \
        "AMD Radeon RX 7900 XT/7900 XTX/7900M"
    assert gpu_linux.display_name("intel", "Meteor Lake-P [Intel Arc Graphics]", 0x8086, 0x7D55) == "Intel Arc Graphics"
    # Old pci.ids (this workstation): unknown device.
    live = '0000:01:00.0 "VGA compatible controller [0300]" "NVIDIA Corporation [10de]" "Device [26b1]" -ra1 ' \
           '"NVIDIA Corporation [10de]" "Device [16a1]"'
    assert gpu_linux.parse_lspci_mm(live)["0000:01:00.0"]["device"] == "Device"
    assert gpu_linux.display_name("nvidia", "Device", 0x10DE, 0x26B1) == "NVIDIA GPU (PCI 10de:26b1)"


@pytest.mark.parametrize("vendor,name,vram,bus,dev,expected", [
    ("amd", "AMD Radeon RX 7900 XTX", None, None, None, "discrete"),
    ("amd", "AMD Radeon(TM) RX Vega 10 Graphics", None, None, None, "integrated"),  # Ryzen 2500U iGPU
    ("amd", "AMD Radeon RX Vega 64", None, None, None, "discrete"),
    ("amd", "AMD Radeon 780M Graphics", None, None, None, "integrated"),
    ("amd", "AMD Radeon RX 6500M", None, None, None, "discrete"),
    ("amd", "AMD Radeon(TM) 8060S Graphics", 96 * GiB, None, None, "integrated"),  # Strix Halo, big carve-out
    ("amd", "AMD Radeon(TM) Graphics", 512 * MiB, None, None, "integrated"),
    ("amd", "AMD Radeon Pro 5500M", None, None, None, "discrete"),
    ("amd", "AMD Radeon PRO W7900", None, None, None, "discrete"),
    ("amd", "Phoenix1", None, None, None, "integrated"),
    ("amd", "AMD GPU (PCI 1002:1234)", 16 * GiB, None, None, "discrete"),
    ("amd", "AMD GPU (PCI 1002:1234)", None, None, None, None),
    ("intel", "Intel(R) UHD Graphics", None, None, None, "integrated"),
    ("intel", "Intel(R) Arc(TM) A770 Graphics", None, None, None, "discrete"),
    ("intel", "Intel(R) Arc(TM) B580 Graphics", None, None, None, "discrete"),
    ("intel", "Intel(R) Arc(TM) Graphics", None, None, None, "integrated"),  # Meteor Lake iGPU
    ("intel", "Intel(R) Arc(TM) 140V GPU (16GB)", None, None, None, "integrated"),  # Lunar Lake iGPU
    ("intel", "DG2 [Arc A380]", None, None, 0x56A5, "discrete"),
    ("intel", "anything", None, "0000:00:02.0", 0x7D55, "integrated"),
    ("nvidia", "NVIDIA GeForce RTX 4090", None, None, None, "discrete"),
    ("nvidia", "NVIDIA GB10", None, None, None, "unified"),
])
def test_classify_kind(vendor, name, vram, bus, dev, expected):
    assert gpus_mod.classify_kind(vendor, name, vram, bus, dev) == expected


# ---------------------------------------------------------------- Vulkan ---

def test_vulkaninfo_amd_summary():
    parsed = vulkan.parse_summary(sample("vulkaninfo_summary_amd_rx7900xtx_raphael.txt"))
    assert parsed["instance_version"] == "1.3.283"
    types = [d["type"] for d in parsed["devices"]]
    assert types == ["discrete", "integrated", "cpu"]
    assert parsed["devices"][0]["vendor_id"] == 0x1002 and parsed["devices"][0]["device_id"] == 0x744C
    res = CmdResult(argv=["vulkaninfo"], returncode=0, stdout=sample("vulkaninfo_summary_amd_rx7900xtx_raphael.txt"))
    available, detail, devices = vulkan.assess(res)
    assert available is True and "RX 7900 XTX" in detail and "llvmpipe" not in detail and len(devices) == 3


def test_vulkaninfo_software_only_and_no_driver():
    soft = CmdResult(argv=["vulkaninfo"], returncode=0, stdout=sample("vulkaninfo_summary_llvmpipe_only.txt"))
    assert vulkan.assess(soft)[0] is False
    none = CmdResult(argv=["vulkaninfo"], returncode=1, stderr=sample("vulkaninfo_no_driver.txt"))
    available, detail, _ = vulkan.assess(none)
    assert available is False and "driver" in detail.lower()
    weird = CmdResult(argv=["vulkaninfo"], returncode=3, stderr="Segmentation fault")
    assert vulkan.assess(weird)[0] is None


def test_vulkaninfo_windows_summary():
    parsed = vulkan.parse_summary(sample("vulkaninfo_summary_windows_nvidia_intel.txt"))
    assert [(d["name"], d["type"]) for d in parsed["devices"]] == [
        ("NVIDIA GeForce RTX 4060 Laptop GPU", "discrete"), ("Intel(R) UHD Graphics", "integrated")]


def test_vulkan_probe_without_tool(monkeypatch, tmp_path):
    from vvl.system.runner import ProbeContext, ProbeLog

    monkeypatch.setattr(vulkan._proc, "find_tool", lambda name, cands=(): None)
    out = vulkan.probe(ProbeContext(system="Linux", env={}, sys_root=str(tmp_path)), ProbeLog())
    assert out["available"] is None and out["detail"] == "vulkaninfo not found"
    lib = tmp_path / "usr/lib/x86_64-linux-gnu"
    lib.mkdir(parents=True)
    (lib / "libvulkan.so.1").write_text("")
    out = vulkan.probe(ProbeContext(system="Linux", env={}, sys_root=str(tmp_path)), ProbeLog())
    assert out["available"] is None and "loader is installed" in out["detail"]


# ------------------------------------------------------------ GPU merge ---

def test_merge_vendor_tool_wins_and_dedupes():
    nv = gpu_nvidia.gpus_from_rows(gpu_nvidia.parse_query_csv(sample("nvidia_smi_query_rtx4060_laptop_windows.csv")))
    reg = gpu_windows.parse_adapters([
        _nv_values(DriverDesc="NVIDIA GeForce RTX 4060 Laptop GPU", MatchingDeviceId="pci\\ven_10de&dev_28e0"),
        {"DriverDesc": "Intel(R) UHD Graphics", "ProviderName": "Intel Corporation",
         "MatchingDeviceId": "pci\\ven_8086&dev_a7a8", "HardwareInformation.qwMemorySize": 128 * MiB,
         "_location": "PCI bus 0, device 2, function 0"},
    ])
    merged = gpus_mod.merge(nv, [], reg, nvidia_authoritative=True)
    assert [(g["index"], g["vendor"], g["source"]) for g in merged] == [(0, "nvidia", "nvidia-smi"), (1, "intel", "registry")]
    # Without nvidia-smi the registry entry stays (so the "driver not working" warning can name it).
    merged2 = gpus_mod.merge([], [], reg, nvidia_authoritative=False)
    assert [g["vendor"] for g in merged2] == ["nvidia", "intel"]


def test_merge_keeps_nvidia_card_that_nvidia_smi_missed():
    nv = gpu_nvidia.gpus_from_rows(gpu_nvidia.parse_query_csv(sample("nvidia_smi_query_2x_rtx6000ada.csv")))
    extra = gpus_mod.make_gpu("nvidia", "NVIDIA GPU (PCI 10de:2204)", kind="discrete", source="sysfs",
                              details={"pci_bus_id": "0000:a1:00.0"})
    dup = gpus_mod.make_gpu("nvidia", "dup", kind="discrete", source="sysfs", details={"pci_bus_id": "0000:01:00.0"})
    merged = gpus_mod.merge(nv, [], [dup, extra], nvidia_authoritative=True)
    assert len(merged) == 3 and merged[2]["details"]["note"] == "not listed by nvidia-smi"


def test_merge_vulkan_fills_kind_and_name():
    g = gpus_mod.make_gpu("amd", "AMD GPU (PCI 1002:744c)", kind=None, source="sysfs",
                          details={"pci_vendor_id": "0x1002", "pci_device_id": "0x744c"})
    devices = vulkan.parse_summary(sample("vulkaninfo_summary_amd_rx7900xtx_raphael.txt"))["devices"]
    (m,) = gpus_mod.merge([], [], [g], devices)
    assert m["kind"] == "discrete" and m["details"]["vulkan"] is True and m["name"] == "AMD Radeon RX 7900 XTX"
    # When the platform probe failed, Vulkan devices stand in (hardware only).
    only_vk = gpus_mod.merge([], [], [], devices, platform_ok=False)
    assert [d["source"] for d in only_vk] == ["vulkaninfo", "vulkaninfo"]
