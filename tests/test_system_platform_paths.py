"""Drive the Windows and macOS probe functions end to end on any OS by faking
the registry, sysctl, and the command runner. This checks the orchestration
(registry key paths, branching, fallbacks) that the parser tests do not."""

from __future__ import annotations

import json
import os

from vvl.system._proc import CmdResult
from vvl.system._util import GiB
from vvl.system.probes import _macos, _windows, cpu, gpu_apple, gpu_windows, os_info
from vvl.system.runner import ProbeContext, ProbeLog

SAMPLES = os.path.join(os.path.dirname(__file__), "fixtures", "system_samples")
CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
ENUM = r"SYSTEM\CurrentControlSet\Enum"


def sample(name):
    with open(os.path.join(SAMPLES, name), "r", encoding="utf-8") as fh:
        return fh.read()


class FakeRegistry:
    def __init__(self, keys, subkeys=None):
        self.keys = keys
        self.subkeys = subkeys or {}
        self.opened = []

    def values(self, root, path):
        self.opened.append(path)
        assert root == "HKEY_LOCAL_MACHINE"
        assert "\\\\" not in path, f"doubled backslash in registry path: {path}"
        v = self.keys.get(path)
        return dict(v) if v is not None else None

    def list(self, root, path):
        return self.subkeys.get(path)


NV_INST = r"PCI\VEN_10DE&DEV_28E0&SUBSYS_0B2A1028&REV_A1\4&2d78ab8f&0&0008"
IGPU_INST = r"PCI\VEN_8086&DEV_A7A8&SUBSYS_0B2A1028&REV_04\3&11583659&0&10"
REG = {
    ENUM + "\\" + NV_INST: {"Driver": "{4d36e968-e325-11ce-bfc1-08002be10318}\\0001",
                            "LocationInformation": "PCI bus 1, device 0, function 0",
                            "DeviceDesc": "@oem50.inf,%nvidia_dev.28e0%;NVIDIA GeForce RTX 4060 Laptop GPU"},
    ENUM + "\\" + IGPU_INST: {"Driver": "{4d36e968-e325-11ce-bfc1-08002be10318}\\0000",
                              "LocationInformation": "PCI bus 0, device 2, function 0"},
    CLASS + "\\0000": {"DriverDesc": "Intel(R) UHD Graphics", "ProviderName": "Intel Corporation",
                       "DriverVersion": "31.0.101.4502", "MatchingDeviceId": "pci\\ven_8086&dev_a7a8",
                       "HardwareInformation.qwMemorySize": 128 * 1024**2},
    CLASS + "\\0001": {"DriverDesc": "NVIDIA GeForce RTX 4060 Laptop GPU", "ProviderName": "NVIDIA",
                       "DriverVersion": "32.0.15.6094", "MatchingDeviceId": "pci\\ven_10de&dev_28e0",
                       "HardwareInformation.qwMemorySize": 8 * GiB},
    # A card that was removed: its driver key survives but it is not present.
    CLASS + "\\0002": {"DriverDesc": "NVIDIA GeForce GTX 1060 6GB", "ProviderName": "NVIDIA",
                       "MatchingDeviceId": "pci\\ven_10de&dev_1c03", "HardwareInformation.qwMemorySize": 6 * GiB},
    r"SOFTWARE\Microsoft\Windows NT\CurrentVersion": {"ProductName": "Windows 10 Home", "CurrentBuildNumber": "22631",
                                                      "DisplayVersion": "23H2", "UBR": 4460,
                                                      "CurrentMajorVersionNumber": 10, "CurrentMinorVersionNumber": 0},
    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0": {"ProcessorNameString": "13th Gen Intel(R) Core(TM) i7-13620H  ",
                                                        "VendorIdentifier": "GenuineIntel", "~MHz": 2400},
}


def _patch_registry(monkeypatch, reg, present):
    monkeypatch.setattr(_windows, "reg_values", reg.values)
    monkeypatch.setattr(_windows, "reg_subkeys", reg.list)
    monkeypatch.setattr(_windows, "present_display_devices", lambda: present)


def test_windows_gpu_probe_uses_present_devices_only(monkeypatch):
    reg = FakeRegistry(REG)
    _patch_registry(monkeypatch, reg, [NV_INST, IGPU_INST])
    out = gpu_windows.probe(ProbeContext(system="Windows", env={}), ProbeLog())
    names = sorted(g["name"] for g in out["gpus"])
    assert names == ["Intel(R) UHD Graphics", "NVIDIA GeForce RTX 4060 Laptop GPU"]  # no removed GTX 1060
    nv = next(g for g in out["gpus"] if g["vendor"] == "nvidia")
    assert nv["vram_total_bytes"] == 8 * GiB and nv["details"]["pci_bus_id"] == "0000:01:00.0"
    assert nv["details"]["presence_checked"] is True and nv["details"]["registry_key"] == "0001"
    assert CLASS + "\\0001" in reg.opened


def test_windows_gpu_probe_without_cfgmgr_reads_every_driver_key(monkeypatch):
    reg = FakeRegistry(REG, {CLASS: ["0000", "0001", "0002", "Configuration", "Properties"]})
    _patch_registry(monkeypatch, reg, None)
    out = gpu_windows.probe(ProbeContext(system="Windows", env={}), ProbeLog())
    assert len(out["gpus"]) == 3 and all(g["details"]["presence_checked"] is False for g in out["gpus"])


def test_windows_gpu_probe_falls_back_to_powershell(monkeypatch):
    reg = FakeRegistry({}, {})
    _patch_registry(monkeypatch, reg, None)
    wmi = json.dumps({"Name": "AMD Radeon RX 580", "AdapterRAM": 4293918720, "DriverVersion": "31.0.12027.9001",
                      "PNPDeviceID": "PCI\\VEN_1002&DEV_67DF", "AdapterCompatibility": "Advanced Micro Devices, Inc."})
    calls = []

    def run(argv, timeout=5.0):
        calls.append(argv)
        return CmdResult(argv=argv, returncode=0, stdout=wmi)

    out = gpu_windows.probe(ProbeContext(system="Windows", env={"SystemRoot": r"C:\Windows"}, run=run), ProbeLog())
    assert out["source"] == "wmi" and out["gpus"][0]["details"]["vram_capped"] is True
    assert calls[0][0].lower().endswith("powershell.exe") and "-NoProfile" in calls[0]


def test_windows_os_and_cpu_probes(monkeypatch):
    reg = FakeRegistry(REG)
    _patch_registry(monkeypatch, reg, [])
    monkeypatch.setattr(_windows, "native_machine", lambda: "x86_64")
    monkeypatch.setattr(_windows, "processor_features", lambda: {"ssse3": True, "sse4_1": True, "sse4_2": True,
                                                                 "avx": True, "avx2": True, "avx512f": False})
    ctx = ProbeContext(system="Windows", env={})
    o = os_info.probe(ctx, ProbeLog())
    assert o["name"] == "Windows 11 Home 23H2" and o["version"] == "10.0.22631.4460" and o["arch"] == "x86_64"
    c = cpu.probe(ctx, ProbeLog())
    assert c["model"] == "13th Gen Intel(R) Core(TM) i7-13620H" and c["vendor"] == "GenuineIntel"
    assert c["features"]["avx2"] is True and c["features_source"] == "windows_api"


def test_windows_on_arm_with_emulated_python(monkeypatch):
    reg = FakeRegistry(REG)
    _patch_registry(monkeypatch, reg, [])
    monkeypatch.setattr(_windows, "native_machine", lambda: "arm64")
    monkeypatch.setattr(os_info, "_machine", lambda: "AMD64")  # what an emulated x64 process sees
    monkeypatch.setattr(os_info.sysconfig, "get_platform", lambda: "win-amd64")
    o = os_info.probe(ProbeContext(system="Windows", env={}), ProbeLog())
    assert o["arch"] == "arm64" and o["python_arch"] == "x86_64" and o["is_translated"] is True


# ------------------------------------------------------------------ macOS ---

def _patch_sysctl(monkeypatch, values):
    """Fake the two sysctl readers; arch_info runs for real on top of them."""
    values = dict(values)
    values.setdefault("hw.ncpu", values.get("hw.logicalcpu", 8))
    monkeypatch.setattr(_macos, "sysctl_int",
                        lambda name, run=None: None if isinstance(values.get(name), str) else values.get(name))
    monkeypatch.setattr(_macos, "sysctl_str",
                        lambda name, run=None: values.get(name) if isinstance(values.get(name), str) else None)


M2 = {"hw.optional.arm64": 1, "sysctl.proc_translated": 0, "hw.memsize": 16 * GiB, "machdep.cpu.brand_string": "Apple M2",
      "hw.physicalcpu": 8, "hw.logicalcpu": 8, "hw.optional.arm.FEAT_DotProd": 1, "hw.optional.arm.FEAT_I8MM": 1}


def _mac_run(profiler_json=None, sw_vers="14.5"):
    def run(argv, timeout=5.0):
        if argv[0].endswith("sw_vers"):
            return CmdResult(argv=argv, returncode=0, stdout=sw_vers + "\n")
        if argv[0].endswith("system_profiler"):
            if profiler_json is None:
                return CmdResult(argv=argv, timed_out=True)
            return CmdResult(argv=argv, returncode=0, stdout=profiler_json)
        return CmdResult(argv=argv, found=False)
    return run


def test_macos_apple_silicon_probes(monkeypatch):
    _patch_sysctl(monkeypatch, M2)
    monkeypatch.setattr(os_info.platform, "mac_ver", lambda: ("", ("", "", ""), ""))
    ctx = ProbeContext(system="Darwin", env={}, run=_mac_run(sample("system_profiler_m2_air.json")))
    o = os_info.probe(ctx, ProbeLog())
    assert o["name"] == "macOS 14.5 Sonoma" and o["is_apple_silicon"] is True and o["arch"] == "arm64"
    c = cpu.probe(ctx, ProbeLog())
    assert c["model"] == "Apple M2" and c["vendor"] == "Apple" and c["max_mhz"] is None
    assert c["features"]["neon"] is True and c["features"]["avx2"] is False
    g = gpu_apple.probe(ctx, ProbeLog())
    assert [x["name"] for x in g["gpus"]] == ["Apple M2"] and g["gpus"][0]["vram_total_bytes"] == 16 * GiB


def test_macos_rosetta_is_detected(monkeypatch):
    _patch_sysctl(monkeypatch, dict(M2, **{"sysctl.proc_translated": 1}))
    monkeypatch.setattr(os_info, "_machine", lambda: "x86_64")
    monkeypatch.setattr(os_info.platform, "mac_ver", lambda: ("14.5", ("", "", ""), "x86_64"))
    o = os_info.probe(ProbeContext(system="Darwin", env={}, run=_mac_run()), ProbeLog())
    assert o["is_apple_silicon"] is True and o["is_translated"] is True and o["arch"] == "arm64"
    assert o["python_arch"] == "x86_64"


def test_macos_system_profiler_failure_still_reports_apple_gpu(monkeypatch):
    _patch_sysctl(monkeypatch, M2)
    log = ProbeLog()
    g = gpu_apple.probe(ProbeContext(system="Darwin", env={}, run=_mac_run(None)), log)
    assert log.errors and g["gpus"][0]["source"] == "sysctl" and g["gpus"][0]["kind"] == "unified"
    assert g["gpus"][0]["name"] == "Apple M2"


def test_macos_intel(monkeypatch):
    values = {"hw.optional.arm64": None, "hw.memsize": 32 * GiB, "hw.physicalcpu": 8, "hw.logicalcpu": 16,
              "machdep.cpu.brand_string": "Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz", "machdep.cpu.vendor": "GenuineIntel",
              "hw.cpufrequency_max": 2300000000, "machdep.cpu.features": "FPU SSE3 FMA AVX1.0 F16C",
              "machdep.cpu.leaf7_features": "BMI1 AVX2 BMI2"}
    _patch_sysctl(monkeypatch, values)
    ctx = ProbeContext(system="Darwin", env={}, run=_mac_run(sample("system_profiler_intel_mbp16_2019.json")))
    c = cpu.probe(ctx, ProbeLog())
    assert c["max_mhz"] == 2300 and c["features"]["avx2"] is True and c["vendor"] == "GenuineIntel"
    g = gpu_apple.probe(ctx, ProbeLog())
    assert [x["vendor"] for x in g["gpus"]] == ["intel", "amd"] and g["is_apple_silicon"] is False
