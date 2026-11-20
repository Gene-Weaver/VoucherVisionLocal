"""fit.assess and capability edge cases: the formula, odd inputs, and machines
that are awkward to keep as full fixtures (broken drivers, WSL without a GPU,
unified NVIDIA parts, capable iGPUs, low RAM, failed probes)."""

from __future__ import annotations

import copy
import json
import os

import pytest

from vvl.system import capability, fit, policy, schema
from vvl.system._util import GiB, MiB
from vvl.system.report import assemble
from vvl.system.runner import ProbeContext, ProbeOutcome

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures", "system_reports")


def load(name):
    with open(os.path.join(FIXTURES, f"{name}.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


BUILD = {"id": "x", "context_default": 8192,
         "files": [{"role": "model", "size_bytes": 8709519456}, {"role": "mmproj", "size_bytes": 752289728}]}


# --------------------------------------------------------------- formula ---

def test_required_bytes_formula():
    expected = 8709519456 + 752289728 + 8192 * policy.KV_BYTES_PER_TOKEN + policy.RUNTIME_MARGIN_BYTES
    assert fit.estimate_required_bytes(BUILD) == expected


def test_context_overhead_scales_with_context():
    small = fit.estimate_required_bytes(dict(BUILD, context_default=4096))
    big = fit.estimate_required_bytes(dict(BUILD, context_default=32768))
    assert big - small == (32768 - 4096) * policy.KV_BYTES_PER_TOKEN
    assert fit.estimate_required_bytes(dict(BUILD, context_default=None)) == fit.estimate_required_bytes(BUILD)


def test_total_size_fallback_and_missing_sizes():
    b = {"total_size_bytes": 9_000_000_000, "context_default": 8192}
    assert fit.estimate_required_bytes(b) > 9_000_000_000
    for bad in (None, {}, {"files": "x"}, {"files": [{"size_bytes": -1}]}, {"files": [{"size_bytes": True}]},
                {"files": [{"role": "model"}]}, {"files": [None]}):
        assert fit.estimate_required_bytes(bad) is None
        res = fit.assess(load("linux_workstation_2x_rtx6000ada"), bad)
        assert res["verdict"] == "unknown" and res["required_bytes"] is None and schema.validate_fit(res) == []


ODD_REPORTS = [
    None, {}, [], "report", 42,
    {"gpus": "x"},
    {"memory": {"total_bytes": "lots"}},
    {"memory": None, "gpus": [None, 3, {"vendor": "nvidia"}]},
    {"capability": "garbage", "os": [], "cpu": 7},
    {"gpus": [{"vendor": "nvidia", "source": "nvidia-smi", "vram_total_bytes": "8 GB", "index": "0",
               "details": "nope"}], "memory": {"total_bytes": 16 * GiB}},
    {"gpus": [{"vendor": "nvidia", "source": "nvidia-smi", "vram_total_bytes": 8 * GiB,
               "details": {"compute_capability": "eight"}}]},
    {"os": {"is_apple_silicon": True}, "gpus": [{"vendor": "apple", "kind": "unified", "details": None}]},
    {"accelerators": {"cuda": "yes", "vulkan": None}, "gpus": [{"vendor": "amd", "kind": "discrete"}]},
]


@pytest.mark.parametrize("report", ODD_REPORTS, ids=range(len(ODD_REPORTS)))
def test_assess_never_raises_on_odd_reports(report):
    res = fit.assess(report, BUILD)
    assert schema.validate_fit(res) == [], res
    cap, warns = capability.compute(report if isinstance(report, dict) else {})
    assert cap["recommended_backend"] in schema.BACKENDS + (None,)


def test_partial_report_gives_unknown_not_a_guess():
    res = fit.assess({"memory": {"total_bytes": None}}, BUILD)
    assert res["verdict"] == "unknown"
    res = fit.assess({"gpus": [{"vendor": "nvidia", "source": "nvidia-smi", "vram_total_bytes": None, "index": 0}],
                      "memory": {"total_bytes": 32 * GiB}}, BUILD)
    assert res["verdict"] == "unknown" and "memory size could not be read" in res["detail"]


# ------------------------------------------------------------- budgets ---

def test_budget_helpers():
    assert capability.usable_discrete(8188 * MiB) == 8188 * MiB - 384 * MiB
    assert capability.usable_discrete(48 * GiB) == 48 * GiB - int(0.04 * 48 * GiB)
    assert capability.usable_discrete(None) is None
    assert capability.unified_budget(16 * GiB) == int(16 * GiB * 2 / 3)
    assert capability.unified_budget(64 * GiB) == int(64 * GiB * 0.75)
    assert capability.unified_budget(16 * GiB, wired_limit=14 * GiB) == 14 * GiB
    assert capability.integrated_budget(512 * MiB, 32 * GiB) == 16 * GiB
    assert capability.integrated_budget(96 * GiB, 128 * GiB) == 96 * GiB
    assert capability.cpu_budget(8 * GiB) == 5 * GiB
    assert capability.cpu_budget(64 * GiB) == 48 * GiB


@pytest.mark.parametrize("budget,tier", [
    (0, "entry"), (6 * GiB, "entry"), (7 * GiB, "mid"), (int(13.9 * GiB), "mid"), (14 * GiB, "high"),
    (39 * GiB, "high"), (40 * GiB, "workstation"), (200 * GiB, "workstation"), (None, None),
])
def test_tier_thresholds(budget, tier):
    assert capability.tier_for("cuda", budget) == tier
    assert capability.tier_for("cpu", budget) == "cpu_only"


# ------------------------------------------------------- scenario helper ---

def machine(system="Linux", *, os_extra=None, ram=32 * GiB, nvidia=None, platform_gpus=(), vulkan=None,
            cpu_features=None, apple=None, disk=None, failed=()):
    """Build a report through the real assembler from probe values."""
    os_v = {"system": system, "name": "Test", "version": "1", "kernel": "k", "arch": "x86_64",
            "is_wsl": False, "is_apple_silicon": False, "python": "3.12.15", "is_translated": False}
    os_v.update(os_extra or {})
    feats = {"avx": True, "avx2": True, "avx512": False, "fma": True, "f16c": True, "neon": False}
    feats.update(cpu_features or {})
    values = {
        "os": os_v,
        "cpu": {"model": "Test CPU", "vendor": "GenuineIntel", "physical_cores": 8, "logical_cores": 16,
                "max_mhz": 4000, "features": feats},
        "memory": {"total_bytes": ram, "available_bytes": ram // 2, "swap_total_bytes": 0, "swap_free_bytes": 0},
        "vulkan": vulkan or {"available": None, "detail": "vulkaninfo not found", "devices": []},
        "disk": disk or {"models_dir": "/data/models", "exists": True, "checked_path": "/data/models",
                         "total_bytes": 2000 * GiB, "free_bytes": 1000 * GiB, "is_synced_folder": False,
                         "synced_folder_hint": None},
    }
    if system == "Darwin":
        values["gpu_apple"] = apple or {"gpus": [], "is_apple_silicon": False}
    else:
        values["gpu_nvidia"] = nvidia or {"state": "not_found", "message": "nvidia-smi not found", "gpus": []}
        values["gpu_linux" if system == "Linux" else "gpu_windows"] = {"gpus": list(platform_gpus)}
    outcomes = {k: ProbeOutcome(k, k not in failed, 5, "failed" if k in failed else None,
                                None if k in failed else v) for k, v in values.items()}
    ctx = ProbeContext(system=system, env={}, models_dir=values["disk"]["models_dir"])
    r = assemble(outcomes, ctx, generated_at="2026-11-19T18:04:11Z", duration_ms=10)
    assert schema.validate_report(r) == [], schema.validate_report(r)
    return r


def codes(r):
    return {w["code"] for w in r["warnings"]}


def gpu(vendor, name, kind, vram=None, source="sysfs", **details):
    from vvl.system.gpus import make_gpu

    return make_gpu(vendor, name, kind=kind, vram_total=vram, source=source, details=details)


# ---------------------------------------------------------- scenarios ---

def test_nvidia_present_but_driver_not_loaded():
    r = machine(nvidia={"state": "driver_not_loaded", "gpus": [],
                        "message": "nvidia-smi could not talk to the NVIDIA driver"},
                platform_gpus=[gpu("nvidia", "NVIDIA GPU (PCI 10de:2204)", "discrete", kernel_driver="nouveau",
                                   pci_bus_id="0000:65:00.0")])
    assert r["accelerators"]["cuda"]["available"] is False
    assert r["capability"]["recommended_backend"] == "cpu"
    assert "NVIDIA_DRIVER_NOT_WORKING" in codes(r)
    assert "driver" in r["capability"]["backend_reason"]
    assert fit.assess(r, BUILD)["verdict"] == "cpu_only"


def test_nvidia_gpu_without_nvidia_smi():
    r = machine("Windows", platform_gpus=[gpu("nvidia", "NVIDIA GeForce GTX 1660", "discrete", 6 * GiB,
                                               source="registry")])
    assert r["accelerators"]["cuda"] == {**r["accelerators"]["cuda"], "available": False, "detail": "nvidia-smi not found"}
    w = next(w for w in r["warnings"] if w["code"] == "NVIDIA_DRIVER_NOT_WORKING")
    assert "nvidia-smi is missing" in w["message"]


def test_gpu_without_vendor_driver_warns():
    from vvl.system.probes import gpu_windows

    basic = {"DriverDesc": "Microsoft Basic Display Adapter", "MatchingDeviceId": "PCI\\CC_0300",
             "_instance_id": "PCI\\VEN_1002&DEV_744C&SUBSYS_E4711DA2&REV_C8\\6&2f1c0e1&0&00000019"}
    r = machine("Windows", platform_gpus=gpu_windows.parse_adapters([basic], presence_checked=True))
    assert "GPU_DRIVER_MISSING" in codes(r) and r["capability"]["recommended_backend"] == "cpu"
    nv = dict(basic, _instance_id="PCI\\VEN_10DE&DEV_2684&SUBSYS_16F310DE&REV_A1\\4&2d78ab8f&0&0008")
    r2 = machine("Windows", platform_gpus=gpu_windows.parse_adapters([nv], presence_checked=True))
    assert "NVIDIA_DRIVER_NOT_WORKING" in codes(r2) and r2["capability"]["recommended_backend"] == "cpu"


def test_leftover_nvidia_smi_without_an_nvidia_gpu_is_not_a_broken_driver():
    r = machine(nvidia={"state": "driver_not_loaded", "gpus": [], "message": "nvidia-smi could not talk to the driver"},
                platform_gpus=[gpu("amd", "AMD Radeon RX 7900 XTX", "discrete", 24 * GiB)])
    assert r["accelerators"]["cuda"]["state"] == "no_gpu" and r["accelerators"]["cuda"]["available"] is False
    assert "NVIDIA_DRIVER_NOT_WORKING" not in codes(r) and r["capability"]["recommended_backend"] == "vulkan"


def test_no_nvidia_anywhere_is_a_plain_no():
    r = machine(platform_gpus=[])
    assert r["accelerators"]["cuda"]["available"] is False and r["accelerators"]["cuda"]["detail"] == "No NVIDIA GPU found."
    assert r["capability"]["recommended_backend"] == "cpu" and not codes(r) & {"NVIDIA_DRIVER_NOT_WORKING"}


def test_wsl_without_gpu_passthrough():
    r = machine(os_extra={"is_wsl": True, "wsl_version": 2, "wsl_gpu_passthrough": False})
    assert r["capability"]["recommended_backend"] == "cpu"
    assert "WSL_NO_GPU" in codes(r)
    r1 = machine(os_extra={"is_wsl": True, "wsl_version": 1, "wsl_gpu_passthrough": False})
    assert "WSL 1" in next(w["message"] for w in r1["warnings"] if w["code"] == "WSL_NO_GPU")


def test_old_nvidia_gpu_is_not_used_for_cuda():
    from vvl.system.probes import gpu_nvidia

    with open(os.path.join(HERE, "fixtures", "system_samples", "nvidia_smi_query_mixed_na.csv")) as fh:
        rows = gpu_nvidia.parse_query_csv(fh.read())
    nv = gpu_nvidia.gpus_from_rows(rows[2:])  # Tesla K40m only (compute 3.5)
    r = machine(nvidia={"state": "ok", "gpus": nv, "driver_version": "470.256.02", "cuda_version": "11.4"})
    assert r["capability"]["recommended_backend"] == "cpu" and "NVIDIA_GPU_TOO_OLD" in codes(r)
    mixed = gpu_nvidia.gpus_from_rows(rows)  # 3090 + 1080 + K40m
    r2 = machine(nvidia={"state": "ok", "gpus": mixed, "driver_version": "550.120", "cuda_version": "12.4"})
    assert r2["capability"]["recommended_backend"] == "cuda" and r2["capability"]["gpu_indices"] == [0, 1]
    assert fit.assess(r2, BUILD)["verdict"] == "fits_gpu"
    big = dict(BUILD, files=[{"size_bytes": 28 * GiB}])
    assert fit.assess(r2, big)["verdict"] == "fits_split" and fit.assess(r2, big)["label"] == "Split across GPUs"


def test_unified_nvidia_gb10():
    from vvl.system.probes import gpu_nvidia

    with open(os.path.join(HERE, "fixtures", "system_samples", "nvidia_smi_query_gb10_spark.csv")) as fh:
        nv = gpu_nvidia.gpus_from_rows(gpu_nvidia.parse_query_csv(fh.read()), arch="arm64", system_ram=128 * GiB)
    r = machine(os_extra={"arch": "arm64"}, ram=128 * GiB,
                cpu_features={"avx": False, "avx2": False, "fma": False, "f16c": False, "neon": True},
                nvidia={"state": "ok", "gpus": nv, "driver_version": "580.95.05", "cuda_version": "13.0"})
    cap = r["capability"]
    assert cap["recommended_backend"] == "cuda" and cap["memory_mode"] == "shared"
    assert cap["gpu_memory_budget_bytes"] == int(128 * GiB * 0.75) and cap["tier"] == "workstation"
    assert "CPU_NO_AVX2" not in codes(r)  # ARM: AVX does not apply
    assert fit.assess(r, dict(BUILD, files=[{"size_bytes": 60 * GiB}]))["verdict"] == "fits_gpu"


def test_capable_integrated_gpu_uses_vulkan_shared_memory():
    vk = {"available": True, "detail": "AMD Radeon 780M", "devices": [
        {"name": "AMD Radeon 780M (RADV GFX1103_R1)", "type": "integrated", "vendor_id": 0x1002, "device_id": 0x15BF}]}
    r = machine(ram=32 * GiB, vulkan=vk, platform_gpus=[
        gpu("amd", "AMD Radeon 780M Graphics", "integrated", 512 * MiB, pci_vendor_id="0x1002",
            pci_device_id="0x15bf", dedicated_memory_bytes=512 * MiB)])
    cap = r["capability"]
    assert cap["recommended_backend"] == "vulkan" and cap["memory_mode"] == "shared"
    assert cap["gpu_memory_budget_bytes"] == 16 * GiB and cap["tier"] == "high"
    assert fit.assess(r, BUILD)["verdict"] == "fits_gpu"
    assert fit.assess(r, dict(BUILD, files=[{"size_bytes": 20 * GiB}]))["verdict"] == "fits_split"


def test_weak_integrated_gpu_stays_on_cpu():
    r = machine(platform_gpus=[gpu("intel", "Intel(R) UHD Graphics 620", "integrated", 128 * MiB)])
    assert r["capability"]["recommended_backend"] == "cpu"
    assert "integrated" in r["capability"]["backend_reason"]


def test_amd_dgpu_with_broken_vulkan():
    r = machine(vulkan={"available": False, "detail": "No working Vulkan driver was found.", "devices": []},
                platform_gpus=[gpu("amd", "AMD Radeon RX 6700 XT", "discrete", 12 * GiB)])
    assert r["capability"]["recommended_backend"] == "cpu" and "VULKAN_NOT_AVAILABLE" in codes(r)


def test_amd_dgpu_unconfirmed_vulkan_still_recommended():
    r = machine("Windows", platform_gpus=[gpu("amd", "AMD Radeon RX 7800 XT", "discrete", 16 * GiB, source="registry")])
    cap = r["capability"]
    assert cap["recommended_backend"] == "vulkan" and "could not be confirmed" in cap["backend_reason"]
    assert cap["tier"] == "high"


def test_vram_unknown_and_capped():
    r = machine("Windows", platform_gpus=[gpu("intel", "Intel(R) Arc(TM) A770 Graphics", "discrete", None,
                                               source="registry")])
    assert r["capability"]["recommended_backend"] == "vulkan" and r["capability"]["tier"] is None
    assert "GPU_MEMORY_UNKNOWN" in codes(r) and fit.assess(r, BUILD)["verdict"] == "unknown"
    r2 = machine("Windows", platform_gpus=[gpu("amd", "AMD Radeon RX 6800 XT", "discrete", 0xFFF00000,
                                                source="registry", vram_capped=True)])
    assert "GPU_MEMORY_CAPPED" in codes(r2)


def test_low_ram_and_low_disk_and_fat32():
    r = machine(ram=4 * GiB, disk={"models_dir": "E:\\models", "exists": True, "checked_path": "E:\\models",
                                   "total_bytes": 32 * GiB, "free_bytes": 5 * GiB, "is_synced_folder": False,
                                   "synced_folder_hint": None, "filesystem": "FAT32", "writable": False})
    c = codes(r)
    assert {"LOW_RAM", "LOW_DISK_SPACE", "MODELS_DIR_FAT32", "MODELS_DIR_NOT_WRITABLE"} <= c
    assert fit.assess(r, BUILD)["verdict"] == "too_large"


def test_apple_wired_limit_raises_budget():
    from vvl.system.gpus import make_gpu

    g = make_gpu("apple", "Apple M2 Pro", kind="unified", vram_total=32 * GiB, source="system_profiler",
                 backend_hints=["metal"], details={"gpu_cores": 19, "wired_limit_bytes": 28 * GiB})
    r = machine("Darwin", ram=32 * GiB, os_extra={"is_apple_silicon": True, "arch": "arm64"},
                cpu_features={"avx": False, "avx2": False, "neon": True},
                apple={"gpus": [g], "is_apple_silicon": True})
    assert r["capability"]["gpu_memory_budget_bytes"] == 28 * GiB
    assert r["accelerators"]["metal"]["available"] is True and "19-core GPU" in r["accelerators"]["metal"]["detail"]


def test_failed_probes_are_reported_and_tolerated():
    r = machine(failed=("cpu", "gpu_nvidia"))
    assert "PROBES_INCOMPLETE" in codes(r)
    assert r["cpu"]["model"] is None and r["accelerators"]["cuda"]["available"] is None
    assert schema.validate_fit(fit.assess(r, BUILD)) == []


def test_free_memory_note_in_detail():
    r = load("linux_workstation_2x_rtx6000ada")
    busy = copy.deepcopy(r)
    for g in busy["gpus"]:
        g["vram_free_bytes"] = 2 * GiB
    res = fit.assess(busy, BUILD)
    assert res["verdict"] == "fits_gpu" and "free right now" in res["detail"]
