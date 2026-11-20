"""Representative machines (tests/fixtures/system_reports/*.json) against the
contract, the capability summary, and every Qwen3-VL build in the registry."""

from __future__ import annotations

import copy
import glob
import json
import os

import pytest
import yaml

from vvl.system import capability, fit, schema

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures", "system_reports")
REGISTRY = os.path.join(os.path.dirname(HERE), "vvl", "registry", "models_local.yaml")
REGEN = "uv run python tests/fixtures/system_reports/_build_fixtures.py"


def load(name):
    with open(os.path.join(FIXTURES, f"{name}.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def qwen_builds():
    with open(REGISTRY, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    fam = next(f for f in data["families"] if f["id"] == "qwen3-vl")
    return {b["id"]: b for b in fam["builds"]}


BUILD_ORDER = ["qwen3-vl-2b-q8", "qwen3-vl-4b-q8", "qwen3-vl-8b-q4", "qwen3-vl-8b-q8",
               "qwen3-vl-30b-a3b-q4", "qwen3-vl-32b-q4", "qwen3-vl-32b-q8"]

G, S, C, T = "fits_gpu", "fits_split", "cpu_only", "too_large"

# machine: (backend, tier, warning codes, verdicts in BUILD_ORDER)
EXPECTED = {
    "linux_workstation_2x_rtx6000ada": ("cuda", "workstation", set(), [G, G, G, G, G, G, G]),
    "windows_laptop_rtx4060_8gb": ("cuda", "mid", {"MODELS_DIR_MISSING"}, [G, G, G, S, T, T, T]),
    "macbook_air_m2_16gb": ("metal", "mid", {"MODELS_DIR_MISSING"}, [G, G, G, S, T, T, T]),
    "macbook_pro_m3_max_64gb_rosetta_dropbox": (
        "metal", "workstation", {"RUNNING_UNDER_ROSETTA", "MODELS_DIR_SYNCED"}, [G, G, G, G, G, G, G]),
    "office_pc_cpu_only_8gb_no_avx2": (
        "cpu", "cpu_only", {"CPU_NO_AVX2", "MODELS_DIR_MISSING", "MODELS_DIR_SYNCED"}, [C, T, T, T, T, T, T]),
    "linux_amd_rx7900xtx": ("vulkan", "high", {"MODELS_DIR_MISSING"}, [G, G, G, G, G, G, S]),
    "wsl2_ubuntu_rtx3090": ("cuda", "high", {"MODELS_DIR_WSL_WINDOWS_DRIVE"}, [G, G, G, G, G, G, S]),
    "intel_mac_2019_radeon_pro": ("cpu", "cpu_only", {"MODELS_DIR_MISSING"}, [C, C, C, C, C, C, T]),
}


def test_every_fixture_is_listed():
    names = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(FIXTURES, "*.json"))}
    assert names == set(EXPECTED)


def test_registry_has_the_expected_qwen_builds():
    assert list(qwen_builds()) == BUILD_ORDER


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_fixture_matches_contract(name):
    r = load(name)
    assert schema.validate_report(r) == []


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_fixture_capability_is_current(name):
    """The stored capability/warnings must equal what the code computes now."""
    r = load(name)
    stored_cap, stored_warn = r["capability"], r["warnings"]
    fresh = copy.deepcopy(r)
    fresh["capability"], fresh["warnings"] = None, []
    cap, warns = capability.compute(fresh)
    assert cap == stored_cap, f"capability changed; if intended, run: {REGEN}"
    assert warns == stored_warn, f"warnings changed; if intended, run: {REGEN}"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_fixture_capability_expectations(name):
    backend, tier, codes, _ = EXPECTED[name]
    r = load(name)
    cap = r["capability"]
    assert cap["recommended_backend"] == backend
    assert cap["tier"] == tier
    assert {w["code"] for w in r["warnings"]} == codes
    assert cap["backend_reason"] and cap["summary"]
    if backend == "cpu":
        assert cap["gpu_memory_budget_bytes"] == 0
    else:
        assert cap["gpu_memory_budget_bytes"] > 0
    assert 0 < cap["cpu_memory_budget_bytes"] < r["memory"]["total_bytes"]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_fit_verdicts_for_every_qwen_build(name):
    r = load(name)
    builds = qwen_builds()
    got = []
    for build_id in BUILD_ORDER:
        res = fit.assess(r, builds[build_id])
        assert schema.validate_fit(res) == [], res
        assert res["required_bytes"] == fit.estimate_required_bytes(builds[build_id])
        assert res["detail"].endswith(".") and "\n" not in res["detail"]
        got.append(res["verdict"])
    assert got == EXPECTED[name][3]


def test_verdicts_are_monotonic_in_model_size():
    """A bigger model never gets a better verdict on the same machine."""
    rank = {G: 0, S: 1, C: 1, T: 2}
    builds = qwen_builds()
    order = sorted(BUILD_ORDER, key=lambda b: fit.estimate_required_bytes(builds[b]))
    for name in EXPECTED:
        r = load(name)
        ranks = [rank[fit.assess(r, builds[b])["verdict"]] for b in order]
        assert ranks == sorted(ranks), name


def test_specific_summaries():
    assert load("linux_workstation_2x_rtx6000ada")["capability"]["summary"] == \
        "2x NVIDIA RTX 6000 Ada (48 GB each), 64 threads, 504 GB RAM"
    assert load("macbook_air_m2_16gb")["capability"]["summary"] == \
        "Apple M2 (10-core GPU), 8 CPU cores, 16 GB unified memory"
    assert load("office_pc_cpu_only_8gb_no_avx2")["capability"]["summary"] == \
        "Intel Core i5-3470, 4 threads, 7.9 GB RAM, no usable GPU"
    assert load("linux_amd_rx7900xtx")["capability"]["summary"].startswith("AMD Radeon RX 7900 XTX (24 GB)")


def test_windows_laptop_dedupes_nvidia_seen_twice():
    r = load("windows_laptop_rtx4060_8gb")
    assert [(g["vendor"], g["source"]) for g in r["gpus"]] == [("nvidia", "nvidia-smi"), ("intel", "registry")]
    assert r["capability"]["gpu_indices"] == [0]
