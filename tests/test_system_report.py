"""Probe isolation (raise / hang), get_report caching and refresh, models-folder
resolution, the disk probe, the subprocess helper, and the live report on
whatever machine runs the tests."""

from __future__ import annotations

import os
import sys
import threading
import time
import types

import pytest

import vvl.system as system
from vvl.system import _proc, models_dir, report as report_mod, schema
from vvl.system.probes import disk
from vvl.system.runner import ProbeContext, ProbeSpec, run_probes


@pytest.fixture(autouse=True)
def _fresh_cache():
    system.clear_cache()
    yield
    system.clear_cache()


# ---------------------------------------------------------------- runner ---

def test_probe_that_raises_is_recorded():
    def boom(ctx, log):
        raise RuntimeError("boom")

    def fine(ctx, log):
        return {"x": 1}

    out = run_probes([ProbeSpec("bad", boom, 2.0), ProbeSpec("good", fine, 2.0)], ProbeContext())
    assert out["bad"].ok is False and out["bad"].error == "RuntimeError: boom" and out["bad"].value is None
    assert out["good"].ok is True and out["good"].value == {"x": 1} and out["good"].error is None


def test_probe_that_hangs_times_out_without_stalling():
    release = threading.Event()

    def hang(ctx, log):
        release.wait(10)
        return {"late": True}

    t0 = time.perf_counter()
    out = run_probes([ProbeSpec("slow", hang, 0.3), ProbeSpec("quick", lambda c, l: {}, 2.0)], ProbeContext())
    elapsed = time.perf_counter() - t0
    release.set()
    assert elapsed < 1.5
    assert out["slow"].ok is False and "timed out" in out["slow"].error and out["slow"].value is None
    assert out["quick"].ok is True


def test_soft_failure_keeps_value():
    def partial(ctx, log):
        log.fail("swap unavailable")
        return {"total_bytes": 1}

    out = run_probes([ProbeSpec("memory", partial, 1.0)], ProbeContext())
    assert out["memory"].ok is False and out["memory"].value == {"total_bytes": 1}
    assert out["memory"].error == "swap unavailable"


def _fake_specs(counter, *, raise_gpu=False, hang_cpu=None):
    def os_probe(ctx, log):
        counter["os"] = counter.get("os", 0) + 1
        time.sleep(0.05)
        return {"system": "Linux", "name": "Test Linux", "arch": "x86_64", "is_wsl": False,
                "is_apple_silicon": False}

    def cpu_probe(ctx, log):
        if hang_cpu is not None:
            hang_cpu.wait(10)
        return {"model": "Test CPU", "logical_cores": 8, "physical_cores": 4,
                "features": {"avx": True, "avx2": True, "avx512": False, "fma": True, "f16c": True, "neon": False}}

    def mem_probe(ctx, log):
        return {"total_bytes": 32 * 1024**3, "available_bytes": 20 * 1024**3,
                "swap_total_bytes": 0, "swap_free_bytes": 0}

    def gpu_probe(ctx, log):
        if raise_gpu:
            raise OSError("nvidia-smi exploded")
        return {"state": "not_found", "gpus": []}

    return [
        ProbeSpec("os", os_probe, 2.0),
        ProbeSpec("cpu", cpu_probe, 0.3 if hang_cpu is not None else 2.0),
        ProbeSpec("memory", mem_probe, 2.0),
        ProbeSpec("gpu_nvidia", gpu_probe, 2.0),
        ProbeSpec("gpu_linux", lambda c, l: {"gpus": []}, 2.0),
        ProbeSpec("vulkan", lambda c, l: {"available": None, "detail": "vulkaninfo not found", "devices": []}, 2.0),
        ProbeSpec("disk", disk.probe, 3.0),
    ]


def _install(monkeypatch, specs):
    monkeypatch.setattr(report_mod, "default_probe_specs", lambda system_name: specs)
    monkeypatch.setattr(report_mod, "ProbeContext", lambda: ProbeContext(system="Linux"))


def test_report_survives_raising_and_hanging_probes(monkeypatch, tmp_path):
    release = threading.Event()
    _install(monkeypatch, _fake_specs({}, raise_gpu=True, hang_cpu=release))
    r = system.get_report(refresh=True, models_dir=str(tmp_path / "models"))
    release.set()
    assert schema.validate_report(r) == []
    probes = {p["name"]: p for p in r["probes"]}
    assert probes["gpu_nvidia"]["ok"] is False and "OSError" in probes["gpu_nvidia"]["error"]
    assert probes["cpu"]["ok"] is False and "timed out" in probes["cpu"]["error"]
    assert r["cpu"]["model"] is None and r["cpu"]["features"]["avx2"] is None  # null, not missing
    assert r["accelerators"]["cuda"]["available"] is None  # could not be determined
    codes = {w["code"] for w in r["warnings"]}
    assert "PROBES_INCOMPLETE" in codes and "MODELS_DIR_MISSING" in codes
    assert r["capability"]["recommended_backend"] == "cpu"


def test_get_report_caches_and_refreshes(monkeypatch, tmp_path):
    counter = {}
    _install(monkeypatch, _fake_specs(counter))
    a = system.get_report(models_dir=str(tmp_path))
    b = system.get_report()
    assert counter["os"] == 1 and a == b
    a["os"]["name"] = "mutated by caller"
    assert system.get_report()["os"]["name"] == "Test Linux"  # callers get copies
    system.get_report(refresh=True)
    assert counter["os"] == 2


def test_get_report_is_thread_safe(monkeypatch, tmp_path):
    counter = {}
    _install(monkeypatch, _fake_specs(counter))
    results, errors = [], []

    def worker():
        try:
            results.append(system.get_report(models_dir=str(tmp_path)))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert not errors and len(results) == 8
    assert counter["os"] == 1
    assert all(r == results[0] for r in results)


def test_changed_models_dir_rechecks_only_disk(monkeypatch, tmp_path):
    counter = {}
    _install(monkeypatch, _fake_specs(counter))
    first = system.get_report(models_dir=str(tmp_path))
    other = tmp_path / "elsewhere" / "models"
    second = system.get_report(models_dir=str(other))
    assert counter["os"] == 1
    assert first["disk"]["exists"] is True and second["disk"]["exists"] is False
    assert second["disk"]["models_dir"] == str(other) and second["disk"]["checked_path"] == str(tmp_path)
    assert "MODELS_DIR_MISSING" in {w["code"] for w in second["warnings"]}
    assert second["generated_at"] == first["generated_at"]


# ------------------------------------------------------ models directory ---

def test_models_dir_resolution_order(monkeypatch, tmp_path):
    settings = tmp_path / "settings.yaml"
    settings.write_text("models_dir: relative/models\nother: 1\n", encoding="utf-8")
    monkeypatch.setenv("VVL_SETTINGS_PATH", str(settings))
    monkeypatch.setenv("VVL_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(models_dir, "from_vvl_paths", lambda: None)

    assert models_dir.resolve(str(tmp_path / "explicit")) == (str(tmp_path / "explicit"), "argument")
    assert models_dir.resolve() == (str(tmp_path / "relative" / "models"), "settings")
    settings.write_text("not: [valid", encoding="utf-8")  # broken YAML falls through
    assert models_dir.resolve() == (str(tmp_path / "home" / "models"), "default")
    monkeypatch.setenv("VVL_SETTINGS_PATH", str(tmp_path / "PRIVATE.yaml"))  # never opened, and need not exist
    assert models_dir.from_settings() is None


def test_models_dir_from_vvl_paths_variants(monkeypatch, tmp_path):
    monkeypatch.setenv("VVL_SETTINGS_PATH", str(tmp_path / "none.yaml"))
    fake = types.ModuleType("vvl.paths")
    fake.models_dir = lambda: str(tmp_path / "from_paths")
    monkeypatch.setitem(sys.modules, "vvl.paths", fake)
    assert models_dir.resolve() == (str(tmp_path / "from_paths"), "vvl.paths")

    fake2 = types.ModuleType("vvl.paths")
    fake2.get_paths = lambda: {"models_dir": str(tmp_path / "from_dict")}
    monkeypatch.setitem(sys.modules, "vvl.paths", fake2)
    assert models_dir.resolve()[0] == str(tmp_path / "from_dict")

    broken = types.ModuleType("vvl.paths")

    def explode():
        raise RuntimeError("paths not ready")

    broken.models_dir = explode
    monkeypatch.setitem(sys.modules, "vvl.paths", broken)
    monkeypatch.setenv("VVL_HOME", str(tmp_path / "home"))
    assert models_dir.resolve() == (str(tmp_path / "home" / "models"), "default")


# ------------------------------------------------------------------ disk ---

def test_disk_probe_walks_up_to_existing_parent(tmp_path):
    from vvl.system.runner import ProbeLog

    target = tmp_path / "a" / "b" / "models"
    ctx = ProbeContext(system="Linux", env={}, models_dir=str(target), models_dir_source="argument")
    out = disk.probe(ctx, ProbeLog())
    assert out["exists"] is False and out["checked_path"] == str(tmp_path)
    assert out["free_bytes"] > 0 and out["total_bytes"] >= out["free_bytes"]
    assert out["is_synced_folder"] is False and out["writable"] is True


@pytest.mark.parametrize("path,expected", [
    ("/home/u/Dropbox/VVL/models", "Dropbox"),
    ("/home/u/Dropbox (Personal)/models", "Dropbox"),
    ("C:\\Users\\Pat\\OneDrive - State University\\Documents\\models", "OneDrive"),
    ("C:\\Users\\Pat\\OneDrive\\models", "OneDrive"),
    ("/Users/a/Library/Mobile Documents/com~apple~CloudDocs/models", "iCloud Drive"),
    ("/Users/a/Library/CloudStorage/GoogleDrive-a@example.org/My Drive/models", "Google Drive"),
    ("/Users/a/Library/CloudStorage/OneDrive-Contoso/models", "OneDrive"),
    ("/Users/a/Library/CloudStorage/Box-Box/models", "Box"),
    ("G:\\My Drive\\models", "Google Drive"),
    ("/datac/VoucherVisionLocal_Models", None),
    ("/home/u/dropboxish/models", None),
])
def test_synced_folder_detection(path, expected):
    assert disk.detect_synced_folder(path) == expected


def test_synced_folder_from_client_records(tmp_path):
    (tmp_path / ".dropbox").mkdir()
    (tmp_path / ".dropbox" / "info.json").write_text('{"personal": {"path": "%s", "host": 1}}'
                                                     % str(tmp_path / "Sync").replace("\\", "\\\\"))
    roots = disk.known_sync_roots({"OneDrive": str(tmp_path / "Cloud")}, str(tmp_path))
    assert disk.detect_synced_folder(str(tmp_path / "Sync" / "models"), roots) == "Dropbox"
    assert disk.detect_synced_folder(str(tmp_path / "Cloud" / "models"), roots) == "OneDrive"
    assert disk.detect_synced_folder(str(tmp_path / "plain"), roots) is None


def test_find_existing_parent(tmp_path):
    assert disk.find_existing_parent(str(tmp_path / "x" / "y")) == str(tmp_path)
    assert disk.find_existing_parent(str(tmp_path)) == str(tmp_path)


# -------------------------------------------------------- subprocess helper ---

def test_proc_run_success_and_exit_code():
    res = _proc.run([sys.executable, "-c", "import sys; print('hi'); sys.exit(3)"], timeout=5)
    assert res.found and res.returncode == 3 and res.stdout.strip() == "hi" and not res.ok


def test_proc_run_missing_tool():
    res = _proc.run(["vvl-definitely-not-a-real-tool"], timeout=1)
    assert res.found is False and not res.ok and "not found" in res.describe_failure()


def test_proc_run_kills_on_timeout():
    t0 = time.perf_counter()
    res = _proc.run([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5)
    assert res.timed_out and res.returncode is None and time.perf_counter() - t0 < 4


def test_decode_output_is_tolerant():
    assert _proc.decode_output("héllo".encode("utf-16")) == "héllo"
    assert _proc.decode_output(b"\xef\xbb\xbfok\r\n") == "ok\n"
    assert isinstance(_proc.decode_output(b"\xff\xfe\xfa bad \x80"), str)
    assert _proc.decode_output(None) == ""


# ------------------------------------------------------------------ live ---

def test_live_report_matches_contract():
    t0 = time.perf_counter()
    r = system.get_report(refresh=True)
    elapsed = time.perf_counter() - t0
    assert schema.validate_report(r) == [], schema.validate_report(r)
    assert elapsed < 10
    assert r["os"]["system"] and r["os"]["python"]
    assert r["memory"]["total_bytes"] and r["memory"]["total_bytes"] > 0
    assert r["capability"]["recommended_backend"] in schema.BACKENDS
    assert r["capability"]["summary"]
    names = [p["name"] for p in r["probes"]]
    assert {"os", "cpu", "memory", "vulkan", "disk"} <= set(names)
    # The second call is served from the cache.
    t1 = time.perf_counter()
    assert system.get_report() == r
    assert time.perf_counter() - t1 < 0.5


def test_cli_summary_runs():
    import subprocess

    out = subprocess.run([sys.executable, "-m", "vvl.system", "--summary"], capture_output=True, text=True,
                         timeout=60, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert out.returncode == 0, out.stderr
    assert "Backend:" in out.stdout and "Summary:" in out.stdout
