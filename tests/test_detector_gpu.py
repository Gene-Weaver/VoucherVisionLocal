"""The label detector on an NVIDIA GPU: the pinned runtime lock, who gets it, its
download-and-install job, the worker process (with a stand-in onnxruntime), the CPU
fallback, the API, and (opt-in) the real GPU.

The real-GPU test runs only with VVL_TEST_DETECTOR_GPU=1 and VVL_TEST_DETECTOR_HOME
pointing at a VVL home that holds the onnx detector and the installed GPU runtime.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from conftest import FIXTURE_IMAGE, FileRoute
from p2util import load_report, wait_job

from textcollage import detector as det
from vvl import detector_gpu
from vvl.jobs import JobManager

REPO = Path(__file__).resolve().parent.parent
MAC = REPO / "tests" / "fixtures" / "system_reports" / "macbook_air_m2_16gb.json"
AMD = REPO / "tests" / "fixtures" / "system_reports" / "linux_amd_rx7900xtx.json"
WINDOWS = REPO / "tests" / "fixtures" / "system_reports" / "windows_laptop_rtx4060_8gb.json"


@pytest.fixture
def gpu_on(monkeypatch):
    """conftest turns the GPU runtime off for every test; these tests turn it back on."""
    monkeypatch.delenv("VVL_DETECTOR_GPU", raising=False)
    monkeypatch.setattr(detector_gpu, "host_platform", lambda: "linux-x86_64")
    monkeypatch.setattr(detector_gpu, "host_glibc", lambda: "2.39")
    monkeypatch.setattr("vvl.downloads.free_bytes", lambda path: 10 ** 15)   # the disk check has its own test


# ------------------------------------------------------------------- lock --

def _tool():
    spec = importlib.util.spec_from_file_location("lock_detector_gpu", REPO / "tools" / "lock_detector_gpu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_lock_matches_uv_lock():
    tool = _tool()
    assert tool.render(tool.build()) == (REPO / "vvl" / "registry" / "detector_gpu.lock.yaml").read_text(), \
        "run: uv run python tools/lock_detector_gpu.py"


def test_the_pinned_lock_is_complete():
    lock = detector_gpu.load_lock()
    assert lock["cuda"] == "12" and lock["onnxruntime_gpu"].startswith("1.26.")
    for platform in ("linux-x86_64", "windows-x86_64"):
        wheels = detector_gpu.wheels_for(lock, platform)
        names = {w["name"] for w in wheels}
        assert {"onnxruntime-gpu", "nvidia-cudnn-cu12", "nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12"} <= names
        # numpy, protobuf, ... come from the app's environment, never from the runtime folder.
        assert not names & {"numpy", "protobuf", "flatbuffers", "packaging", "onnxruntime"}
        for w in wheels:
            assert len(w["sha256"]) == 64 and w["size_bytes"] > 0 and w["url"].endswith(w["filename"])
        assert 1.5e9 < detector_gpu.size_bytes(lock, platform) < 2.5e9
    assert lock["platforms"]["linux-x86_64"]["min_glibc"] == "2.27"


# ------------------------------------------------------------ eligibility --

def test_a_workstation_gets_the_gpu_with_most_free_memory():
    lock = detector_gpu.load_lock()
    got = detector_gpu.eligibility(lock, load_report(), platform="linux-x86_64", glibc="2.35")
    assert got["usable"] and got["device"]["index"] == 1 and "RTX 6000 Ada" in got["device"]["name"]


def test_a_windows_laptop_gets_it():
    got = detector_gpu.eligibility(detector_gpu.load_lock(), load_report(WINDOWS), platform="windows-x86_64")
    assert got["usable"] and "4060" in got["device"]["name"]


@pytest.mark.parametrize("change, platform, glibc, reason", [
    (None, "macos-arm64", None, "not_supported"),                 # macOS, Linux on ARM, ...
    ("mac", "linux-x86_64", "2.39", "no_nvidia_gpu"),
    ("amd", "linux-x86_64", "2.39", "no_nvidia_gpu"),
    ("old_driver", "linux-x86_64", "2.39", "driver_too_old"),
    ("old_gpu", "linux-x86_64", "2.39", "gpu_too_old"),
    (None, "linux-x86_64", "2.17", "glibc_too_old"),
])
def test_who_does_not_get_it(change, platform, glibc, reason):
    report = {"mac": lambda: load_report(MAC), "amd": lambda: load_report(AMD)}.get(change, load_report)()
    for gpu in report.get("gpus") or []:
        if change == "old_driver":
            gpu["driver_version"] = "470.82.01"
        if change == "old_gpu":
            gpu["details"]["compute_capability"] = "5.0"
    got = detector_gpu.eligibility(detector_gpu.load_lock(), report, platform=platform, glibc=glibc)
    assert not got["usable"] and got["reason"] == reason and got["message"]


# --------------------------------------------------------------- the job --

def _wheel(name: str, version: str, files: dict[str, bytes]) -> bytes:
    """A minimal valid py3-none-any wheel."""
    dist = f"{name.replace('-', '_')}-{version}.dist-info"
    files = dict(files)
    files[f"{dist}/METADATA"] = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode()
    files[f"{dist}/WHEEL"] = b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    rows = []
    for path, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append([path, f"sha256={digest}", str(len(data))])
    rows.append([f"{dist}/RECORD", "", ""])
    record = io.StringIO()
    csv.writer(record, lineterminator="\n").writerows(rows)
    files[f"{dist}/RECORD"] = record.getvalue().encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path, data in files.items():
            zf.writestr(path, data)
    return buf.getvalue()


FAKE_ORT = (REPO / "tests" / "fakes" / "fake_onnxruntime.py").read_bytes()


@pytest.fixture
def fake_gpu_lock(tmp_path, fake_server):
    wheels = {
        "onnxruntime_gpu-9.9.0-py3-none-any.whl": _wheel("onnxruntime-gpu", "9.9.0", {"onnxruntime/__init__.py": FAKE_ORT}),
        "nvidia_fake_cu12-1.0-py3-none-any.whl": _wheel("nvidia-fake-cu12", "1.0", {"nvidia/fake/lib/libfake.so": b"x" * 5000}),
    }
    entries = []
    for filename, data in wheels.items():
        fake_server.files[f"/{filename}"] = FileRoute(data)
        entries.append({"name": filename.split("-")[0].replace("_", "-"), "version": filename.split("-")[1],
                        "filename": filename, "url": f"https://example.invalid/{filename}",
                        "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)})
    lock = {"onnxruntime_gpu": "9.9.0", "cuda": "12", "min_compute_capability": "6.0",
            "platforms": {"linux-x86_64": {"min_driver": "525.60.13", "min_glibc": "2.27", "wheels": entries}}}
    path = tmp_path / "detector_gpu.lock.yaml"
    path.write_text(yaml.safe_dump(lock), encoding="utf-8")
    return path


def _store(fake_gpu_lock, fake_server, report=None, **kw):
    report = report if report is not None else load_report()
    return detector_gpu.GpuRuntimeStore(JobManager(), lock=lambda: detector_gpu.load_lock(fake_gpu_lock),
                                        report=lambda: report, base_url=fake_server.base_url, **kw)


needs_uv = pytest.mark.skipif(detector_gpu.uv_binary() is None, reason="uv is not on PATH")


@needs_uv
def test_download_and_install(gpu_on, fake_gpu_lock, fake_server):
    loaded = []
    store = _store(fake_gpu_lock, fake_server, on_installed=lambda: loaded.append(1))
    first = store.state()
    assert first["state"] == "not_installed" and first["device"]["index"] == 1 and first["size_bytes"] > 0
    assert store.should_auto_install()
    done = wait_job(store.install(), timeout=120)
    assert done["state"] == "done", done
    assert done["kind"] == "detector_gpu_download" and done["label"] == "GPU runtime"
    assert store.is_installed() and store.state()["state"] == "installed" and loaded == [1]
    assert not store.should_auto_install()
    root = detector_gpu.install_dir(store.lock(), "linux-x86_64")
    assert (root / "onnxruntime" / "__init__.py").read_bytes() == FAKE_ORT
    assert (root / "nvidia" / "fake" / "lib" / "libfake.so").is_file()
    # The wheels and uv's cache are gone; only the installed runtime stays.
    assert sorted(p.name for p in detector_gpu.runtime_root().iterdir()) == [root.name]


def test_a_damaged_wheel_fails_the_check(gpu_on, fake_gpu_lock, fake_server):
    key = next(k for k in fake_server.files if "onnxruntime" in k)
    fake_server.files[key] = FileRoute(b"not the wheel" * 100)
    store = _store(fake_gpu_lock, fake_server)
    done = wait_job(store.install(), timeout=60)
    assert done["state"] == "error" and done["error"]["code"] in ("CHECKSUM_MISMATCH", "SIZE_MISMATCH")
    assert store.state()["state"] == "error" and not store.should_auto_install()   # no retry loop at each start


def test_too_little_disk_space(gpu_on, fake_gpu_lock, fake_server, monkeypatch):
    monkeypatch.setattr("vvl.downloads.free_bytes", lambda path: 1000)
    done = wait_job(_store(fake_gpu_lock, fake_server).install())
    assert done["state"] == "error" and done["error"]["code"] == "INSUFFICIENT_DISK_SPACE"


def test_nothing_to_do_without_a_usable_gpu(gpu_on, fake_gpu_lock, fake_server, monkeypatch):
    mac = _store(fake_gpu_lock, fake_server, report=load_report(MAC))
    assert mac.state()["state"] == "not_needed" and not mac.should_auto_install()
    assert mac.maybe_auto_install() is None
    monkeypatch.setenv("VVL_DETECTOR_GPU", "0")
    off = _store(fake_gpu_lock, fake_server)
    assert off.state()["state"] == "disabled" and not off.should_auto_install()


# ------------------------------------------------------------ the worker --

def _runtime_dir(tmp_path) -> Path:
    root = tmp_path / "gpu_runtime"
    (root / "onnxruntime").mkdir(parents=True)
    shutil.copy(REPO / "tests" / "fakes" / "fake_onnxruntime.py", root / "onnxruntime" / "__init__.py")
    return root


def _model_dir(tmp_path) -> Path:
    model = tmp_path / "model" / "onnx" / "model.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"fake")
    return model.parent.parent


def _gpu(runtime_dir: Path, **env) -> dict:
    base = {k: v for k, v in os.environ.items() if not k.startswith("FAKE_ORT_")}
    base.update(env)
    return {"runtime_dir": str(runtime_dir), "device_index": 0, "device_name": "Fake RTX",
            "env": detector_gpu.worker_env(runtime_dir, base), "python": sys.executable}


def test_the_worker_runs_the_model_on_the_gpu(tmp_path):
    backend = det.CudaWorkerBackend(_model_dir(tmp_path), _gpu(_runtime_dir(tmp_path)))
    try:
        assert backend.label == "ONNX Runtime, Fake RTX" and backend.onnxruntime_version == "9.9.0-fake"
        out = backend.run(np.zeros((1, 3, 64, 96), np.float32))
        assert out.shape == (300, 6) and out.dtype == np.float32
        assert out[0].tolist() == pytest.approx([1.0, 2.0, 64.0, 96.0, 0.9, 3.0])
    finally:
        backend.close()
    assert backend.proc.poll() is not None       # closing stdin ends the worker


class _CpuStandIn:
    format = "onnx"
    label = "ONNX Runtime, CPU (stand-in)"

    def __init__(self, model_dir):
        pass

    def run(self, x):
        return np.zeros((300, 6), np.float32)

    def close(self):
        pass


def test_a_gpu_that_does_not_start_leaves_the_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(det, "OnnxBackend", _CpuStandIn)
    detector = det.ArchivalDetector(_model_dir(tmp_path), "onnx", classes=["Label"], imgsz=64,
                                    gpu=_gpu(_runtime_dir(tmp_path), FAKE_ORT_NO_CUDA="1"))
    assert detector.label == "ONNX Runtime, CPU (stand-in)"
    assert "no CUDA provider" in detector.gpu_error


def test_a_gpu_that_fails_mid_run_falls_back_to_the_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(det, "OnnxBackend", _CpuStandIn)
    detector = det.ArchivalDetector(_model_dir(tmp_path), "onnx", classes=["Label"], imgsz=64,
                                    gpu=_gpu(_runtime_dir(tmp_path), FAKE_ORT_CRASH="1"))
    assert isinstance(detector.backend, det.CudaWorkerBackend)
    worker = detector.backend
    assert detector.detect(np.full((80, 120, 3), 200, np.uint8)) == []
    assert detector.label == "ONNX Runtime, CPU (stand-in)" and "stopped" in detector.gpu_error
    assert worker.proc.poll() is not None


def test_the_worker_must_load_the_runtime_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(det, "OnnxBackend", _CpuStandIn)
    gpu = _gpu(_runtime_dir(tmp_path))
    gpu["runtime_dir"] = str(tmp_path / "somewhere_else")
    detector = det.ArchivalDetector(_model_dir(tmp_path), "onnx", classes=["Label"], imgsz=64, gpu=gpu)
    assert "wrong onnxruntime" in detector.gpu_error


def test_worker_config(gpu_on, fake_gpu_lock, monkeypatch):
    lock = detector_gpu.load_lock(fake_gpu_lock)
    assert detector_gpu.worker_config(lock, load_report()) is None            # not installed yet
    root = detector_gpu.install_dir(lock, "linux-x86_64")
    (root / "onnxruntime").mkdir(parents=True)
    (root / "onnxruntime" / "__init__.py").write_text("")
    (root / detector_gpu.MARKER).write_text(json.dumps(detector_gpu._marker_for(lock, "linux-x86_64")))
    cfg = detector_gpu.worker_config(lock, load_report())
    assert cfg["runtime_dir"] == str(root) and cfg["device_index"] == 1
    assert cfg["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(root)
    assert cfg["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert detector_gpu.worker_config(lock, load_report(MAC)) is None
    monkeypatch.setenv("VVL_DETECTOR_GPU", "0")
    assert detector_gpu.worker_config(lock, load_report()) is None


# -------------------------------------------------------------------- API --

def test_routes(gpu_on, make_app, auth, fake_gpu_lock, fake_server):
    report = load_report()
    app = make_app(VVL_DETECTOR_GPU_LOCK_PATH=str(fake_gpu_lock), VVL_DETECTOR_GPU_BASE_URL=fake_server.base_url,
                   VVL_SYSTEM_REPORT=lambda refresh=False: report)
    client = app.test_client()
    gpu = client.get("/v1/detector", headers=auth).get_json()["gpu"]
    assert gpu["state"] == "not_installed" and gpu["device"]["index"] == 1 and gpu["cuda"] == "12"
    if detector_gpu.uv_binary() is not None:
        started = client.post("/v1/detector/gpu/install", headers=auth, json={})
        assert started.status_code == 202
        job = app.extensions["vvl"]["runtime"].jobs.get(started.get_json()["job"]["id"])
        assert wait_job(job, timeout=120)["state"] == "done"
        assert client.get("/v1/detector", headers=auth).get_json()["gpu"]["state"] == "installed"
        assert client.post("/v1/detector/gpu/install", headers=auth, json={}).status_code == 409


def test_no_route_without_a_gpu(gpu_on, make_app, auth, fake_gpu_lock):
    mac = load_report(MAC)
    app = make_app(VVL_DETECTOR_GPU_LOCK_PATH=str(fake_gpu_lock), VVL_SYSTEM_REPORT=lambda refresh=False: mac)
    client = app.test_client()
    assert client.get("/v1/detector", headers=auth).get_json()["gpu"]["state"] == "not_needed"
    refused = client.post("/v1/detector/gpu/install", headers=auth, json={})
    assert refused.status_code == 409 and refused.get_json()["error_code"] == "CONFLICT"


def test_the_detector_download_starts_the_gpu_runtime(gpu_on, make_app, fake_gpu_lock, fake_server):
    report = load_report()
    app = make_app(VVL_DETECTOR_GPU_LOCK_PATH=str(fake_gpu_lock), VVL_DETECTOR_GPU_BASE_URL=fake_server.base_url,
                   VVL_SYSTEM_REPORT=lambda refresh=False: report)
    rt = app.extensions["vvl"]["runtime"]
    rt.detector.on_installed()                 # what the detector download calls when it is done
    job = rt.jobs.active_job("detector_gpu_download") or rt.jobs.latest_for("detector_gpu_download", "linux-x86_64")
    assert job is not None and job.label == "GPU runtime"
    wait_job(job, timeout=120)


# ----------------------------------------------------------- the real GPU --

REAL_HOME = os.environ.get("VVL_TEST_DETECTOR_HOME")


@pytest.mark.skipif(not (REAL_HOME and os.environ.get("VVL_TEST_DETECTOR_GPU") == "1"),
                    reason="set VVL_TEST_DETECTOR_GPU=1 and VVL_TEST_DETECTOR_HOME (detector + GPU runtime installed)")
def test_real_gpu_matches_the_cpu(monkeypatch):
    from vvl import label_detector

    monkeypatch.setenv("VVL_HOME", REAL_HOME)
    monkeypatch.delenv("VVL_DETECTOR_GPU", raising=False)
    lock = label_detector.load_lock()
    if detector_gpu.worker_config() is None:
        pytest.skip(f"no usable GPU runtime in {REAL_HOME}")
    image = det.imread(FIXTURE_IMAGE)
    gpu = label_detector.load_detector(lock, "onnx")
    try:
        assert isinstance(gpu.backend, det.CudaWorkerBackend) and gpu.gpu_error is None
        on_gpu = gpu.raw(image)
    finally:
        gpu.backend.close()
    monkeypatch.setenv("VVL_DETECTOR_GPU", "0")
    on_cpu = label_detector.load_detector(lock, "onnx").raw(image)
    # The same boxes (rows may swap where two scores are nearly equal).
    assert len(on_gpu) == len(on_cpu)
    key = lambda rows: rows[np.lexsort((rows[:, 1], rows[:, 0], rows[:, 5]))]  # noqa: E731
    a, b = key(on_gpu), key(on_cpu)
    assert np.array_equal(a[:, 5], b[:, 5])
    assert np.abs(a[:, :4] - b[:, :4]).max() < 0.5 and np.abs(a[:, 4] - b[:, 4]).max() < 0.01
