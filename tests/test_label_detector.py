"""The TextCollage label detector: preprocessing and tail math, the collage class
mapping, the pinned first-start download, its API, and (opt-in) the real model.

The real-model tests run only when VVL_TEST_DETECTOR_HOME points at a VVL home
that holds the downloaded detector (`VVL_HOME=<dir> vvl detector --download`,
once per format).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pytest
import yaml

from conftest import FIXTURE_IMAGE
from p2util import wait_job

from textcollage import detector as det
from textcollage.CollageEngine import CollageEngine, whole_image
from vvl import label_detector
from vvl.jobs import JobManager


# ------------------------------------------------------------- preprocessing --

def test_letterbox_long_side_and_stride_padding():
    img = np.zeros((1669, 1329, 3), np.uint8)
    out = det.letterbox(img, (1280, 1280))
    # long side 1280, short side round(1329 * 1280 / 1669) = 1019, padded to 1024
    assert out.shape == (1280, 1024, 3)
    assert (out[:, :2] == det.PAD_VALUE).all() and (out[:, -3:] == det.PAD_VALUE).all()
    x = det.preprocess(img, 1280)
    assert x.shape == (1, 3, 1280, 1024) and x.dtype == np.float32
    assert x.max() <= 1.0


def test_preprocess_is_rgb():
    img = np.zeros((64, 64, 3), np.uint8)
    img[..., 0] = 255                     # pure blue in BGR
    x = det.preprocess(img, 64)
    assert x[0, 2].min() == pytest.approx(1.0) and x[0, 0].max() == 0.0


def test_scale_boxes_undoes_the_letterbox():
    shape0 = (1669, 1329, 3)
    gain = 1280 / 1669
    pad_x = round((1024 - round(1329 * gain)) / 2 - 0.1)
    original = np.array([[100.0, 200.0, 700.0, 900.0]], np.float32)
    letterboxed = original * gain
    letterboxed[:, [0, 2]] += pad_x
    back = det.scale_boxes((1280, 1024), letterboxed.copy(), shape0)
    assert np.abs(back - original).max() < 0.01
    clipped = det.scale_boxes((1280, 1024), np.array([[-50.0, -5.0, 5000.0, 5000.0]], np.float32), shape0)
    assert clipped.tolist() == [[0.0, 0.0, 1329.0, 1669.0]]


def _tail_reference(boxes, scores, max_det):
    """The end2end tail written the slow, obvious way."""
    n, nc = scores.shape[1:]
    k = min(max_det, n)
    anchors = sorted(range(n), key=lambda a: (-scores[0, a].max(), a))[:k]
    pairs = sorted(((scores[0, a, c], i, c) for i, a in enumerate(anchors) for c in range(nc)),
                   key=lambda t: (-t[0], t[1] * nc + t[2]))[:k]
    return np.array([[*boxes[0, anchors[i]], s, c] for s, i, c in pairs], np.float32)


def test_end2end_tail_matches_the_reference():
    rng = np.random.default_rng(7)
    boxes = rng.uniform(0, 1280, (1, 60, 4)).astype(np.float32)
    scores = rng.uniform(0, 1, (1, 60, 9)).astype(np.float32)
    got = det.end2end_tail(boxes, scores, 12)
    assert got.shape == (1, 12, 6)
    assert np.allclose(got[0], _tail_reference(boxes, scores, 12))
    filtered = det.end2end_filter(got[0], 0.9, 5)
    assert len(filtered) <= 5 and (filtered[:, 4] > 0.9).all()


# ------------------------------------------------------------------ collage --

class _Fixed:
    label = "fixed"

    def __init__(self, detections):
        self.detections = detections

    def detect(self, image):
        return self.detections


def test_collage_maps_the_model_class_names(tmp_path):
    import cv2

    path = tmp_path / "sheet.jpg"
    cv2.imwrite(str(path), np.full((400, 300, 3), 200, np.uint8))
    engine = CollageEngine(
        _Fixed([det.Detection("Label", 0.9, (10.4, 20.0, 110.9, 80.0)),
                det.Detection("Attached Item", 0.8, (0, 0, 50, 50)),
                det.Detection("Label", 0.5, (5, 5, 5, 40)),           # empty box: dropped
                det.Detection("Unknown", 0.9, (0, 0, 9, 9))]),        # not a TextCollage class
        collage_classes=["label"], output_path=None)
    _, boxes = engine.get_bounding_boxes(str(path))
    assert boxes["label"] == [[10, 20, 110, 80]]
    assert boxes["attached_item"] == [[0, 0, 50, 50]]
    out = engine.run(str(path))
    assert out["position_original"]["label"] == [[10, 20, 110, 80]]
    assert out["base64image_text_collage"]


def test_whole_image_needs_no_detector(tmp_path):
    import cv2

    path = tmp_path / "sheet.jpg"
    cv2.imwrite(str(path), np.zeros((40, 30, 3), np.uint8))
    out = whole_image(str(path))
    assert out["position_original"] == {} and out["base64image_text_collage"]


# ----------------------------------------------------------------- download --

FILES = {"onnx/model.onnx": b"fake onnx weights " * 4000, "coreml_fp32/model.json": b'{"format": "coreml_fp32"}'}


@pytest.fixture
def fake_lock(tmp_path, fake_server):
    lock = {
        "repo": "TestOrg/test_detector", "revision": "0123456789abcdef0123", "imgsz": 64, "conf": 0.25,
        "max_det": 10, "classes": ["Label"],
        "formats": {"onnx": {"files": [{"path": "onnx/model.onnx", "size_bytes": len(FILES["onnx/model.onnx"]),
                                        "sha256": hashlib.sha256(FILES["onnx/model.onnx"]).hexdigest()}]}},
    }
    path = tmp_path / "detector.lock.yaml"
    path.write_text(yaml.safe_dump(lock), encoding="utf-8")
    from conftest import FileRoute

    for name, data in FILES.items():
        fake_server.files[f"/{lock['repo']}/resolve/{lock['revision']}/{name}"] = FileRoute(data)
    return path


def _store(fake_lock, fake_server, **kw):
    return label_detector.DetectorStore(JobManager(), lock=lambda: label_detector.load_lock(fake_lock),
                                        hf_base_url=fake_server.base_url, fmt=lambda: "onnx", **kw)


def test_download_verifies_and_marks_installed(fake_lock, fake_server):
    loaded = []
    store = _store(fake_lock, fake_server, on_installed=lambda: loaded.append(1))
    assert store.state()["state"] == "not_installed"
    done = wait_job(store.install())
    assert done["state"] == "done", done
    assert done["kind"] == "detector_download" and done["label"] == "Label detector"
    assert store.is_installed() and store.state()["state"] == "installed"
    assert loaded == [1]
    root = label_detector.install_dir(store.lock())
    assert (root / "onnx" / "model.onnx").read_bytes() == FILES["onnx/model.onnx"]
    assert json.loads((root / "verified.json").read_text())["onnx"]["onnx/model.onnx"]

    # A second install with the file in place re-checks it and downloads nothing.
    hits = fake_server.files[next(iter(fake_server.files))].hits
    (root / "verified.json").unlink()
    assert not store.is_installed()
    again = wait_job(store.install())
    assert again["state"] == "done" and again["files"][0]["state"] == "skipped"
    assert fake_server.files[next(iter(fake_server.files))].hits == hits


def test_a_damaged_file_fails_the_check(fake_lock, fake_server):
    from conftest import FileRoute

    store = _store(fake_lock, fake_server)
    lock = store.lock()
    key = f"/{lock['repo']}/resolve/{lock['revision']}/onnx/model.onnx"
    fake_server.files[key] = FileRoute(b"X" * len(FILES["onnx/model.onnx"]))
    done = wait_job(store.install())
    assert done["state"] == "error" and done["error"]["code"] == "CHECKSUM_MISMATCH"
    state = store.state()
    assert state["state"] == "error" and state["error"]["code"] == "CHECKSUM_MISMATCH"
    assert not store.is_installed()


def test_load_detector_refuses_before_the_download(fake_lock):
    with pytest.raises(label_detector.DetectorNotInstalled):
        label_detector.load_detector(label_detector.load_lock(fake_lock), "onnx")


def test_the_pinned_lock_is_complete():
    lock = label_detector.load_lock()
    assert lock["revision"] and len(lock["revision"]) == 40
    assert lock["classes"][3] == "Label" and len(lock["classes"]) == 9
    for fmt in label_detector.FORMATS:
        files = label_detector.files_for(lock, fmt)
        assert files and all(len(f["sha256"]) == 64 and f["size_bytes"] > 0 for f in files)
    assert label_detector.default_format() == ("coreml_fp32" if sys.platform == "darwin" else "onnx")


def test_format_override(monkeypatch):
    monkeypatch.setenv("VVL_DETECTOR_FORMAT", "onnx")
    assert label_detector.default_format() == "onnx"


# ---------------------------------------------------------------------- API --

def test_detector_routes(make_app, auth, fake_lock, fake_server):
    app = make_app(VVL_DETECTOR_LOCK_PATH=str(fake_lock), VVL_HF_BASE_URL=fake_server.base_url)
    app.extensions["vvl"]["runtime"].detector._fmt = lambda: "onnx"
    client = app.test_client()
    body = client.get("/v1/detector", headers=auth).get_json()
    assert body["state"] == "not_installed" and body["format"] == "onnx" and body["runtime"] == "ONNX Runtime, CPU"
    assert body["size_bytes"] == len(FILES["onnx/model.onnx"])
    started = client.post("/v1/detector/install", headers=auth, json={})
    assert started.status_code == 202
    job = app.extensions["vvl"]["runtime"].jobs.get(started.get_json()["job"]["id"])
    assert wait_job(job)["state"] == "done"
    assert client.get("/v1/detector", headers=auth).get_json()["state"] == "installed"
    assert client.post("/v1/detector/install", headers=auth, json={}).status_code == 409
    assert client.get("/v1/detector").status_code in (401, 403)


def test_doctor_warns_before_the_download(monkeypatch):
    from vvl import doctor

    monkeypatch.setattr(label_detector, "is_installed", lambda lock, fmt: False)
    status, detail = doctor._textcollage()
    assert status == "warn" and "not downloaded yet" in detail


# ------------------------------------------------------------- real model --

REAL_HOME = os.environ.get("VVL_TEST_DETECTOR_HOME")
needs_real = pytest.mark.skipif(not REAL_HOME, reason="set VVL_TEST_DETECTOR_HOME to a home with the detector")


def _real(monkeypatch, fmt):
    monkeypatch.setenv("VVL_HOME", REAL_HOME)
    lock = label_detector.load_lock()
    if not label_detector.is_installed(lock, fmt):
        pytest.skip(f"the {fmt} detector is not downloaded in {REAL_HOME}")
    return label_detector.load_detector(lock, fmt)


@needs_real
def test_real_onnx_finds_the_labels(monkeypatch):
    found = _real(monkeypatch, "onnx").detect(det.imread(FIXTURE_IMAGE))
    names = [d.cls for d in found]
    assert names.count("Label") >= 5 and "Barcode" in names
    assert all(d.score > 0.25 for d in found)


# The ONNX detections on the fixture image on an Apple M2 Pro (onnxruntime 1.32.0, CPU).
# Other machines must match them closely: the letterbox resize is bit-exact across CPUs
# (INTER_LINEAR_EXACT), so only the CPU kernels' float rounding may differ.
MAC_REFERENCE = [
    ("Label", 0.9599, (4.09, 1023.55, 416.78, 1431.81)),
    ("Label", 0.9160, (421.15, 1025.91, 858.64, 1237.66)),
    ("Label", 0.8833, (0.00, 1.64, 1317.76, 839.36)),
    ("Label", 0.8402, (0.00, 1432.37, 589.46, 1506.66)),
    ("Label", 0.8225, (10.86, 1512.86, 348.78, 1592.60)),
    ("Label", 0.8113, (0.00, 851.98, 1318.54, 1008.23)),
    ("Barcode", 0.6007, (888.54, 1031.52, 1259.98, 1221.62)),
    ("Label", 0.5234, (76.20, 1598.77, 257.30, 1667.02)),
    ("Barcode", 0.4890, (888.92, 1031.79, 1259.93, 1222.39)),
]


@needs_real
def test_real_onnx_matches_the_mac_reference(monkeypatch):
    found = _real(monkeypatch, "onnx").detect(det.imread(FIXTURE_IMAGE))
    assert [d.cls for d in found] == [c for c, _s, _b in MAC_REFERENCE]
    for d, (_c, score, box) in zip(found, MAC_REFERENCE):
        assert abs(d.score - score) < 0.005, (d, score)
        assert max(abs(a - b) for a, b in zip(d.box, box)) < 0.5, (d, box)


@needs_real
@pytest.mark.skipif(sys.platform != "darwin", reason="Core ML runs on macOS only")
def test_real_coreml_matches_onnx(monkeypatch):
    img = det.imread(FIXTURE_IMAGE)
    a = _real(monkeypatch, "onnx").raw(img)
    b = _real(monkeypatch, "coreml_fp32").raw(img)
    assert a.shape == b.shape and (a[:, 5] == b[:, 5]).all()
    assert np.abs(a[:, :4] - b[:, :4]).max() < 0.05 and np.abs(a[:, 4] - b[:, 4]).max() < 1e-3
