# Ports code from LeafMachine3 (leafmachine3/inference/ultra_replacements.py and
# coreml_native.py), which ports Ultralytics 8.4 (https://github.com/ultralytics/ultralytics),
# Copyright Ultralytics Inc. All three are AGPL-3.0, as is this file.
"""TextCollage's label detector: LeafMachine3's archival detector, without torch.

The model is a YOLO26x detector at 1280 (Hugging Face
``phyloforfun/lm3_archival_detector__yolo26x_det_1280``). It is an *end2end*
export: the network emits its final, score-sorted detections, so there is no
NMS step. Two runtimes take the same input tensor and give the same rows
(x1, y1, x2, y2, score, class) to float32 rounding:

* ``coreml_fp32`` (macOS): Core ML on the Apple GPU. The package holds every
  input shape the letterbox below can make (long side 1280, short side a
  multiple of 32) and stops at the YOLO26 head's split; :func:`end2end_tail`
  runs the graph's top-k tail in numpy.
* ``onnx`` (everywhere): onnxruntime on the CPU, the full end2end graph.

Preprocessing and box scaling follow LeafMachine3 line for line (letterbox with
``INTER_LINEAR_EXACT``, gray 114 padding, BGR to RGB, the reciprocal-multiply
form of ``/ 255`` and ``/ gain``), so a sheet gets the same boxes here as in LM3.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FORMATS = ("coreml_fp32", "onnx")
PAD_VALUE = 114                    # LetterBox(padding_value=114)
RESIZE = cv2.INTER_LINEAR_EXACT    # bit-exact bilinear: the same pixels on x86 and ARM (see LM3)
STRIDE = 32
COREML_UNITS = "CPU_AND_GPU"       # Core ML's CPU-only path crashes on multi-shape packages; the
                                   # Neural Engine computes in float16. Neither is used.


class DetectorError(RuntimeError):
    """The detector's files are missing, damaged, or cannot be loaded."""


@dataclass(frozen=True)
class Detection:
    cls: str          # the model's class name, e.g. "Label", "Attached Item"
    score: float
    box: tuple        # (x1, y1, x2, y2) in original image pixels


# ------------------------------------------------------------------ image I/O --

def imread(path) -> np.ndarray | None:
    """BGR uint8, like cv2.imread, but also for non-ASCII paths on Windows."""
    try:
        buf = np.fromfile(str(path), np.uint8)
    except OSError:
        return None
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is not None and image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


# -------------------------------------------------------------- preprocessing --

def letterbox(img: np.ndarray, new_shape: tuple[int, int], stride: int = STRIDE, auto: bool = True) -> np.ndarray:
    """Ultralytics ``LetterBox`` for the predict path: long side to ``new_shape``, the short side
    padded (centered, gray 114) up to the next multiple of ``stride`` when ``auto``."""
    shape = img.shape[:2]                                  # (h, w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = round(shape[1] * r), round(shape[0] * r)   # (w, h)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=RESIZE)
    return cv2.copyMakeBorder(img, int(top), int(bottom), int(left), int(right),
                              cv2.BORDER_CONSTANT, value=(PAD_VALUE, PAD_VALUE, PAD_VALUE))


def _scale(x: np.ndarray, divisor: float) -> np.ndarray:
    """In-place ``x /= divisor`` as torch's CUDA kernel does it (float32 reciprocal, then multiply).
    LM3's reference results use this form; true division moves boxes by up to 0.4 px."""
    x *= np.float32(1.0) / np.float32(divisor)
    return x


def preprocess(img_bgr: np.ndarray, imgsz: int) -> np.ndarray:
    """letterbox -> BGR to RGB -> (1, 3, H, W) float32 in [0, 1]."""
    im = letterbox(img_bgr, (imgsz, imgsz))[None]
    im = im[..., ::-1].transpose((0, 3, 1, 2))
    return _scale(np.ascontiguousarray(im).astype(np.float32), 255)


# ------------------------------------------------------------- postprocessing --

def scale_boxes(img1_shape, boxes: np.ndarray, img0_shape) -> np.ndarray:
    """Undo the letterbox on xyxy boxes (in place), then clip to the original image."""
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad_x = round((img1_shape[1] - round(img0_shape[1] * gain)) / 2 - 0.1)
    pad_y = round((img1_shape[0] - round(img0_shape[0] * gain)) / 2 - 0.1)
    boxes[..., [0, 2]] -= pad_x
    boxes[..., [1, 3]] -= pad_y
    _scale(boxes[..., :4], gain)
    h, w = img0_shape[:2]
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h)
    return boxes


def _topk(a: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """onnxruntime ``TopK(largest=1, sorted=1)`` on the last axis; equal values keep index order."""
    idx = np.argsort(-a, axis=-1, kind="stable")[..., :k]
    return np.take_along_axis(a, idx, -1), idx


def end2end_tail(boxes: np.ndarray, scores: np.ndarray, max_det: int) -> np.ndarray:
    """The YOLO26 end2end tail in numpy: (b, n, 4) boxes and (b, n, nc) scores -> (b, k, 6).

    Keep the ``k`` anchors with the highest class score, then the ``k`` best (anchor, class)
    pairs among them; each row is box, score, class -- what the ONNX graph's own tail does.
    """
    b, n, nc = scores.shape
    k = min(int(max_det), n)
    _, anchors = _topk(scores.max(-1), k)
    cand = np.take_along_axis(scores, np.repeat(anchors[..., None], nc, -1), 1)
    vals, flat = _topk(cand.reshape(b, -1), k)
    anchor = np.take_along_axis(anchors, flat // nc, 1)
    cls = (flat % nc).astype(scores.dtype)
    picked = np.take_along_axis(boxes, np.repeat(anchor[..., None], boxes.shape[-1], -1), 1)
    return np.concatenate([picked, vals[..., None], cls[..., None]], -1)


def end2end_filter(pred: np.ndarray, conf: float, max_det: int) -> np.ndarray:
    """Ultralytics' end2end branch of non_max_suppression: rows above ``conf``, at most ``max_det``."""
    return pred[pred[:, 4] > conf][:max_det]


# ------------------------------------------------------------------- runtimes --

class OnnxBackend:
    """onnxruntime on the CPU with the full end2end graph."""

    format = "onnx"
    label = "ONNX Runtime, CPU"

    def __init__(self, model_dir: Path):
        import onnxruntime as ort

        path = Path(model_dir) / "onnx" / "model.onnx"
        if not path.is_file():
            raise DetectorError(f"{path} is missing")
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def run(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.session.run(None, {self.input_name: x})[0], np.float32)[0]


def _compiled_coreml(pkg: Path, key: str, cache_dir: Path) -> Path:
    """The package compiled to ``.mlmodelc``: once per package, then reused from ``cache_dir``."""
    import coremltools as ct

    dest = cache_dir / f"archival-{key}.mlmodelc"
    if dest.is_dir():
        return dest
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    tmp = Path(ct.utils.compile_model(str(pkg)))
    staged = dest.with_name(f"{dest.name}.tmp{os.getpid()}")
    shutil.move(str(tmp), str(staged))
    try:
        os.replace(staged, dest)
    except OSError:                       # another process finished first: use theirs
        shutil.rmtree(staged, ignore_errors=True)
    logger.info("Core ML: compiled the label detector in %.1f s", time.perf_counter() - t0)
    return dest


class CoremlBackend:
    """Core ML on the Apple GPU with the ``coreml_fp32`` package (macOS only)."""

    format = "coreml_fp32"
    label = "Core ML, Apple GPU"

    def __init__(self, model_dir: Path, cache_dir: Path):
        if sys.platform != "darwin":
            raise DetectorError("Core ML runs on macOS only")
        import coremltools as ct

        root = Path(model_dir) / "coreml_fp32"
        pkg = root / "model.mlpackage"
        try:
            contract = json.loads((root / "model.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DetectorError(f"cannot read {root / 'model.json'}: {exc}") from exc
        if contract.get("format") != "coreml_fp32" or str(contract.get("schema")) != "1":
            raise DetectorError(f"{root} is not a coreml_fp32 package this version can run")
        split = list(contract.get("split") or [])
        if contract.get("tail") != "end2end" or len(split) != 2 or split[0] != 4:
            raise DetectorError(f"{root} is not a YOLO26 end2end detector package")
        self.input_name = contract["input"]["name"]
        self.shapes = frozenset(tuple(int(d) for d in s) for s in contract["input"]["shapes"])
        self.outputs = tuple(contract["outputs"])
        self.max_det = int(contract.get("max_det") or 300)
        key = str(contract.get("mlmodel_sha256") or "")[:16] or "unknown"
        compiled = _compiled_coreml(pkg, key, Path(cache_dir))
        t0 = time.perf_counter()
        self.model = ct.models.CompiledMLModel(str(compiled), getattr(ct.ComputeUnit, COREML_UNITS))
        logger.info("Core ML: label detector loaded in %.1f s", time.perf_counter() - t0)

    def run(self, x: np.ndarray) -> np.ndarray:
        shape = tuple(int(d) for d in x.shape)
        if shape not in self.shapes:
            # Only an image with a side under ~2 px letterboxes to a shape the package lacks.
            logger.warning("Core ML: no input shape %s in the label detector; no labels found.", shape)
            return np.zeros((0, 6), np.float32)
        out = self.model.predict({self.input_name: x})
        boxes = np.asarray(out[self.outputs[0]], np.float32)
        scores = np.asarray(out[self.outputs[1]], np.float32)
        return end2end_tail(boxes, scores, self.max_det)[0]


# ------------------------------------------------------------------- detector --

class ArchivalDetector:
    """Image in, labels, barcodes, rulers (and the other archival classes) out."""

    def __init__(self, model_dir, fmt: str, *, classes, imgsz: int = 1280, conf: float = 0.25,
                 max_det: int = 100, cache_dir=None):
        if fmt not in FORMATS:
            raise DetectorError(f"unknown detector format {fmt!r}")
        self.classes = list(classes)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.max_det = int(max_det)
        if fmt == "coreml_fp32":
            self.backend = CoremlBackend(Path(model_dir), Path(cache_dir or Path(model_dir) / "compiled"))
        else:
            self.backend = OnnxBackend(Path(model_dir))
        self._lock = threading.Lock()

    @property
    def format(self) -> str:
        return self.backend.format

    @property
    def label(self) -> str:
        return self.backend.label

    def raw(self, img_bgr: np.ndarray) -> np.ndarray:
        """(n, 6) rows in original-image pixels: x1, y1, x2, y2, score, class index."""
        x = preprocess(img_bgr, self.imgsz)
        with self._lock:
            pred = self.backend.run(x)
        pred = end2end_filter(pred, self.conf, self.max_det).copy()
        scale_boxes(x.shape[2:], pred[:, :4], img_bgr.shape)
        return pred

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        out = []
        for x1, y1, x2, y2, score, cls in self.raw(img_bgr):
            index = int(cls)
            name = self.classes[index] if 0 <= index < len(self.classes) else f"class_{index}"
            out.append(Detection(name, float(score), (float(x1), float(y1), float(x2), float(y2))))
        return out
