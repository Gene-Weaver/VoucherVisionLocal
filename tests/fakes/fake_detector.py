"""A stand-in for textcollage.detector.ArchivalDetector: fixed boxes, no model files.

The app fixtures use it by default (VVL_DETECTOR_FACTORY), so tests that process
an image get a real label collage without the 223 MB download.
"""

from __future__ import annotations

from textcollage.detector import Detection


class FakeDetector:
    format = "onnx"
    label = "Fake detector"

    def __init__(self, detections=None):
        # detections: list[Detection], or a callable(image) -> list[Detection]
        self.detections = detections
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        if callable(self.detections):
            return list(self.detections(image))
        if self.detections is not None:
            return list(self.detections)
        h, w = image.shape[:2]
        return [
            Detection("Label", 0.95, (0.05 * w, 0.60 * h, 0.60 * w, 0.85 * h)),
            Detection("Barcode", 0.80, (0.65 * w, 0.60 * h, 0.95 * w, 0.72 * h)),
            Detection("Ruler", 0.90, (0.0, 0.92 * h, w, h)),
        ]
