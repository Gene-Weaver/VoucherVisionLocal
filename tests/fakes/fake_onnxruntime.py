"""A stand-in for onnxruntime-gpu in the GPU worker tests (copied in as onnxruntime/__init__.py).

It answers like the CUDA provider without a GPU: every run returns 300 rows, the first
(1, 2, W, H, 0.9, 3) for the input's width W and height H, the rest zeros.
FAKE_ORT_NO_CUDA=1: no CUDA provider. FAKE_ORT_CRASH=1: the process dies on the first
run after the warm-up.
"""

import os

import numpy as np

__version__ = "9.9.0-fake"


def preload_dlls(*args, **kwargs):
    print("fake preload_dlls: this line must go to stderr, not into the protocol")


def get_available_providers():
    if os.environ.get("FAKE_ORT_NO_CUDA") == "1":
        return ["CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


class SessionOptions:
    log_severity_level = 2


class _Input:
    name = "images"


class InferenceSession:
    def __init__(self, path, options=None, providers=None):
        self.providers = providers
        self.runs = 0

    def get_providers(self):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def get_inputs(self):
        return [_Input()]

    def run(self, outputs, feeds):
        x = feeds["images"]
        self.runs += 1
        if os.environ.get("FAKE_ORT_CRASH") == "1" and self.runs > 1:
            os._exit(3)
        out = np.zeros((1, 300, 6), np.float32)
        out[0, 0] = (1.0, 2.0, float(x.shape[2]), float(x.shape[3]), 0.9, 3.0)
        return [out]
