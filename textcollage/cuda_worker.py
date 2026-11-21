# Part of VoucherVisionLocal, AGPL-3.0 (see textcollage/detector.py).
"""The label detector on an NVIDIA GPU, in a process of its own.

Started by textcollage.detector.CudaWorkerBackend as ``python -m textcollage.cuda_worker``
with the GPU runtime folder first on PYTHONPATH, so ``import onnxruntime`` finds
onnxruntime-gpu there and not the app's CPU build. The two cannot share one process:
both are the ``onnxruntime`` package.

Protocol on stdin/stdout, one frame per message: a type byte, an 8-byte little-endian
length, then the payload. Type ``J`` is UTF-8 JSON, type ``A`` is one array in
NumPy's .npy format (no pickles).

    parent -> J {"model": path, "device_id": n, "runtime_dir": path}
    worker -> J {"ok": true, "onnxruntime": version, "file": path, "providers": [...]}
              or J {"ok": false, "error": "..."} and exit
    parent -> A input tensor (1, 3, H, W) float32
    worker -> A network output, or J {"ok": false, "error": "..."}

The worker ends when stdin closes, so it never outlives the app. Anything printed
by the libraries goes to stderr, never into the protocol.
"""

from __future__ import annotations

import io
import json
import os
import struct
import sys

HEADER = struct.Struct("<cQ")
MAX_FRAME = 512 * 1024 * 1024


def read_frame(stream):
    head = stream.read(HEADER.size)
    if not head or len(head) < HEADER.size:
        return None, None
    kind, length = HEADER.unpack(head)
    if length > MAX_FRAME:
        raise ValueError(f"frame of {length} bytes is too large")
    payload = b""
    while len(payload) < length:
        chunk = stream.read(length - len(payload))
        if not chunk:
            return None, None
        payload += chunk
    return kind, payload


def write_frame(stream, kind: bytes, payload: bytes) -> None:
    stream.write(HEADER.pack(kind, len(payload)))
    stream.write(payload)
    stream.flush()


def write_json(stream, obj) -> None:
    write_frame(stream, b"J", json.dumps(obj).encode("utf-8"))


def array_bytes(array) -> bytes:
    import numpy as np

    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(array), allow_pickle=False)
    return buf.getvalue()


def bytes_array(payload: bytes):
    import numpy as np

    return np.load(io.BytesIO(payload), allow_pickle=False)


# CUDA provider options: grow the memory pool only by what is asked for (2.2 GB measured
# instead of 2.5), so the GPU keeps room for a local model's engine.
CUDA_OPTIONS = {"arena_extend_strategy": "kSameAsRequested"}
WARMUP_SHAPE = (1, 3, 1280, 1280)


def main() -> int:
    # The protocol keeps the real stdout; fd 1 and sys.stdout point at stderr from here on.
    proto = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    inp = sys.stdin.buffer

    kind, payload = read_frame(inp)
    if kind != b"J":
        return 2
    cfg = json.loads(payload)
    try:
        import numpy as np
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            # Loads CUDA and cuDNN from the NVIDIA wheels next to onnxruntime (no LD_LIBRARY_PATH).
            ort.preload_dlls()
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(f"this onnxruntime ({ort.__version__}) has no CUDA provider")
        options = ort.SessionOptions()
        options.log_severity_level = 3
        provider = ("CUDAExecutionProvider", {"device_id": int(cfg.get("device_id") or 0), **CUDA_OPTIONS})
        session = ort.InferenceSession(cfg["model"], options, providers=[provider])
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("onnxruntime could not start the CUDA provider")
        input_name = session.get_inputs()[0].name
        # Self-test: one pass at the largest input shape, so a cuDNN or memory problem shows now.
        session.run(None, {input_name: np.zeros(WARMUP_SHAPE, np.float32)})
    except Exception as exc:  # noqa: BLE001 - every failure goes back to the parent
        write_json(proto, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    write_json(proto, {"ok": True, "onnxruntime": getattr(ort, "__version__", "?"),
                       "file": getattr(ort, "__file__", None), "providers": session.get_providers()})

    while True:
        kind, payload = read_frame(inp)
        if kind is None:
            return 0
        if kind != b"A":
            write_json(proto, {"ok": False, "error": f"unexpected frame {kind!r}"})
            continue
        try:
            x = bytes_array(payload)
            out = session.run(None, {input_name: x})[0]
            write_frame(proto, b"A", array_bytes(np.asarray(out, np.float32)))
        except Exception as exc:  # noqa: BLE001
            write_json(proto, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    sys.exit(main())
