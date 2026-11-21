"""GET /v1/detector, POST /v1/detector/install and POST /v1/detector/gpu/install:
the TextCollage label detector and its NVIDIA GPU runtime."""

from __future__ import annotations

from flask import Blueprint, jsonify

from vvl.api.common import runtime, vvl_state
from vvl.errors import ApiError

bp = Blueprint("vvl_detector", __name__)


def detector_payload() -> dict:
    payload = runtime().detector.state()
    payload.update(vvl_state()["processor"].collage_status())
    payload["gpu"] = runtime().detector_gpu.state()
    return payload


@bp.get("/v1/detector")
def detector_status():
    return jsonify(detector_payload())


@bp.post("/v1/detector/install")
def detector_install():
    """Start (or retry) the download. Returns the running job if one is already going."""
    store = runtime().detector
    if store.is_installed():
        raise ApiError("CONFLICT", "The label detector is already downloaded.", status=409)
    job = store.install()
    return jsonify({"job": job.to_dict()}), 202


@bp.post("/v1/detector/gpu/install")
def detector_gpu_install():
    """Start (or retry) the GPU runtime download. Returns the running job if one is already going."""
    store = runtime().detector_gpu
    state = store.state()
    if state["state"] in ("not_needed", "disabled"):
        raise ApiError("CONFLICT", state.get("message") or "No usable NVIDIA GPU.", status=409)
    if state["state"] == "installed":
        raise ApiError("CONFLICT", "The GPU runtime is already installed.", status=409)
    job = store.install()
    return jsonify({"job": job.to_dict()}), 202
