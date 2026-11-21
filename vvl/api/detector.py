"""GET /v1/detector and POST /v1/detector/install: the TextCollage label detector."""

from __future__ import annotations

from flask import Blueprint, jsonify

from vvl.api.common import runtime, vvl_state
from vvl.errors import ApiError

bp = Blueprint("vvl_detector", __name__)


def detector_payload() -> dict:
    payload = runtime().detector.state()
    payload.update(vvl_state()["processor"].collage_status())
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
