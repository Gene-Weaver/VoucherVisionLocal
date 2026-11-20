"""The managed llama.cpp engine (docs/API.md section 9, "Engine")."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from vvl.api.common import json_body, runtime
from vvl.errors import ApiError
from vvl.jobs import ACTIVE_STATES

bp = Blueprint("vvl_engine", __name__)

LOG_LIMIT_DEFAULT = 200
LOG_LIMIT_MAX = 2000

NO_BUILD_REASON = "No llama.cpp build is published for this computer's operating system and processor type."
NO_COMPATIBLE_REASON = "None of the llama.cpp builds can run on this computer."


def plan_summary(rt) -> dict:
    """What the engine pool will run for the selected local models (no engine is started).

    `staged` tells a batch to run in two passes (all OCR, then all parsing):
    always when the models cannot be loaded together, or when the user asked.
    `concurrency` is how many images a batch should send at once.
    """
    from vvl.llama.pool import describe_layout

    try:
        layout = rt.supervisor.plan()
    except Exception:  # noqa: BLE001 - the Engine tab must load without a plan
        layout = None
    labels = {}
    try:
        for family in rt.local_registry().families:
            for build in family.builds:
                labels[f"local:{build.id}"] = build.label
    except Exception:  # noqa: BLE001
        pass
    mode = layout.mode if layout is not None else "none"
    slots = {m: layout.slots_for(m) for m in layout.models()} if layout is not None else {}
    wants_staged = bool(rt.engine_settings().get("staged_batches"))
    return {
        "mode": mode,
        "staged": mode == "exclusive" or (wants_staged and mode != "none"),
        "summary": describe_layout(layout, labels),
        "concurrency": max(slots.values()) if slots else None,
        "fits": bool(layout.fits) if layout is not None else True,
    }


def engine_payload() -> dict:
    rt = runtime()
    assessed = rt.installer.assess()
    recommended = rt.installer.recommended(assessed)
    job = rt.jobs.active_job("engine_install")
    available = any(v["compatible"] for v in assessed) or any(v["installed"] for v in assessed)
    reason = None
    if not available:
        reason = NO_BUILD_REASON if not assessed else NO_COMPATIBLE_REASON
    return {
        "available": available,
        "unavailable_reason": reason,
        "models_dir": rt.models_dir(),
        "managed": rt.supervisor.status(),
        "recommended_variant": recommended,
        "variants": assessed,
        "install_job_id": job.id if job is not None and job.state in ACTIVE_STATES else None,
        "settings": rt.engine_settings(),
        "plan": plan_summary(rt),
    }


@bp.get("/v1/engine")
def engine_status():
    return jsonify(engine_payload())


@bp.post("/v1/engine/install")
def engine_install():
    body = json_body(required=False)
    unknown = sorted(set(body) - {"variant"})
    if unknown:
        raise ApiError("INVALID_INPUT", "Unknown fields.", status=400, details={"unknown_keys": unknown})
    variant = body.get("variant")
    if variant is not None and not isinstance(variant, str):
        raise ApiError("INVALID_INPUT", "variant must be a build id.", status=400)
    job = runtime().installer.start_install(variant)
    return jsonify({"job": job.to_dict()}), 202


@bp.post("/v1/engine/uninstall")
def engine_uninstall():
    body = json_body(required=False)
    variant = body.get("variant")
    if variant is not None and not isinstance(variant, str):
        raise ApiError("INVALID_INPUT", "variant must be a build id.", status=400)
    runtime().installer.uninstall(variant)
    return jsonify(engine_payload())


@bp.post("/v1/engine/start")
def engine_start():
    body = json_body(required=False)
    model_id = body.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ApiError("INVALID_INPUT", "Send model_id, e.g. {\"model_id\": \"local:qwen3-vl-8b-q8\"}.", status=400)
    runtime().supervisor.start(model_id.strip())
    return jsonify(engine_payload()), 202


@bp.post("/v1/engine/stop")
def engine_stop():
    runtime().supervisor.stop()
    return jsonify(engine_payload())


@bp.get("/v1/engine/logs")
def engine_logs():
    def _int_arg(name, default, low, high):
        raw = request.args.get(name)
        if raw in (None, ""):
            return default
        try:
            value = int(raw)
        except ValueError:
            raise ApiError("INVALID_INPUT", f"{name} must be a number.", status=400) from None
        return max(low, min(high, value))

    after = _int_arg("after", None, 0, 2 ** 62)
    limit = _int_arg("limit", LOG_LIMIT_DEFAULT, 1, LOG_LIMIT_MAX)
    return jsonify(runtime().supervisor.logs.read(after, limit))
