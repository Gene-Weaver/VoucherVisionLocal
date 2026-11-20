"""GET /v1/generation and PUT /v1/generation: per-step model settings on the VVL tab.

Reasoning, reasoning strength, output length, image detail (per step) and the context window (per
model), for the models chosen in the saved selection. Each model lists only the
controls it supports, and `sent` shows the literal values the current choices
send (docs/API.md section 13).
"""

from __future__ import annotations

from flask import Blueprint, jsonify

from vvl import generation, settings
from vvl.api.common import json_body, local_registry, runtime, settings_file
from vvl.errors import ApiError

bp = Blueprint("vvl_generation", __name__)


def _label(registry, model_id: str) -> str:
    found = registry.find_build(model_id.split(":", 1)[1]) if model_id.startswith("local:") else None
    return found[1].label if found else model_id


def _context_entry(registry, rt, current: dict, model_id: str) -> dict:
    found = registry.find_build(model_id.split(":", 1)[1])
    entry = {"model_id": model_id, "label": _label(registry, model_id),
             "steps": generation.steps_using(model_id, current),
             "min": generation.CONTEXT_MIN, "max": None, "value": "auto", "mode": None,
             "effective": None, "needed": None}
    if not found:
        return entry
    family, build = found
    chosen = generation.choices(current)["context"].get(model_id)
    entry["value"] = chosen if isinstance(chosen, int) else "auto"
    try:
        params = rt.supervisor.launch_params(model_id, family, build)
    except Exception:  # noqa: BLE001 - the controls must still show
        params = None
    if params:
        entry.update({"max": params.get("model_max"), "mode": params["rule"]["mode"],
                      "needed": params["rule"]["needed"], "effective": params["context_size"]})
    return entry


def payload() -> dict:
    rt = runtime()
    registry = local_registry()
    current = settings.load(settings_file())
    sel = current.get("selection") or {}
    steps = {"ocr": [m for m in (sel.get("ocr") or []) if isinstance(m, str) and m.startswith("local:")],
             "parse": [m for m in [sel.get("parse")] if isinstance(m, str) and m.startswith("local:")]}
    chosen = generation.choices(current)
    out = {}
    for step, model_ids in steps.items():
        models = []
        for model_id in model_ids:
            info = generation.describe_step(model_id, step, current, registry)
            models.append({"model_id": model_id, "label": _label(registry, model_id), **info})
        applicable = {key: any(key in m["controls"] for m in models)
                      for key in ("reasoning", "strength", "output_length", "image_detail")}
        out[step] = {"models": models, "values": chosen[step], "applicable": applicable}
    context_ids = []
    for model_id in steps["ocr"] + steps["parse"]:
        if model_id not in context_ids:
            context_ids.append(model_id)
    out["context"] = {"min": generation.CONTEXT_MIN,
                      "models": [_context_entry(registry, rt, current, m) for m in context_ids]}
    return out


@bp.get("/v1/generation")
def get_generation():
    return jsonify(payload())


@bp.put("/v1/generation")
def put_generation():
    body = json_body()
    unknown = sorted(set(body) - {"ocr", "parse", "context"})
    if unknown:
        raise ApiError("INVALID_INPUT", "Unknown fields.", status=400, details={"unknown_keys": unknown})
    current = settings.load(settings_file())
    update = {}
    for step in ("ocr", "parse"):
        if step in body:
            if not isinstance(body[step], dict):
                raise ApiError("INVALID_INPUT", f"{step} must be an object.", status=400)
            merged = dict(current["generation"][step])
            merged.update(body[step])
            update[step] = merged
    if "context" in body:
        if not isinstance(body["context"], dict):
            raise ApiError("INVALID_INPUT", "context must be an object of model id -> tokens or \"auto\".",
                           status=400)
        merged = dict(current["generation"]["context"])
        for model_id, tokens in body["context"].items():
            if tokens in (None, "auto"):
                merged.pop(model_id, None)
            else:
                merged[model_id] = tokens
        update["context"] = merged
    changes = settings.validate_update({"generation": update})
    settings.save(changes, settings_file())
    return jsonify(payload())
