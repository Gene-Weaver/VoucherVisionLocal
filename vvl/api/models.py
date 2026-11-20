"""GET /v1/models, PUT /v1/selection, and local model downloads (docs/API.md section 9)."""

from __future__ import annotations

from flask import Blueprint, jsonify

from vvl import settings
from vvl.api.common import (
    json_body,
    local_registry,
    provider_state,
    providers_registry,
    runtime,
    secret_store,
    settings_file,
)
from vvl.catalog import build_catalog, validate_model_ids
from vvl.errors import ApiError

bp = Blueprint("vvl_models", __name__)


def id_validator():
    """Callable for settings.validate_update: raises ModelIdError for unknown ids."""
    providers = providers_registry()
    local_models = local_registry()
    state = provider_state()
    rt = runtime()

    def _validate(model_ids, role):
        validate_model_ids(model_ids, role, providers=providers, local_models=local_models, state=state, runtime=rt)

    return _validate


@bp.get("/v1/models")
def list_models():
    current = settings.load(settings_file())
    catalog = build_catalog(
        providers=providers_registry(),
        local_models=local_registry(),
        store=secret_store(),
        state=provider_state(),
        selection=current["selection"],
        runtime=runtime(),
        models_dir=current["models_dir"],
        settings=current,
    )
    return jsonify(catalog)


@bp.put("/v1/selection")
def put_selection():
    body = json_body()
    changes = settings.validate_update({"selection": body}, id_validator())
    saved = settings.save(changes, settings_file())
    return jsonify({"selection": saved["selection"]})


@bp.post("/v1/models/local/<build_id>/download")
def download_local_model(build_id):
    body = json_body(required=False)
    unknown = sorted(set(body) - {"confirm_size_bytes"})
    if unknown:
        raise ApiError("INVALID_INPUT", "Unknown fields.", status=400, details={"unknown_keys": unknown})
    job = runtime().models.start_download(build_id, body.get("confirm_size_bytes"))
    return jsonify({"job": job.to_dict()}), 202


@bp.post("/v1/models/local/<build_id>/download/cancel")
def cancel_local_download(build_id):
    job = runtime().models.cancel_download(build_id)
    return jsonify({"job": job.to_dict()})


@bp.delete("/v1/models/local/<build_id>")
def delete_local_model(build_id):
    freed = runtime().models.delete(build_id)
    return jsonify({"ok": True, "freed_bytes": freed})
