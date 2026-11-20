"""Providers and keys:

    GET    /v1/providers
    PUT    /v1/keys/<provider_id>          {"fields": {"api_key": "..."}}  ("" removes)
    DELETE /v1/keys/<provider_id>
    POST   /v1/keys/<provider_id>/test
    POST   /v1/providers/<provider_id>/models/refresh
"""

from __future__ import annotations

from flask import Blueprint, jsonify

from vvl import providers as provider_service
from vvl.api.common import find_provider, json_body, provider_state, providers_registry, secret_store
from vvl.catalog import api_provider_entry
from vvl.errors import ApiError

bp = Blueprint("vvl_keys", __name__)


@bp.get("/v1/providers")
def list_providers():
    store = secret_store()
    state = provider_state()
    items = [
        provider_service.provider_object(spec, store, state)
        for spec in providers_registry().providers
        if spec.enabled
    ]
    return jsonify({"providers": items})


@bp.put("/v1/keys/<provider_id>")
def save_keys(provider_id):
    spec = find_provider(provider_id)
    body = json_body()
    unknown = sorted(set(body) - {"fields"})
    if unknown:
        raise ApiError("INVALID_INPUT", f"Unknown key(s) in the request body: {', '.join(unknown)}.", status=400)
    fields = body.get("fields")
    if not isinstance(fields, dict):
        raise ApiError("INVALID_INPUT", 'Send {"fields": {"<field name>": "<value>"}}.', status=400)
    store = secret_store()
    store.set_fields(spec.id, fields, spec.field_names())
    state = provider_state()
    state.clear_provider(spec.id)
    return jsonify(provider_service.provider_object(spec, store, state))


@bp.delete("/v1/keys/<provider_id>")
def delete_keys(provider_id):
    spec = find_provider(provider_id)
    store = secret_store()
    store.delete_provider(spec.id)
    state = provider_state()
    state.clear_provider(spec.id)
    return jsonify(provider_service.provider_object(spec, store, state))


@bp.post("/v1/keys/<provider_id>/test")
def test_key(provider_id):
    spec = find_provider(provider_id)
    result = provider_service.run_key_test(spec, secret_store(), provider_state())
    return jsonify(result)


@bp.post("/v1/providers/<provider_id>/models/refresh")
def refresh_models(provider_id):
    spec = find_provider(provider_id)
    if spec.role != "model_provider" or spec.models.source != "list_endpoint":
        raise ApiError("INVALID_INPUT", f"{spec.label} has no model list.", status=400)
    store = secret_store()
    state = provider_state()
    provider_service.refresh_models(spec, store, state)
    return jsonify(api_provider_entry(spec, store, state))
