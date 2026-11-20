"""GET /v1/settings and PUT /v1/settings (partial, allowlisted keys only)."""

from __future__ import annotations

from flask import Blueprint, jsonify

from vvl import settings
from vvl.api.common import json_body, settings_file
from vvl.api.models import id_validator

bp = Blueprint("vvl_settings", __name__)


def _public(current: dict) -> dict:
    return {key: current[key] for key in settings.ALLOWED_KEYS}


@bp.get("/v1/settings")
def get_settings():
    return jsonify(_public(settings.load(settings_file())))


@bp.put("/v1/settings")
def put_settings():
    body = json_body()
    changes = settings.validate_update(body, id_validator())
    saved = settings.save(changes, settings_file())
    payload = _public(saved)
    payload["warnings"] = settings.warnings_for(saved, changes)
    return jsonify(payload)
