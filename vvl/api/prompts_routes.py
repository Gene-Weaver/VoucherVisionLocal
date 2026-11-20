"""Prompt templates:

    GET    /v1/prompts
    GET    /v1/prompts/<source>/<name>      source = builtin | user
    POST   /v1/prompts/import               multipart `file`, or JSON {"name", "yaml_text"}
    DELETE /v1/prompts/user/<name>          (built-ins cannot be deleted)
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from vvl import prompts
from vvl.errors import ApiError

bp = Blueprint("vvl_prompts", __name__)


@bp.get("/v1/prompts")
def list_prompts():
    return jsonify({"prompts": prompts.list_all()})


@bp.get("/v1/prompts/<source>/<name>")
def prompt_detail(source, name):
    if source not in prompts.SOURCES:
        raise ApiError("NOT_FOUND", f"Unknown prompt source '{source}'. Use builtin or user.", status=404)
    return jsonify(prompts.detail(source, name))


@bp.post("/v1/prompts/import")
def import_prompt():
    if request.files:
        upload = request.files.get("file")
        if upload is None:
            raise ApiError("NO_FILE", 'Send the YAML as the multipart field "file".', status=400)
        name = (request.form.get("name") or upload.filename or "").strip()
        raw = upload.read(prompts.USER_PROMPTS_MAX_BYTES + 1)
    else:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ApiError(
                "INVALID_INPUT", 'Send multipart "file", or JSON {"name": "x.yaml", "yaml_text": "..."}.', status=400,
            )
        unknown = sorted(set(data) - {"name", "yaml_text"})
        if unknown:
            raise ApiError("INVALID_INPUT", f"Unknown key(s): {', '.join(unknown)}.", status=400)
        name = data.get("name")
        text = data.get("yaml_text")
        if not isinstance(name, str) or not isinstance(text, str):
            raise ApiError("INVALID_INPUT", "Both 'name' and 'yaml_text' must be strings.", status=400)
        name = name.strip()
        raw = text.encode("utf-8")
    summary = prompts.import_prompt(name, raw)
    return jsonify(summary), 201


@bp.delete("/v1/prompts/<source>/<name>")
def delete_prompt(source, name):
    if source == "builtin":
        raise ApiError("FORBIDDEN", "Built-in prompts cannot be deleted.", status=403)
    if source != "user":
        raise ApiError("NOT_FOUND", f"Unknown prompt source '{source}'.", status=404)
    prompts.delete_user_prompt(name)
    return jsonify({"ok": True, "name": name, "source": "user"})
