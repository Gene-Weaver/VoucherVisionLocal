"""Prompt templates:

    GET    /v1/prompts
    GET    /v1/prompts/<source>/<name>      source = builtin | user
    POST   /v1/prompts/import               multipart `file`, or JSON {"name", "yaml_text"}
    POST   /v1/prompts/user                 JSON {"name", "prompt"} — create from Designer
    PUT    /v1/prompts/<source>/<name>      JSON {"prompt"} — update custom prompt from Designer
    POST   /v1/prompts/render               JSON {"prompt"} — validate & render YAML
    DELETE /v1/prompts/user/<name>          (built-ins cannot be deleted)
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from vvl import paths, prompts
from vvl.errors import ApiError

bp = Blueprint("vvl_prompts", __name__)


@bp.get("/v1/prompts")
def list_prompts():
    return jsonify({"prompts": prompts.list_all(), "user_dir": str(paths.prompts_user_dir())})


@bp.get("/v1/prompts/<source>/<name>")
def prompt_detail(source, name):
    if source not in prompts.SOURCES:
        raise ApiError("NOT_FOUND", f"Unknown prompt source '{source}'. Use builtin or user.", status=404)
    return jsonify(prompts.detail(source, name))


@bp.post("/v1/prompts/user")
def create_user_prompt():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ApiError("INVALID_INPUT", 'Send JSON {"name": "My_prompt.yaml", "prompt": {...}}.', status=400)
    unknown = sorted(set(data) - {"name", "prompt"})
    if unknown:
        raise ApiError("INVALID_INPUT", f"Unknown key(s): {', '.join(unknown)}.", status=400)
    name = data.get("name")
    prompt_obj = data.get("prompt")
    if not isinstance(name, str) or not isinstance(prompt_obj, dict):
        raise ApiError("INVALID_INPUT", "'name' must be a string and 'prompt' must be an object.", status=400)
    created = prompts.save_user_prompt(name.strip(), prompt_obj, overwrite=False)
    return jsonify(created), 201


@bp.put("/v1/prompts/<source>/<name>")
def update_prompt(source, name):
    if source == "builtin":
        raise ApiError("FORBIDDEN", "Built-in prompts cannot be edited. Make a copy instead.", status=403)
    if source != "user":
        raise ApiError("NOT_FOUND", f"Unknown prompt source '{source}'.", status=404)
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("prompt"), dict):
        raise ApiError("INVALID_INPUT", 'Send JSON {"prompt": {...}}.', status=400)
    unknown = sorted(set(data) - {"prompt"})
    if unknown:
        raise ApiError("INVALID_INPUT", f"Unknown key(s): {', '.join(unknown)}.", status=400)
    updated = prompts.save_user_prompt(name, data["prompt"], overwrite=True)
    return jsonify(updated)


@bp.post("/v1/prompts/render")
def render_prompt():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("prompt"), dict):
        raise ApiError("INVALID_INPUT", 'Send JSON {"prompt": {...}}.', status=400)
    try:
        _, yaml_text = prompts.render_designer_yaml(data["prompt"])
    except prompts.UserPromptValidationError as exc:
        raise ApiError("INVALID_INPUT", str(exc), status=400) from None
    return jsonify({"valid": True, "yaml_text": yaml_text})


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
