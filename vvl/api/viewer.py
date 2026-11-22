"""Viewer routes: browse saved run projects, inspect JSON records, decode embedded images.

    GET  /v1/viewer/projects                               list run folders in output_dir
    POST /v1/viewer/open                                   open a folder -> {project, stats, records}
    GET  /v1/viewer/projects/<id>/records/<file>           lightweight record JSON (base64 stripped)
    GET  /v1/viewer/projects/<id>/images/<file>/<kind>     independent-process decoded image bytes
"""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, Response, jsonify

from vvl import outputs, settings, viewer
from vvl.api.common import json_body, settings_file, vvl_state
from vvl.errors import ApiError

bp = Blueprint("vvl_viewer", __name__)


def _store() -> viewer.ViewerStore:
    return vvl_state()["viewer"]


def _api_error(exc: viewer.ViewerError) -> ApiError:
    return ApiError(exc.code, str(exc), status=exc.status)


@bp.get("/v1/viewer/projects")
def list_projects():
    base_dir = settings.load(settings_file())["output_dir"]
    folders = outputs.list_existing_folders(base_dir)
    projects = [f for f in folders if f["json_count"] > 0]
    return jsonify({
        "output_dir": base_dir,
        "projects": projects,
    })


@bp.post("/v1/viewer/open")
def open_project():
    body = json_body(required=True)
    raw_path = body.get("path")
    folder = body.get("folder")
    if raw_path:
        target = raw_path
    elif isinstance(folder, str) and folder.strip():
        try:
            clean = outputs.validate_folder_name(folder)
        except outputs.OutputError as exc:
            raise ApiError(exc.code, str(exc), status=exc.status) from None
        target = str(Path(settings.load(settings_file())["output_dir"]) / clean)
    else:
        raise ApiError("INVALID_INPUT", "Provide 'path' or 'folder'.", status=400)
    try:
        result = _store().open_project(target)
    except viewer.ViewerError as exc:
        raise _api_error(exc) from None
    return jsonify(result)


@bp.get("/v1/viewer/projects/<project_id>/records/<filename>")
def get_record(project_id, filename):
    try:
        data = _store().get_record(project_id, filename)
    except viewer.ViewerError as exc:
        raise _api_error(exc) from None
    return jsonify(data)


@bp.get("/v1/viewer/projects/<project_id>/images/<filename>/<kind>")
def get_image(project_id, filename, kind):
    try:
        raw_bytes, mime = _store().decode_image(project_id, filename, kind)
    except viewer.ViewerError as exc:
        raise _api_error(exc) from None
    return Response(raw_bytes, status=200, mimetype=mime, headers={"Cache-Control": "private, max-age=300"})
