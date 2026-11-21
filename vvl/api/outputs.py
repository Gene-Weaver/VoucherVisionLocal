"""Where results are saved (vvl/outputs.py): the folder, and one results folder per run.

    GET  /v1/output                       the folder runs are saved in
    POST /v1/output/batches               start a run's results folder -> its id
    POST /v1/output/batches/<id>/finish   write the spreadsheet a last time -> what was saved

A run sends the id as `output_batch` with /process, /process-url and
/v1/process/finish. Without it nothing is saved. The folder itself is the
`output_dir` setting (PUT /v1/settings).
"""

from __future__ import annotations

import os

from flask import Blueprint, jsonify

from vvl import outputs, paths, settings
from vvl.api.common import json_body, settings_file, vvl_state
from vvl.errors import ApiError

bp = Blueprint("vvl_outputs", __name__)


def _store() -> outputs.OutputStore:
    return vvl_state()["outputs"]


def _api_error(exc: outputs.OutputError) -> ApiError:
    return ApiError(exc.code, str(exc), status=exc.status)


@bp.get("/v1/output")
def get_output():
    current = settings.load(settings_file())["output_dir"]
    default = str(outputs.default_dir())
    return jsonify({
        "dir": current,
        "default_dir": default,
        "is_default": os.path.normpath(current) == os.path.normpath(default),
        "exists": os.path.isdir(current),
        "synced_folder": paths.synced_folder_hint(current),
        "subfolders": list(outputs.SUBDIRS),
        "stamp_format": "YYYYMMDD-HH-MM-SS",
    })


@bp.post("/v1/output/batches")
def create_batch():
    body = json_body(required=False)
    append = body.get("append", False)
    if not isinstance(append, bool):
        raise ApiError("INVALID_INPUT", "append must be true or false.", status=400)
    try:
        batch = _store().create(settings.load(settings_file())["output_dir"], name=body.get("name"), append=append)
    except outputs.OutputError as exc:
        raise _api_error(exc) from None
    return jsonify(batch.summary()), 201


@bp.post("/v1/output/batches/<batch_id>/finish")
def finish_batch(batch_id):
    try:
        batch = _store().get(batch_id)
    except outputs.OutputError as exc:
        raise _api_error(exc) from None
    return jsonify(batch.finish())
