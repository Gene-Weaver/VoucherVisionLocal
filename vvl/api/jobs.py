"""GET /v1/jobs and GET /v1/jobs/<id> (docs/API.md section 9, "Jobs")."""

from __future__ import annotations

from flask import Blueprint, jsonify

from vvl.api.common import runtime
from vvl.errors import ApiError

bp = Blueprint("vvl_jobs", __name__)


@bp.get("/v1/jobs")
def list_jobs():
    return jsonify({"jobs": [job.to_dict() for job in runtime().jobs.list()]})


@bp.get("/v1/jobs/<job_id>")
def get_job(job_id):
    job = runtime().jobs.get(job_id)
    if job is None:
        raise ApiError("NOT_FOUND", f"No job '{job_id}'.", status=404)
    return jsonify(job.to_dict())
