"""GET /v1/system[?refresh=1]: the system report from vvl.system.

vvl.system is written separately and imported lazily. When it is missing or
raises, a minimal report in the same shape is returned so the UI always has
something to show.
"""

from __future__ import annotations

import datetime
import logging
import platform

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

bp = Blueprint("vvl_system", __name__)


def fallback_report(reason: str) -> dict:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        os_info = {
            "system": platform.system() or None,
            "name": None,
            "version": platform.version() or None,
            "kernel": platform.release() or None,
            "arch": platform.machine() or None,
            "is_wsl": None,
            "is_apple_silicon": None,
            "python": platform.python_version(),
        }
    except Exception:
        os_info = None
    return {
        "schema": 1,
        "generated_at": now,
        "duration_ms": 0,
        "os": os_info,
        "cpu": None,
        "memory": None,
        "gpus": [],
        "accelerators": {},
        "disk": None,
        "capability": None,
        "warnings": [{
            "code": "SYSTEM_CHECK_UNAVAILABLE",
            "message": "The system check could not run, so hardware details are missing.",
        }],
        "probes": [{"name": "vvl.system", "ok": False, "duration_ms": 0, "error": reason}],
    }


def get_system_report(refresh: bool = False) -> dict:
    try:
        from vvl import system as vvl_system

        get_report = getattr(vvl_system, "get_report", None)
        if get_report is None:
            return fallback_report("vvl.system.get_report is not available.")
        report = get_report(refresh=refresh)
        if not isinstance(report, dict):
            return fallback_report("vvl.system.get_report did not return a report.")
        return report
    except Exception as exc:
        logger.warning("System report failed: %s", type(exc).__name__)
        return fallback_report(f"The system check failed ({type(exc).__name__}).")


@bp.get("/v1/system")
def system_report():
    refresh = request.args.get("refresh", "").strip().lower() in ("1", "true", "yes")
    return jsonify(get_system_report(refresh=refresh))


__all__ = ["bp", "fallback_report", "get_system_report"]
