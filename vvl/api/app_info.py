"""GET /v1/app and POST /v1/shutdown."""

from __future__ import annotations

import logging
import os
import threading
import time

from flask import Blueprint, current_app, jsonify

import vvl
from vvl import paths, settings
from vvl.api.common import runtime, secret_store, settings_file
from vvl.enrichment.vvgo_client import WFO_UNAVAILABLE_REASON
from vvl.engines import API_UNAVAILABLE_REASON

logger = logging.getLogger(__name__)

bp = Blueprint("vvl_app", __name__)

FEATURES = {
    "demo_engine": True,
    "local_inference": True,
    "api_inference": False,
    "engine_install": True,
    "model_download": True,
    "enrichment_elevation": True,
    "enrichment_wfo": False,
}

FEATURE_NOTES = {
    "api_inference": API_UNAVAILABLE_REASON,
    "enrichment_wfo": WFO_UNAVAILABLE_REASON,
}

SHUTDOWN_DELAY_SECONDS = 0.25


@bp.get("/v1/app")
def app_info():
    current = settings.load(settings_file())
    return jsonify({
        "name": vvl.APP_NAME,
        "version": vvl.__version__,
        "phase": vvl.PHASE,
        "from_source": paths.from_source(),
        "paths": {
            "home": str(paths.home()),
            "models_dir": current["models_dir"],
            # The path only; the file is never opened for this route.
            "private_file": str(secret_store().path),
            "settings_file": str(settings_file()),
            "prompts_builtin": str(paths.prompts_builtin_dir()),
            "prompts_user": str(paths.prompts_user_dir()),
        },
        "features": dict(FEATURES),
        "feature_notes": dict(FEATURE_NOTES),
    })


@bp.post("/v1/shutdown")
def shutdown():
    exit_func = current_app.config.get("VVL_EXIT_FUNC") or os._exit
    rt = runtime()

    def _later():
        time.sleep(SHUTDOWN_DELAY_SECONDS)
        logger.info("Shutting down (requested by the client).")
        # os._exit skips atexit: stop the engine (frees graphics memory) first.
        try:
            rt.shutdown(timeout=15.0)
        except Exception:  # noqa: BLE001
            logger.exception("Engine shutdown failed.")
        exit_func(0)

    threading.Thread(target=_later, name="vvl-shutdown", daemon=True).start()
    return jsonify({"ok": True})
