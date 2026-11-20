"""GET /v1/metrics: live processor, memory and graphics-card use for the header monitor.

Cheap enough to poll every two seconds: hardware readings are cached for about
1.5 s (`vvl.system.live`), the system report is only read from its cache, and
nothing here waits on a slow probe. Whatever cannot be measured is `null`; the
route itself does not fail because a reading is missing.
"""

from __future__ import annotations

import logging
import threading
import time

from flask import Blueprint, current_app, jsonify

from vvl.api.common import local_registry, runtime, vvl_state
from vvl.llama.plan import MODE_NOTES, memory_plan
from vvl.system import live

logger = logging.getLogger(__name__)

bp = Blueprint("vvl_metrics", __name__)

REPORT_MAX_AGE_S = 30.0
REPORT_FIRST_WAIT_S = 0.3


class ReportHolder:
    """The system report, fetched off the request thread.

    The first report can take a few seconds to build; a metrics request waits
    only `REPORT_FIRST_WAIT_S` for it and answers without it until it is ready.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._value: dict | None = None
        self._at = 0.0
        self._thread: threading.Thread | None = None

    def _fetch(self, rt) -> None:
        try:
            value = rt.system_report(False)
        except Exception:  # noqa: BLE001
            value = None
        with self._lock:
            if isinstance(value, dict):
                self._value, self._at = value, time.monotonic()
            self._thread = None

    def get(self, rt) -> dict | None:
        with self._lock:
            stale = self._value is None or time.monotonic() - self._at > REPORT_MAX_AGE_S
            thread = self._thread
            if stale and thread is None:
                thread = self._thread = threading.Thread(target=self._fetch, args=(rt,), name="vvl-metrics-report",
                                                         daemon=True)
                thread.start()
            waiting = self._value is None
        if waiting and thread is not None:
            thread.join(REPORT_FIRST_WAIT_S)
        with self._lock:
            return self._value


def _holder() -> ReportHolder:
    state = vvl_state()
    holder = state.get("metrics_report")
    if holder is None:
        holder = state["metrics_report"] = ReportHolder()
    return holder


def metrics_payload() -> dict:
    rt = runtime()
    sampler = current_app.config.get("VVL_LIVE_SAMPLER") or live.default_sampler()
    sample = sampler.sample()
    report = _holder().get(rt)

    try:
        status = rt.supervisor.status()
    except Exception:  # noqa: BLE001
        logger.exception("Engine status failed.")
        status = {}
    try:
        pids = list(rt.supervisor.engine_pids())
    except Exception:  # noqa: BLE001
        pids = []
    gpus = live.build_gpus(report, sample, engine_pids=pids, engine_gpu_devices=status.get("gpu_devices"))

    try:
        plan = memory_plan(rt.supervisor, rt.settings().get("selection") or {}, local_registry(), report, gpus)
    except Exception:  # noqa: BLE001 - the monitor must keep working without a plan
        logger.exception("The memory plan could not be computed.")
        plan = {"mode": "none", "note": MODE_NOTES["none"], "models": [], "gpus": {}}
    per_card = plan.pop("gpus", {})
    for g in gpus:
        g.update(per_card.get(g["index"]) or {"planned_bytes": 0, "planned_for": [], "planned_fits": True})

    return {
        "ts": sample.get("ts"),
        "cpu": sample.get("cpu"),
        "memory": sample.get("memory"),
        "gpus": gpus,
        "engine": {"state": status.get("state"), "loaded_model": status.get("loaded_model"),
                   "gpu_devices": status.get("gpu_devices")},
        "plan": plan,
    }


@bp.get("/v1/metrics")
def metrics():
    return jsonify(metrics_payload())
