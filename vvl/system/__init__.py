"""System performance checker for VoucherVisionLocal.

Public API (see docs/API.md, ``GET /v1/system``):

- ``get_report(refresh=False, models_dir=None) -> dict``: the hardware report,
  cached in-process and thread-safe; ``refresh=True`` re-probes.
- ``fit.assess(report, build) -> dict``: whether one model build will run.

Run ``python -m vvl.system`` for the JSON report, ``--summary`` for a short
human-readable version, ``--fit`` for verdicts on every registry build.
"""

from __future__ import annotations

from . import fit
from .report import SCHEMA_VERSION, clear_cache, get_report

__all__ = ["SCHEMA_VERSION", "clear_cache", "fit", "get_report"]
