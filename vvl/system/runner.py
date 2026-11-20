"""Run probes in parallel, each with its own timeout and exception guard.

A probe is a function ``probe(ctx, log) -> value``. It may raise, hang, or
report a soft problem through ``log.fail(message)``; in every case the runner
returns a ``ProbeOutcome`` and the report is still produced. A probe that hangs
is abandoned (its daemon thread keeps running in the background and its result
is discarded); any subprocess it started is killed by ``_proc.run``'s own
timeout.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import _proc
from ._util import system_name, truncate


@dataclass
class ProbeContext:
    """What probes may read. Tests replace parts of it."""

    system: str = field(default_factory=system_name)
    env: dict = field(default_factory=lambda: dict(os.environ))
    run: Callable = _proc.run
    sys_root: str = "/"
    models_dir: str | None = None
    models_dir_source: str | None = None


@dataclass
class ProbeLog:
    errors: list = field(default_factory=list)

    def fail(self, message):
        """Record a problem that makes this probe's result incomplete."""
        if message:
            self.errors.append(str(message))


@dataclass
class ProbeSpec:
    name: str
    func: Callable[[ProbeContext, ProbeLog], Any]
    timeout_s: float = 6.0


@dataclass
class ProbeOutcome:
    name: str
    ok: bool
    duration_ms: int
    error: str | None
    value: Any = None

    def record(self) -> dict:
        return {
            "name": self.name,
            "ok": bool(self.ok),
            "duration_ms": int(self.duration_ms),
            "error": self.error,
        }


def _target(spec, ctx, box):
    log = ProbeLog()
    start = time.perf_counter()
    try:
        box["value"] = spec.func(ctx, log)
    except BaseException as exc:  # noqa: BLE001 - a probe must never break the report
        box["exc"] = f"{type(exc).__name__}: {exc}"
    finally:
        box["log"] = log
        box["duration_ms"] = int((time.perf_counter() - start) * 1000)
        box["done"] = True


def run_probes(specs, ctx) -> dict:
    """Start every probe at once; collect each within its own timeout."""
    started = []
    for spec in specs:
        box = {}
        thread = threading.Thread(
            target=_target, args=(spec, ctx, box), name=f"vvl-probe-{spec.name}", daemon=True
        )
        t0 = time.perf_counter()
        try:
            thread.start()
        except RuntimeError as exc:  # pragma: no cover - cannot start threads
            box.update(exc=f"RuntimeError: {exc}", done=True, duration_ms=0)
        started.append((spec, thread, box, t0))

    outcomes = {}
    for spec, thread, box, t0 in started:
        remaining = spec.timeout_s - (time.perf_counter() - t0)
        if remaining > 0 and thread.is_alive():
            thread.join(remaining)
        if not box.get("done"):
            outcomes[spec.name] = ProbeOutcome(
                spec.name,
                ok=False,
                duration_ms=int((time.perf_counter() - t0) * 1000),
                error=f"timed out after {spec.timeout_s:g} s",
            )
            continue
        if "exc" in box:
            outcomes[spec.name] = ProbeOutcome(
                spec.name, False, box.get("duration_ms", 0), truncate(box["exc"])
            )
            continue
        log = box.get("log") or ProbeLog()
        error = truncate("; ".join(log.errors)) if log.errors else None
        outcomes[spec.name] = ProbeOutcome(
            spec.name, ok=not log.errors, duration_ms=box.get("duration_ms", 0),
            error=error, value=box.get("value"),
        )
    return outcomes
