"""Shape checks for the system report and fit results (docs/API.md, GET /v1/system).

``validate_report(report)`` and ``validate_fit(result)`` return a list of
problems (empty when the shape is right). They check the contract fields only;
extra fields are allowed. Any contract field may be null.
"""

from __future__ import annotations

import json
import re

GPU_KINDS = ("discrete", "integrated", "unified")
GPU_VENDORS = ("nvidia", "amd", "intel", "apple", "unknown")
BACKENDS = ("cuda", "metal", "vulkan", "cpu")
TIERS = ("cpu_only", "entry", "mid", "high", "workstation")
VERDICTS = ("fits_gpu", "fits_split", "cpu_only", "too_large", "unknown")

INT = "int"
NUM = "number"
STR = "str"
BOOL = "bool"
DICT = "dict"
LIST = "list"

OS_FIELDS = {"system": STR, "name": STR, "version": STR, "kernel": STR, "arch": STR,
             "is_wsl": BOOL, "is_apple_silicon": BOOL, "python": STR}
CPU_FIELDS = {"model": STR, "vendor": STR, "physical_cores": INT, "logical_cores": INT, "max_mhz": NUM,
              "features": DICT}
CPU_FEATURES = ("avx", "avx2", "avx512", "fma", "f16c", "neon")
MEMORY_FIELDS = {"total_bytes": INT, "available_bytes": INT, "swap_total_bytes": INT, "swap_free_bytes": INT}
GPU_FIELDS = {"index": INT, "vendor": STR, "name": STR, "kind": STR, "vram_total_bytes": INT,
              "vram_free_bytes": INT, "driver_version": STR, "backend_hints": LIST, "source": STR, "details": DICT}
ACCEL_FIELDS = {"available": BOOL, "detail": STR}
DISK_FIELDS = {"models_dir": STR, "exists": BOOL, "checked_path": STR, "total_bytes": INT, "free_bytes": INT,
               "is_synced_folder": BOOL, "synced_folder_hint": STR}
CAP_FIELDS = {"recommended_backend": STR, "backend_reason": STR, "gpu_memory_budget_bytes": INT,
              "cpu_memory_budget_bytes": INT, "tier": STR, "summary": STR}
WARNING_FIELDS = {"code": STR, "message": STR}
PROBE_FIELDS = {"name": STR, "ok": BOOL, "duration_ms": INT, "error": STR}
FIT_FIELDS = {"verdict": STR, "label": STR, "required_bytes": INT, "detail": STR}


def _type_ok(value, kind):
    if value is None:
        return True
    if kind == INT:
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == NUM:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == STR:
        return isinstance(value, str)
    if kind == BOOL:
        return isinstance(value, bool)
    if kind == DICT:
        return isinstance(value, dict)
    if kind == LIST:
        return isinstance(value, list)
    return False


def _check_obj(obj, fields, path, errs, *, nullable_obj=False):
    if obj is None and nullable_obj:
        return
    if not isinstance(obj, dict):
        errs.append(f"{path}: expected an object, got {type(obj).__name__}")
        return
    for key, kind in fields.items():
        if key not in obj:
            errs.append(f"{path}.{key}: missing")
        elif not _type_ok(obj[key], kind):
            errs.append(f"{path}.{key}: expected {kind} or null, got {type(obj[key]).__name__}")


def _enum(value, allowed, path, errs, nullable=True):
    if value is None and nullable:
        return
    if value not in allowed:
        errs.append(f"{path}: {value!r} is not one of {', '.join(allowed)}")


def validate_report(report):
    errs = []
    if not isinstance(report, dict):
        return [f"report: expected an object, got {type(report).__name__}"]
    if report.get("schema") != 1:
        errs.append(f"schema: expected 1, got {report.get('schema')!r}")
    ga = report.get("generated_at")
    if not isinstance(ga, str) or not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", ga):
        errs.append(f"generated_at: expected an ISO-8601 UTC timestamp like 2026-11-19T18:04:11Z, got {ga!r}")
    dm = report.get("duration_ms")
    if not (isinstance(dm, int) and not isinstance(dm, bool) and dm >= 0):
        errs.append(f"duration_ms: expected a non-negative int, got {dm!r}")

    _check_obj(report.get("os"), OS_FIELDS, "os", errs)
    cpu = report.get("cpu")
    _check_obj(cpu, CPU_FIELDS, "cpu", errs)
    if isinstance(cpu, dict) and isinstance(cpu.get("features"), dict):
        for f in CPU_FEATURES:
            if f not in cpu["features"]:
                errs.append(f"cpu.features.{f}: missing")
            elif not _type_ok(cpu["features"][f], BOOL):
                errs.append(f"cpu.features.{f}: expected bool or null")
    _check_obj(report.get("memory"), MEMORY_FIELDS, "memory", errs)

    gpus = report.get("gpus")
    if not isinstance(gpus, list):
        errs.append("gpus: expected a list")
    else:
        for i, g in enumerate(gpus):
            p = f"gpus[{i}]"
            _check_obj(g, GPU_FIELDS, p, errs)
            if isinstance(g, dict):
                _enum(g.get("vendor"), GPU_VENDORS, f"{p}.vendor", errs, nullable=False)
                _enum(g.get("kind"), GPU_KINDS, f"{p}.kind", errs)
                if g.get("index") != i:
                    errs.append(f"{p}.index: expected {i}, got {g.get('index')!r}")
                hints = g.get("backend_hints")
                if isinstance(hints, list) and not all(isinstance(h, str) for h in hints):
                    errs.append(f"{p}.backend_hints: expected a list of strings")
                if not isinstance(g.get("source"), str):
                    errs.append(f"{p}.source: expected a string")

    acc = report.get("accelerators")
    if not isinstance(acc, dict):
        errs.append("accelerators: expected an object")
    else:
        for name in ("cuda", "metal", "vulkan"):
            _check_obj(acc.get(name), ACCEL_FIELDS, f"accelerators.{name}", errs)

    _check_obj(report.get("disk"), DISK_FIELDS, "disk", errs)
    cap = report.get("capability")
    _check_obj(cap, CAP_FIELDS, "capability", errs)
    if isinstance(cap, dict):
        _enum(cap.get("recommended_backend"), BACKENDS, "capability.recommended_backend", errs)
        _enum(cap.get("tier"), TIERS, "capability.tier", errs)

    for key, fields in (("warnings", WARNING_FIELDS), ("probes", PROBE_FIELDS)):
        items = report.get(key)
        if not isinstance(items, list):
            errs.append(f"{key}: expected a list")
            continue
        for i, item in enumerate(items):
            _check_obj(item, fields, f"{key}[{i}]", errs)
            if key == "warnings" and isinstance(item, dict):
                if not isinstance(item.get("code"), str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", item.get("code") or ""):
                    errs.append(f"warnings[{i}].code: expected an UPPER_SNAKE_CASE string")
                if not isinstance(item.get("message"), str) or not item.get("message"):
                    errs.append(f"warnings[{i}].message: expected a non-empty string")
            if key == "probes" and isinstance(item, dict):
                if not isinstance(item.get("ok"), bool):
                    errs.append(f"probes[{i}].ok: expected a bool")
    try:
        json.dumps(report)
    except (TypeError, ValueError) as exc:
        errs.append(f"report is not JSON-serializable: {exc}")
    return errs


def validate_fit(result):
    errs = []
    _check_obj(result, FIT_FIELDS, "fit", errs)
    if isinstance(result, dict):
        _enum(result.get("verdict"), VERDICTS, "fit.verdict", errs, nullable=False)
        for key in ("label", "detail"):
            if not isinstance(result.get(key), str) or not result.get(key):
                errs.append(f"fit.{key}: expected a non-empty string")
    return errs
