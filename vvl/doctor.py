"""Readiness checks for `vvl doctor`. Each check returns ok / warn / fail and one line.

Phase 2 adds `engine` (the installed llama.cpp build, its version and devices)
and `local models` (verified and incomplete downloads), both read from disk.

PRIVATE.yaml is only checked for existence; it is never opened here.
"""

from __future__ import annotations

import os
import platform
import sys

from vvl import paths, prompts, settings


def _check(name, func):
    try:
        status, detail = func()
    except Exception as exc:
        status, detail = "fail", f"{type(exc).__name__}: {exc}"
    return {"name": name, "status": status, "detail": detail}


def _python():
    version = platform.python_version()
    ok = sys.version_info >= (3, 11)
    return ("ok" if ok else "fail"), f"Python {version}"


def _registry():
    from vvl.registry import loader

    providers = loader.load_providers()
    local = loader.load_local_models()
    builds = len(local.builds())
    return "ok", f"{len(providers.providers)} providers, {len(local.families)} model families, {builds} local builds"


def _prompts():
    items = prompts.list_all()
    builtin = [p for p in items if p["source"] == "builtin"]
    user = [p for p in items if p["source"] == "user"]
    bad = [p["name"] for p in items if not p["valid"]]
    if not builtin:
        return "fail", f"No built-in prompts in {paths.prompts_builtin_dir()}"
    detail = f"{len(builtin)} built-in, {len(user)} user"
    if bad:
        return "warn", detail + f"; invalid: {', '.join(bad[:5])}"
    return "ok", detail + ", all valid"


def _textcollage():
    from vvl import label_detector
    from vvl.pipeline import VoucherVisionProcessor

    lock = label_detector.load_lock()
    fmt = label_detector.default_format()
    if not label_detector.is_installed(lock, fmt):
        size_mb = label_detector.size_bytes(lock, fmt) / 1e6
        return "warn", (f"Label detector not downloaded yet ({size_mb:.0f} MB; `vvl serve` downloads it at "
                        "first start). Until then the whole image is used instead of a label collage.")
    processor = VoucherVisionProcessor()
    if processor.collage_engine is None:
        return "fail", f"The label detector could not be loaded: {processor.collage_status()['load_error']}"
    status = processor.collage_status()
    if status.get("gpu_error"):
        return "warn", f"Label detector loads ({status['loaded_runtime']}); the GPU failed: {status['gpu_error']}"
    return "ok", f"Label detector loads ({status['loaded_runtime']})"


def _formats():
    import pillow_heif  # noqa: F401
    import pymupdf  # noqa: F401

    return "ok", "HEIC/HEIF and PDF input available"


def _engine_installer():
    from vvl.jobs import JobManager
    from vvl.llama.installer import EngineInstaller
    from vvl.llama.lock import load_lock

    models_dir = settings.load()["models_dir"]

    def _report():
        from vvl.api.system import get_system_report

        return get_system_report(refresh=False)

    return EngineInstaller(JobManager(), lock=load_lock, models_dir=lambda: models_dir, report=_report)


def _engine():
    installer = _engine_installer()
    tag = installer.lock.release.tag
    preferred = settings.load()["engine"].get("variant", "auto")
    active = installer.active_install(preferred)
    if active is not None:
        devices = ", ".join(d["name"] for d in active.get("devices") or []) or "no GPU devices"
        others = sorted(installer.installed_ids() - {active["variant"]})
        extra = f"; also installed: {', '.join(others)}" if others else ""
        return "ok", f"llama.cpp {tag} {active['variant']} ({active.get('version')}); devices: {devices}{extra}"
    rec = installer.recommended()
    if rec is None:
        return "warn", f"llama.cpp {tag} is not installed, and no build fits this computer"
    return "warn", f"llama.cpp {tag} is not installed (recommended build: {rec['id']}; Engine tab)"


def _local_models():
    from vvl.local_models import disk_status
    from vvl.registry import loader

    models_dir = settings.load()["models_dir"]
    installed, partial = [], []
    for _family, build in loader.load_local_models().builds():
        state = disk_status(models_dir, build)["state"]
        if state == "installed":
            installed.append(build.id)
        elif state == "partial":
            partial.append(build.id)
    detail = f"installed: {', '.join(installed) if installed else 'none'}"
    if partial:
        detail += f"; incomplete downloads: {', '.join(partial)}"
    return ("ok" if installed else "warn"), detail


def _demo_engine():
    from vvl.engines import DEMO_MODEL_ID, ResolveContext, resolve_engine

    engine = resolve_engine(DEMO_MODEL_ID, ResolveContext())
    return "ok", f"{engine.label} is runnable"


def _settings():
    current = settings.load()
    models_dir = current["models_dir"]
    hint = paths.synced_folder_hint(models_dir)
    where = paths.settings_path()
    source = "file" if where.exists() else "defaults (no file yet)"
    if hint:
        return "warn", f"models_dir {models_dir} is inside a synced folder ({hint}); settings from {source}"
    exists = "exists" if os.path.isdir(models_dir) else "not created yet"
    return "ok", f"models_dir {models_dir} ({exists}); settings from {source}"


def _credentials():
    path = paths.private_path()
    if path.exists():
        return "ok", f"Credentials file present: {path}"
    return "ok", f"No credentials file yet (created when a key is saved): {path}"


def _system():
    from vvl.api.system import get_system_report

    report = get_system_report(refresh=False)
    capability = report.get("capability") or {}
    warnings = [w.get("code") for w in report.get("warnings") or [] if isinstance(w, dict)]
    if "SYSTEM_CHECK_UNAVAILABLE" in warnings:
        return "warn", "System checker not available (vvl.system)"
    summary = capability.get("summary") or "report generated"
    backend = capability.get("recommended_backend")
    return "ok", f"{summary}" + (f"; recommended backend: {backend}" if backend else "")


def run_checks() -> list[dict]:
    return [
        _check("python", _python),
        _check("registry", _registry),
        _check("prompts", _prompts),
        _check("textcollage", _textcollage),
        _check("image formats", _formats),
        _check("demo engine", _demo_engine),
        _check("engine", _engine),
        _check("local models", _local_models),
        _check("settings", _settings),
        _check("credentials", _credentials),
        _check("system", _system),
    ]
