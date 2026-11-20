"""The model catalog behind GET /v1/models and the selection checks.

Combines the demo engine, the local registry (with a fit verdict from the
system checker), and API providers (with model lists fetched on request).
"""

from __future__ import annotations

import logging

from vvl.engines import (
    API_UNAVAILABLE_REASON,
    DEMO_MODEL_ID,
    ModelIdError,
    ResolveContext,
    availability,
    parse_model_id,
)
from vvl import openrouter
from vvl.engines import local as local_engines
from vvl.engines import mock
from vvl.engines import openrouter as openrouter_engines
from vvl.providers import ProviderState, cached_models
from vvl.secrets.store import SecretStore

logger = logging.getLogger(__name__)

FIT_UNKNOWN_DETAIL = "The system check is not available, so model fit was not checked."


def fit_unknown(detail: str = FIT_UNKNOWN_DETAIL) -> dict:
    return {"verdict": "unknown", "label": "Not checked", "required_bytes": None, "detail": detail}


def _system_report():
    """The cached system report, or None when vvl.system is missing or fails."""
    try:
        from vvl import system as vvl_system  # lazy: written in parallel, may be absent

        get_report = getattr(vvl_system, "get_report", None)
        if get_report is None:
            return None
        report = get_report(refresh=False)
        return report if isinstance(report, dict) else None
    except Exception as exc:
        logger.info("System report unavailable for model fit (%s).", type(exc).__name__)
        return None


def _fit_for(report, build_dict) -> dict:
    if report is None:
        return fit_unknown()
    try:
        from vvl.system import fit as vvl_fit

        result = vvl_fit.assess(report, build_dict)
        if isinstance(result, dict) and result.get("verdict"):
            return {
                "verdict": result.get("verdict"),
                "label": result.get("label"),
                "required_bytes": result.get("required_bytes"),
                "detail": result.get("detail"),
            }
    except Exception as exc:
        logger.info("Model fit check failed (%s).", type(exc).__name__)
    return fit_unknown()


VERDICT_RANK = {"fits_gpu": 3, "fits_split": 2, "cpu_only": 1}


def recommended_builds(family, fits: dict) -> str | None:
    """The one build to recommend in a family (docs/API.md section 9).

    Builds are listed from least to most capable in models_local.yaml. The most
    capable build whose verdict is fits_gpu wins. When none fits a GPU: the most
    capable "Split across GPUs" build, else the smallest build that runs
    partly on the GPU or on the processor (speed matters more there).
    """
    builds = list(family.builds)
    gpu = [b for b in builds if (fits.get(b.id) or {}).get("verdict") == "fits_gpu"]
    if gpu:
        return gpu[-1].id
    split = [b for b in builds if (fits.get(b.id) or {}).get("verdict") == "fits_split"]
    multi = [b for b in split if (fits.get(b.id) or {}).get("label") == "Split across GPUs"]
    if multi:
        return multi[-1].id
    if split:
        return split[0].id
    cpu = [b for b in builds if (fits.get(b.id) or {}).get("verdict") == "cpu_only"]
    return cpu[0].id if cpu else None


def _install_info(build, runtime, models_dir) -> dict:
    if runtime is not None:
        return runtime.models.status(build)
    from vvl import settings as settings_mod
    from vvl.local_models import disk_status

    models_dir = models_dir or settings_mod.load()["models_dir"]
    disk = disk_status(models_dir, build)
    return {"install_state": disk["state"], "installed_bytes": disk["installed_bytes"], "download_job_id": None}


def _engine_installed(runtime, models_dir) -> bool:
    try:
        if runtime is not None:
            return bool(runtime.installer.installed_ids())
        from vvl import settings as settings_mod
        from vvl.jobs import JobManager
        from vvl.llama.installer import EngineInstaller
        from vvl.llama.lock import load_lock

        models_dir = models_dir or settings_mod.load()["models_dir"]
        installer = EngineInstaller(JobManager(), lock=load_lock, models_dir=lambda: models_dir, report=lambda: None)
        return bool(installer.installed_ids())
    except Exception as exc:  # noqa: BLE001
        logger.info("Engine install state unavailable (%s).", type(exc).__name__)
        return False


def _local_section(local_models, report, runtime=None, models_dir=None) -> tuple[dict, list[dict], list[dict]]:
    families = []
    sel_ocr, sel_parse = [], []
    engine_installed = _engine_installed(runtime, models_dir)
    caps = local_engines.CAPABILITIES.to_dict()
    for family in local_models.families:
        builds = []
        fits = {}
        for build in family.builds:
            build_dict = build.model_dump()
            build_dict["total_size_bytes"] = build.total_size_bytes
            fits[build.id] = _fit_for(report, build_dict)
        recommended = recommended_builds(family, fits)
        for build in family.builds:
            info = _install_info(build, runtime, models_dir)
            reason = local_engines.reason_for(info["install_state"], engine_installed)
            entry = {
                "id": build.id,
                "model_id": f"local:{build.id}",
                "label": build.label,
                "params": build.params,
                "quant": build.quant,
                "tier": build.tier,
                "hf_repo": build.hf_repo,
                "total_size_bytes": build.total_size_bytes,
                "files": [{"role": f.role, "path": f.path, "size_bytes": f.size_bytes} for f in build.files],
                "install_state": info["install_state"],
                "installed_bytes": info["installed_bytes"],
                "download_job_id": info["download_job_id"],
                "recommended": build.id == recommended,
                "fit": fits[build.id],
                "runnable": reason is None,
                "unavailable_reason": reason,
            }
            builds.append(entry)
            pick = {
                "model_id": entry["model_id"],
                "label": build.label,
                "group": "Local",
                "provider_label": family.label,
                "family_id": family.id,
                "family_label": family.label,
                "family_color": family.brand_color,
                "family_primary": bool(family.primary_family),
                "runnable": reason is None,
                "unavailable_reason": reason,
                "capabilities": dict(caps),
            }
            if "ocr" in family.roles:
                sel_ocr.append(dict(pick))
            if "parse" in family.roles:
                sel_parse.append(dict(pick))
        families.append({
            "id": family.id,
            "label": family.label,
            "developer": family.developer,
            "license": family.license,
            "status": family.status,
            "roles": list(family.roles),
            "summary": family.summary,
            "homepage": family.homepage,
            "brand_color": family.brand_color,
            "primary_family": bool(family.primary_family),
            "builds": builds,
        })
    return {"families": families}, sel_ocr, sel_parse


def openrouter_entry(spec, store: SecretStore, settings: dict | None) -> dict:
    """OpenRouter's entry of /v1/models `api.providers`: the models added from the gallery."""
    configured = store.is_configured(spec.id, spec.field_names())
    block = openrouter.settings_block(settings)
    reason = None if configured else openrouter_engines.REASON_NO_KEY
    models = []
    for slug in block["models"]:
        known = openrouter.catalog.find(slug)
        caps = openrouter_engines.capabilities_for(known).to_dict()
        models.append({
            "id": slug,
            "model_id": openrouter.model_id_for(slug),
            "label": (known or {}).get("full_name") or slug,
            "roles": list((known or {}).get("roles") or ["ocr", "parse"]),
            "known": known is not None,
            "capabilities": caps,
        })
    return {
        "id": spec.id,
        "label": spec.label,
        "kind": spec.kind,
        "brand_color": spec.brand_color,
        "sends_images": spec.sends_images,
        "configured": configured,
        "models_state": "catalog",
        "models_error": None,
        "models": models,
        "concurrency": block["concurrency"],
        "runnable": configured,
        "unavailable_reason": reason,
    }


def api_provider_entry(spec, store: SecretStore, state: ProviderState, settings: dict | None = None) -> dict:
    """One entry of /v1/models `api.providers` (also the refresh route's answer)."""
    if spec.kind == "openrouter":
        return openrouter_entry(spec, store, settings)
    configured = store.is_configured(spec.id, spec.field_names())
    cache = cached_models(spec, state)
    if not configured:
        models_state, models_error, model_ids = "no_key", None, []
    elif cache is None:
        models_state, models_error, model_ids = "not_loaded", None, []
    elif cache.get("error"):
        models_state, models_error, model_ids = "error", cache["error"], []
    else:
        models_state, models_error, model_ids = "loaded", None, list(cache.get("models") or [])
    return {
        "id": spec.id,
        "label": spec.label,
        "kind": spec.kind,
        "brand_color": spec.brand_color,
        "sends_images": spec.sends_images,
        "configured": configured,
        "models_state": models_state,
        "models_error": models_error,
        "models": [{"id": m, "model_id": f"api:{spec.id}:{m}", "label": m} for m in model_ids],
        "runnable": False,
        "unavailable_reason": API_UNAVAILABLE_REASON,
    }


def model_providers(providers):
    return [p for p in providers.providers if p.enabled and p.role == "model_provider"]


def build_catalog(*, providers, local_models, store: SecretStore, state: ProviderState, selection: dict,
                  runtime=None, models_dir: str | None = None, settings: dict | None = None) -> dict:
    report = _system_report()
    local, sel_ocr_local, sel_parse_local = _local_section(local_models, report, runtime, models_dir)

    specs = model_providers(providers)
    api_entries = [api_provider_entry(p, store, state, settings) for p in specs]
    colors = {p.id: p.brand_color for p in specs}
    sel_api_ocr, sel_api_parse = [], []
    for entry in api_entries:
        for model in entry["models"]:
            pick = {
                "model_id": model["model_id"],
                "label": model["label"],
                "group": "API",
                "provider_label": entry["label"],
                "family_id": entry["id"],
                "family_label": entry["label"],
                "family_color": colors.get(entry["id"]),
                "family_primary": False,
                "runnable": bool(entry["runnable"]),
                "unavailable_reason": entry["unavailable_reason"],
                "capabilities": model.get("capabilities") or {
                    "vision": bool(entry["sends_images"]), "thinking_levels": [], "structured_output": True},
            }
            roles = model.get("roles") or ["ocr", "parse"]
            if "ocr" in roles:
                sel_api_ocr.append(pick)
            if "parse" in roles:
                sel_api_parse.append(dict(pick))

    caps = mock.MockEngine.capabilities.to_dict()
    demo_pick = {
        "model_id": DEMO_MODEL_ID,
        "label": mock.SHORT_LABEL,
        "group": "Demo",
        "provider_label": None,
        "family_id": "demo",
        "family_label": "Demo",
        "family_color": None,
        "family_primary": False,
        "runnable": True,
        "unavailable_reason": None,
        "capabilities": caps,
    }
    return {
        "local": local,
        "api": {"providers": api_entries},
        "demo": [{"model_id": DEMO_MODEL_ID, "label": mock.LABEL, "roles": ["ocr", "parse"], "runnable": True}],
        "selection": {"ocr": list(selection.get("ocr") or []), "parse": selection.get("parse")},
        "selectable": {
            "ocr": [dict(demo_pick)] + sel_ocr_local + sel_api_ocr,
            "parse": [dict(demo_pick)] + sel_parse_local + sel_api_parse,
        },
    }


def validate_model_ids(model_ids, role, *, providers, local_models, state: ProviderState, runtime=None) -> None:
    """Raise ModelIdError unless every id names a known model usable for `role`.

    Known means: demo:mock; a local build whose family serves the role; or a
    model of an enabled model provider. When that provider's model list has
    been fetched, the model must be in it.
    """
    context = ResolveContext(providers=providers, local_models=local_models, runtime=runtime)
    for model_id in model_ids:
        ref = parse_model_id(model_id)
        availability(model_id, context)  # raises ModelIdError for unknown builds/providers
        if ref.source == "local":
            family, _build = local_models.find_build(ref.name)
            if role not in family.roles:
                raise ModelIdError(model_id, f"'{model_id}' cannot be used for {role}.")
        elif ref.source == "api":
            spec = next(p for p in providers.providers if p.id == ref.provider)
            if spec.kind == "openrouter":
                known = openrouter.catalog.find(ref.name)
                if known is None and openrouter.catalog.loaded():
                    raise ModelIdError(model_id, f"'{ref.name}' is not in OpenRouter's model list.")
                if known is not None and role not in known["roles"]:
                    raise ModelIdError(model_id, f"'{ref.name}' cannot read images, so it cannot be used for OCR."
                                       if role == "ocr" else f"'{model_id}' cannot be used for {role}.")
                continue
            cache = cached_models(spec, state)
            if cache and cache.get("models") is not None and ref.name not in cache["models"]:
                raise ModelIdError(model_id, f"'{ref.name}' is not in the model list of {spec.label}.")
