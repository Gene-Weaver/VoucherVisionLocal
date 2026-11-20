"""OpenRouter routes (docs/API.md section 14):

    GET  /v1/openrouter/catalog[?refresh=1]   the model gallery: every usable model with its
                                              owner, inputs, prices, release date, and the
                                              estimated cost per 1,000 images
    PUT  /v1/openrouter/models                {"models": [...]} or {"add": [...], "remove": [...]}
                                              the models shown in the OCR and parsing pickers
    GET  /v1/openrouter/account               credits and limits of the saved key

The number of requests sent at once is `openrouter.concurrency` (1 to 4) in
PUT /v1/settings.
"""

from __future__ import annotations

import logging

import requests
from flask import Blueprint, jsonify, request

from vvl import openrouter, settings
from vvl.api.common import find_provider, json_body, secret_store, settings_file
from vvl.errors import ApiError

logger = logging.getLogger(__name__)

bp = Blueprint("vvl_openrouter", __name__)

ACCOUNT_TIMEOUT_SECONDS = 15.0


def _spec():
    return find_provider(openrouter.PROVIDER_ID)


def _owners(models: list[dict]) -> list[dict]:
    counts: dict[str, dict] = {}
    for model in models:
        entry = counts.setdefault(model["owner"], {"id": model["owner"], "label": model["owner_label"],
                                                   "count": 0, "vision": 0})
        entry["count"] += 1
        entry["vision"] += 1 if model["vision"] else 0
    return sorted(counts.values(), key=lambda o: (-o["count"], o["label"].lower()))


def catalog_payload(*, refresh: bool = False) -> dict:
    spec = _spec()
    current = settings.load(settings_file())
    block = openrouter.settings_block(current)
    snap = openrouter.catalog.snapshot(spec.base_url, refresh=refresh)
    store = secret_store()
    return {
        "models": snap["models"],
        "owners": _owners(snap["models"]),
        "fetched_at": snap["fetched_at"],
        "stale": snap["stale"],
        "error": snap["error"],
        "added": block["models"],
        "concurrency": block["concurrency"],
        "concurrency_range": [openrouter.CONCURRENCY_MIN, openrouter.CONCURRENCY_MAX],
        "private_only": block["private_only"],
        "configured": store.is_configured(spec.id, spec.field_names()),
        "selection": {"ocr": list(current["selection"]["ocr"]), "parse": current["selection"]["parse"]},
        "cost_tokens": openrouter.COST_TOKENS,
        "key_url": spec.key_url,
    }


@bp.get("/v1/openrouter/catalog")
def get_catalog():
    refresh = request.args.get("refresh") in ("1", "true")
    return jsonify(catalog_payload(refresh=refresh))


def _name_list(body: dict, key: str) -> list[str]:
    value = body.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise ApiError("INVALID_INPUT", f"{key} must be a list of OpenRouter model names.", status=400)
    return value


@bp.put("/v1/openrouter/models")
def put_models():
    body = json_body()
    unknown = sorted(set(body) - {"models", "add", "remove"})
    if unknown:
        raise ApiError("INVALID_INPUT", "Unknown fields.", status=400, details={"unknown_keys": unknown})
    if "models" in body and ("add" in body or "remove" in body):
        raise ApiError("INVALID_INPUT", "Send either models, or add and remove.", status=400)
    path = settings_file()
    current = settings.load(path)
    before = openrouter.settings_block(current)["models"]
    if "models" in body:
        wanted = _name_list(body, "models")
    else:
        remove = set(_name_list(body, "remove"))
        wanted = [m for m in before if m not in remove]
        wanted += [m for m in _name_list(body, "add") if m not in wanted]
    # A new name must be a model OpenRouter lists (when the list is on hand).
    if openrouter.catalog.loaded():
        missing = [m for m in wanted if m not in before and openrouter.catalog.find(m) is None]
        if missing:
            raise ApiError("INVALID_INPUT", f"'{missing[0]}' is not in OpenRouter's model list.", status=400,
                           details={"unknown_models": missing[:20]})
    try:
        changes = settings.validate_update({"openrouter": {"models": wanted}})
    except settings.SettingsError as exc:
        raise ApiError("INVALID_INPUT", str(exc), status=400, details=exc.details) from None
    kept = set(changes["openrouter"]["models"])
    # A removed model cannot stay chosen for OCR or parsing.
    sel = current["selection"]
    gone = lambda model_id: openrouter.is_openrouter_id(model_id) and openrouter.slug_of(model_id) not in kept  # noqa: E731
    ocr = [m for m in sel["ocr"] if not gone(m)] or [settings.DEMO_MODEL_ID]
    parse = settings.DEMO_MODEL_ID if gone(sel["parse"]) else sel["parse"]
    if ocr != sel["ocr"] or parse != sel["parse"]:
        changes["selection"] = {"ocr": ocr, "parse": parse}
    settings.save(changes, path)
    return jsonify(catalog_payload())


@bp.get("/v1/openrouter/account")
def get_account():
    """Credits and limits of the saved key. The key itself is never returned."""
    spec = _spec()
    store = secret_store()
    secret = store.get_value(spec.id, "api_key") if store.is_configured(spec.id, spec.field_names()) else None
    if not secret:
        return jsonify({"configured": False})
    out = {"configured": True, "ok": False, "message": None}
    try:
        response = requests.get(spec.base_url + "/key", timeout=ACCOUNT_TIMEOUT_SECONDS, allow_redirects=False,
                                headers={"Authorization": f"Bearer {secret}", "Accept": "application/json",
                                         "User-Agent": "VoucherVisionLocal"})
    except requests.RequestException as exc:
        logger.info("OpenRouter account: request failed (%s).", type(exc).__name__)
        out["message"] = "Could not reach OpenRouter."
        return jsonify(out)
    try:
        if response.status_code in (401, 403):
            out["message"] = "OpenRouter rejected the saved key."
        elif response.status_code != 200:
            out["message"] = f"OpenRouter answered with HTTP {response.status_code}."
        else:
            data = response.json().get("data") or {}
            number = lambda v: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None  # noqa: E731
            out.update({
                "ok": True,
                "usage_usd": number(data.get("usage")),
                "limit_usd": number(data.get("limit")),
                "limit_remaining_usd": number(data.get("limit_remaining")),
                "is_free_tier": bool(data.get("is_free_tier")),
            })
    except (ValueError, AttributeError):
        out["message"] = "OpenRouter's answer could not be read."
    finally:
        response.close()
    return jsonify(out)
