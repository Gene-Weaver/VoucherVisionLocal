"""Shared helpers for the /v1 blueprints."""

from __future__ import annotations

from pathlib import Path

from flask import current_app, request

from vvl import paths
from vvl.errors import ApiError
from vvl.providers import ProviderState
from vvl.registry import loader
from vvl.secrets.store import SecretStore


def vvl_state() -> dict:
    return current_app.extensions["vvl"]


def provider_state() -> ProviderState:
    return vvl_state()["provider_state"]


def runtime():
    """The app's vvl.runtime.Runtime (jobs, model store, engine installer, supervisor)."""
    return vvl_state()["runtime"]


def secret_store() -> SecretStore:
    override = current_app.config.get("VVL_PRIVATE_PATH")
    return SecretStore(Path(override) if override else None)


def settings_file() -> Path:
    override = current_app.config.get("VVL_SETTINGS_PATH")
    return Path(override) if override else paths.settings_path()


def providers_registry() -> loader.ProvidersFile:
    return loader.load_providers(current_app.config.get("VVL_PROVIDERS_PATH"))


def local_registry() -> loader.LocalModelsFile:
    return loader.load_local_models(current_app.config.get("VVL_LOCAL_MODELS_PATH"))


def find_provider(provider_id: str) -> loader.ProviderSpec:
    for spec in providers_registry().providers:
        if spec.enabled and spec.id == provider_id:
            return spec
    raise ApiError("NOT_FOUND", f"Unknown provider '{provider_id}'.", status=404)


def json_body(*, required: bool = True) -> dict:
    data = request.get_json(silent=True)
    if data is None:
        if required:
            raise ApiError("INVALID_INPUT", "The request body must be JSON.", status=400)
        return {}
    if not isinstance(data, dict):
        raise ApiError("INVALID_INPUT", "The request body must be a JSON object.", status=400)
    return data
