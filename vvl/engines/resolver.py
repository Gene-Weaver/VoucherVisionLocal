"""Model id -> engine.

    demo:mock                    the demo engine (always runnable)
    local:<build_id>             a build in vvl/registry/models_local.yaml
    api:<provider_id>:<model>    a model of a `model_provider` in providers.yaml

Engines are created by factories registered per source (and, for `api`, per
provider kind). A factory may come with a `check(ref, context) -> (runnable,
reason)` hook, used by `availability()` and before `resolve_engine()` creates
an engine. The demo, local, and OpenRouter factories are registered here;
the other `api:` kinds raise ModelNotAvailable until an adapter calls
`register_factory("api", ..., kind="gemini")` and so on. The pipeline only
ever calls `resolve_engine()`.
"""

from __future__ import annotations

import re
import threading
from typing import Callable

from vvl.engines import local as local_engines
from vvl.engines import mock
from vvl.engines import openrouter as openrouter_engines
from vvl.engines.base import Engine, ModelIdError, ModelNotAvailable, ModelRef, ResolveContext

DEMO_MODEL_ID = mock.MODEL_ID
LOCAL_UNAVAILABLE_REASON = "This local model is not available yet."
API_UNAVAILABLE_REASON = "This provider is not available in this version yet. Use OpenRouter for hosted models."

# "~" starts OpenRouter's "latest" pointers, e.g. ~anthropic/claude-opus-latest.
_MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9~][A-Za-z0-9._:/@+~-]{0,199}$")
_BUILD_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
_PROVIDER_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

Factory = Callable[[ModelRef, ResolveContext], Engine]
Check = Callable[[ModelRef, ResolveContext], "tuple[bool, str | None]"]

_factories: dict[tuple[str, str | None], Factory] = {
    ("demo", None): mock.create,
    ("local", None): local_engines.create,
    ("api", "openrouter"): openrouter_engines.create,
}
_checks: dict[tuple[str, str | None], Check] = {
    ("local", None): local_engines.availability,
    ("api", "openrouter"): openrouter_engines.availability,
}
_factories_lock = threading.Lock()


def register_factory(source: str, factory: Factory, *, kind: str | None = None, check: Check | None = None) -> None:
    """Make a source (or, for api, one provider kind) runnable."""
    if source not in ("demo", "local", "api"):
        raise ValueError(f"unknown model source {source!r}")
    with _factories_lock:
        _factories[(source, kind)] = factory
        if check is not None:
            _checks[(source, kind)] = check
        else:
            _checks.pop((source, kind), None)


def unregister_factory(source: str, *, kind: str | None = None) -> None:
    with _factories_lock:
        _factories.pop((source, kind), None)
        _checks.pop((source, kind), None)


def _factory_for(source: str, kind: str | None) -> Factory | None:
    with _factories_lock:
        return _factories.get((source, kind)) or (_factories.get((source, None)) if kind else None)


def _check_for(source: str, kind: str | None) -> Check | None:
    with _factories_lock:
        if (source, kind) in _factories:
            return _checks.get((source, kind))
        return _checks.get((source, None)) if kind else None


def parse_model_id(model_id) -> ModelRef:
    """Split a model id into its parts. Checks syntax only."""
    if not isinstance(model_id, str) or not model_id.strip():
        raise ModelIdError(str(model_id), "A model id must be a non-empty string.")
    model_id = model_id.strip()
    source, sep, rest = model_id.partition(":")
    if not sep:
        raise ModelIdError(
            model_id,
            f"Model id '{model_id}' has no source prefix. Use demo:mock, "
            "local:<build_id>, or api:<provider_id>:<model>.",
        )
    if source == "demo":
        if rest != "mock":
            raise ModelIdError(model_id, f"Unknown demo model '{model_id}'. The demo model is demo:mock.")
        return ModelRef(model_id, "demo", "mock")
    if source == "local":
        if not _BUILD_ID_PATTERN.match(rest):
            raise ModelIdError(model_id, f"'{model_id}' is not a valid local model id.")
        return ModelRef(model_id, "local", rest)
    if source == "api":
        provider, sep2, name = rest.partition(":")
        if not sep2 or not _PROVIDER_ID_PATTERN.match(provider) or not _MODEL_NAME_PATTERN.match(name):
            raise ModelIdError(model_id, f"'{model_id}' is not a valid API model id (api:<provider_id>:<model>).")
        return ModelRef(model_id, "api", name, provider=provider)
    raise ModelIdError(model_id, f"Unknown model source '{source}' in '{model_id}'.")


def _check_known(ref: ModelRef, context: ResolveContext):
    """Raise ModelIdError when the id names nothing in the registries. Returns the
    registry entry (build or provider) for later checks."""
    if ref.source == "local":
        found = context.local_models.find_build(ref.name) if context.local_models is not None else None
        if not found:
            raise ModelIdError(ref.model_id, f"Unknown local model '{ref.name}'.")
        return found
    if ref.source == "api":
        provider = None
        if context.providers is not None:
            provider = next(
                (p for p in context.providers.providers if p.id == ref.provider and p.enabled), None
            )
        if provider is None or provider.role != "model_provider":
            raise ModelIdError(ref.model_id, f"Unknown API provider '{ref.provider}'.")
        return provider
    return None


def availability(model_id: str, context: ResolveContext) -> tuple[bool, str | None]:
    """(runnable, unavailable_reason) for a known id. Raises ModelIdError otherwise."""
    ref = parse_model_id(model_id)
    entry = _check_known(ref, context)
    kind = getattr(entry, "kind", None) if ref.source == "api" else None
    if _factory_for(ref.source, kind) is not None:
        check = _check_for(ref.source, kind)
        return check(ref, context) if check is not None else (True, None)
    if ref.source == "local":
        return False, LOCAL_UNAVAILABLE_REASON
    if ref.source == "api":
        return False, API_UNAVAILABLE_REASON
    return False, "This model cannot run."


def resolve_engine(model_id: str, context: ResolveContext) -> Engine:
    """An engine ready to run, or ModelIdError (400) / ModelNotAvailable (409)."""
    ref = parse_model_id(model_id)
    entry = _check_known(ref, context)
    kind = getattr(entry, "kind", None) if ref.source == "api" else None
    factory = _factory_for(ref.source, kind)
    if factory is None:
        reason = LOCAL_UNAVAILABLE_REASON if ref.source == "local" else API_UNAVAILABLE_REASON
        raise ModelNotAvailable(ref.model_id, reason)
    check = _check_for(ref.source, kind)
    if check is not None:
        runnable, reason = check(ref, context)
        if not runnable:
            raise ModelNotAvailable(ref.model_id, reason or "This model cannot run yet.")
    return factory(ref, context)
