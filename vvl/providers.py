"""Credentials and API providers: what the Keys tab and the API models list show.

State kept in memory only (per process): the last key-test result per
provider and the fetched model lists. Nothing here calls a provider unless a
route asks for a key test or a model-list refresh.

The saved secret is read from the SecretStore only to build the outgoing
request. It never appears in a return value, a log line, or an exception
message: request exceptions are reduced to their type name because their
text can include the URL (and a `?key=` query parameter).
"""

from __future__ import annotations

import datetime
import logging
import threading

import requests

from vvl.registry.loader import ProviderSpec, RequestSpec
from vvl.secrets.store import SecretStore

logger = logging.getLogger(__name__)

KEY_TEST_TIMEOUT_SECONDS = 10.0
MODELS_TIMEOUT_SECONDS = 15.0
MAX_MODELS = 2000

TEST_MESSAGES = {
    "ok": "Key accepted.",
    "invalid_key": "The service rejected this key.",
    "not_configured": "No key is saved for this service.",
    "network_error": "Could not reach the service. Check the internet connection and try again.",
    "error": "The service gave an unexpected answer.",
}


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ProviderState:
    """In-memory, per-process state for providers."""

    def __init__(self):
        self.lock = threading.Lock()
        self.last_tests: dict[str, dict] = {}
        # provider_id -> {"models": [...] | None, "error": str | None}
        self.models: dict[str, dict] = {}

    def clear_provider(self, provider_id: str) -> None:
        with self.lock:
            self.last_tests.pop(provider_id, None)
            self.models.pop(provider_id, None)


# ----------------------------------------------------------- request build --

def build_request(spec: RequestSpec, base_url: str, secret: str) -> tuple[str, str, dict, dict]:
    """(method, url, headers, params) for a registry request spec."""
    url = spec.url.replace("{base_url}", base_url.rstrip("/"))
    headers = {"Accept": "application/json", "User-Agent": "VoucherVisionLocal"}
    headers.update(spec.extra_headers or {})
    params: dict[str, str] = {}
    auth = spec.auth
    if auth == "bearer":
        headers["Authorization"] = f"Bearer {secret}"
    elif auth.startswith("header:"):
        headers[auth.split(":", 1)[1]] = secret
    elif auth.startswith("query:"):
        params[auth.split(":", 1)[1]] = secret
    return spec.method, url, headers, params


def _send(spec: RequestSpec, base_url: str, secret: str, timeout: float) -> requests.Response:
    method, url, headers, params = build_request(spec, base_url, secret)
    # No redirects: a redirect could carry a custom auth header to another host.
    return requests.request(method, url, headers=headers, params=params, timeout=timeout, allow_redirects=False)


def _primary_secret(spec: ProviderSpec, store: SecretStore) -> str | None:
    return store.get_value(spec.id, spec.fields[0].name)


# ---------------------------------------------------------------- key test --

def _looks_like_bad_key(response: requests.Response) -> bool:
    """Google answers 400 API_KEY_INVALID for a bad query-string key."""
    if response.status_code != 400:
        return False
    try:
        text = response.text[:4000].upper()
    except Exception:
        return False
    return "API_KEY_INVALID" in text or "API KEY NOT VALID" in text


def run_key_test(spec: ProviderSpec, store: SecretStore, state: ProviderState,
                 timeout: float = KEY_TEST_TIMEOUT_SECONDS) -> dict:
    """Run the registry `test` request with the saved key. Never echoes key or body."""
    result = {"ok": False, "status": "error", "http_status": None, "message": "", "tested_at": utc_now_iso()}
    if spec.test is None:
        result["message"] = "This service has no key test."
        return result
    secret = _primary_secret(spec, store)
    if not store.is_configured(spec.id, spec.field_names()) or not secret:
        result.update(status="not_configured", message=TEST_MESSAGES["not_configured"])
    else:
        try:
            response = _send(spec.test, spec.base_url, secret, timeout)
        except (requests.ConnectionError, requests.Timeout) as exc:
            logger.info("Key test for %s: network error (%s).", spec.id, type(exc).__name__)
            result.update(status="network_error", message=TEST_MESSAGES["network_error"])
        except requests.RequestException as exc:
            logger.info("Key test for %s: request failed (%s).", spec.id, type(exc).__name__)
            result.update(status="error", message=TEST_MESSAGES["error"])
        else:
            code = response.status_code
            result["http_status"] = code
            if code in spec.test.ok_status:
                result.update(ok=True, status="ok", message=TEST_MESSAGES["ok"])
            elif code in (401, 403) or _looks_like_bad_key(response):
                result.update(status="invalid_key", message=TEST_MESSAGES["invalid_key"])
            else:
                result.update(status="error", message=f"The service answered with HTTP {code}.")
            response.close()
            logger.info("Key test for %s: %s (HTTP %s).", spec.id, result["status"], code)
    with state.lock:
        state.last_tests[spec.id] = dict(result)
    return result


# ------------------------------------------------------------- model lists --

def extract_ids(payload, id_path: str, strip_prefix: str = "") -> list[str]:
    """Pull model ids out of a list response, e.g. id_path 'data[].id' or 'models[].name'."""
    nodes = [payload]
    for segment in id_path.split("."):
        iterate = segment.endswith("[]")
        key = segment[:-2] if iterate else segment
        next_nodes = []
        for node in nodes:
            value = node.get(key) if isinstance(node, dict) and key else node
            if iterate:
                if isinstance(value, list):
                    next_nodes.extend(value)
            elif value is not None:
                next_nodes.append(value)
        nodes = next_nodes
    ids = []
    for node in nodes:
        if isinstance(node, str) and node:
            ident = node[len(strip_prefix):] if strip_prefix and node.startswith(strip_prefix) else node
            if ident and ident not in ids:
                ids.append(ident)
    return ids[:MAX_MODELS]


def refresh_models(spec: ProviderSpec, store: SecretStore, state: ProviderState,
                   timeout: float = MODELS_TIMEOUT_SECONDS) -> None:
    """Fetch and cache the provider's model list. Failures are cached as an error."""
    if spec.models.source != "list_endpoint":
        return
    secret = _primary_secret(spec, store)
    if not store.is_configured(spec.id, spec.field_names()) or not secret:
        with state.lock:
            state.models.pop(spec.id, None)
        return
    error = None
    ids: list[str] | None = None
    try:
        list_spec = RequestSpec(
            method="GET", url=spec.models.url, auth=spec.models.auth,
            extra_headers=dict(spec.models.extra_headers or {}), ok_status=[200],
        )
        response = _send(list_spec, spec.base_url, secret, timeout)
        if response.status_code == 200:
            try:
                ids = extract_ids(response.json(), spec.models.id_path, spec.models.strip_prefix)
            except ValueError:
                error = "The service answered with something that is not a model list."
        elif response.status_code in (401, 403) or _looks_like_bad_key(response):
            error = "The service rejected the saved key."
        else:
            error = f"The service answered with HTTP {response.status_code}."
        response.close()
    except (requests.ConnectionError, requests.Timeout) as exc:
        logger.info("Model list for %s: network error (%s).", spec.id, type(exc).__name__)
        error = TEST_MESSAGES["network_error"]
    except requests.RequestException as exc:
        logger.info("Model list for %s: request failed (%s).", spec.id, type(exc).__name__)
        error = "The model list could not be fetched."
    with state.lock:
        state.models[spec.id] = {"models": ids if error is None else None, "error": error}
    logger.info("Model list for %s: %s", spec.id, f"{len(ids)} models" if error is None else "error")


def cached_models(spec: ProviderSpec, state: ProviderState) -> dict | None:
    with state.lock:
        entry = state.models.get(spec.id)
        return dict(entry) if entry else None


# ------------------------------------------------------------ API objects --

def provider_object(spec: ProviderSpec, store: SecretStore, state: ProviderState) -> dict:
    """One entry of GET /v1/providers (docs/API.md)."""
    status = store.field_status(spec.id, spec.field_names())
    fields = []
    for f in spec.fields:
        st = status.get(f.name, {"saved": False, "masked": None})
        fields.append({
            "name": f.name,
            "label": f.label,
            "placeholder": f.placeholder,
            "saved": st["saved"],
            "masked": st["masked"],
        })
    with state.lock:
        last_test = state.last_tests.get(spec.id)
    return {
        "id": spec.id,
        "label": spec.label,
        "role": spec.role,
        "kind": spec.kind,
        "purpose": spec.purpose,
        "sends_images": spec.sends_images,
        "optional": spec.optional,
        "key_url": spec.key_url,
        "docs_url": spec.docs_url,
        "fields": fields,
        "configured": all(f["saved"] for f in fields),
        "last_test": dict(last_test) if last_test else None,
    }
