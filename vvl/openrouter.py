"""OpenRouter: one key, every hosted model family (Claude, GPT, Gemini, Mistral, Qwen, ...).

Three things live here:

  * the model catalog. `GET https://openrouter.ai/api/v1/models` is public (no
    key), and each entry says what the model reads (text, image, ...), what it
    costs per token, when it was released, and which request fields it accepts.
    The list is cached on disk (<VVL_HOME>/cache/openrouter_models.json) and
    refreshed once a day or on request.
  * which role a model can take: OCR needs text + image input and text output;
    parsing needs text input and text output.
  * the estimated cost per 1,000 images, using the token counts of the
    VoucherVisionGO cost calculator and OpenRouter's prices.

"Private" follows OpenRouter's green shield: a model is private when at least
one of its providers keeps no copy of the request (zero data retention). That
list is public too (`GET /endpoints/zdr`). With `openrouter.private_only` on,
requests are sent with `provider: {"zdr": true}`, so OpenRouter only routes
them to such a provider.

It also holds the request limiter: OpenRouter requests from this app run at most
`openrouter.concurrency` (1 to 4) at a time, whatever the batch asks for.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
import time
from pathlib import Path

import requests

from vvl import paths
from vvl.fileutil import atomic_write_text

logger = logging.getLogger(__name__)

PROVIDER_ID = "openrouter"
MODEL_ID_PREFIX = f"api:{PROVIDER_ID}:"
CATALOG_URL_PATH = "/models"
PRIVATE_URL_PATH = "/endpoints/zdr"
MAX_PRIVATE_PROVIDERS = 12
CATALOG_TIMEOUT_SECONDS = 30.0
CATALOG_MAX_AGE_SECONDS = 24 * 3600
CATALOG_MAX_BYTES = 32 * 1024 * 1024
MAX_CATALOG_MODELS = 5000
DESCRIPTION_CHARS = 420

CONCURRENCY_MIN = 1
CONCURRENCY_MAX = 4
CONCURRENCY_DEFAULT = 4
MAX_ADDED_MODELS = 200

THINKING_LEVELS = ("low", "medium", "high")

# Tokens per image, from the VoucherVisionGO cost calculator (webpage/index.html,
# "Cost Calculator"): the typical case and the worst case of the herbarium
# benchmarks. Thinking tokens are extra and are billed at the output price.
COST_TOKENS = {
    "ocr": {"typical": {"in": 1300, "out": 150}, "high": {"in": 1500, "out": 1000}},
    "parse": {"typical": {"in": 2350, "out": 650}, "high": {"in": 3000, "out": 1000}},
}

# Display names for owner slugs whose model names do not carry one ("Owner: Model").
OWNER_LABELS = {
    "openai": "OpenAI", "google": "Google", "anthropic": "Anthropic", "mistralai": "Mistral",
    "meta-llama": "Meta", "meta": "Meta", "qwen": "Qwen", "x-ai": "xAI", "z-ai": "Z.ai",
    "deepseek": "DeepSeek", "moonshotai": "MoonshotAI", "nvidia": "NVIDIA", "amazon": "Amazon",
    "cohere": "Cohere", "microsoft": "Microsoft", "minimax": "MiniMax", "xiaomi": "Xiaomi",
    "tencent": "Tencent", "perplexity": "Perplexity", "openrouter": "OpenRouter",
}


def model_id_for(slug: str) -> str:
    return MODEL_ID_PREFIX + slug


def is_openrouter_id(model_id) -> bool:
    return isinstance(model_id, str) and model_id.startswith(MODEL_ID_PREFIX)


def slug_of(model_id: str) -> str:
    return model_id[len(MODEL_ID_PREFIX):]


# ----------------------------------------------------------------- normalize --

def _price_per_million(value) -> float | None:
    """OpenRouter prices are USD per token, as strings. Negative means "varies"."""
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if price < 0:
        return None
    return round(price * 1_000_000, 6)


def _owner(slug: str, name: str) -> tuple[str, str]:
    owner = slug.split("/", 1)[0].lstrip("~") if "/" in slug else "other"
    if ":" in name:
        label = name.split(":", 1)[0].strip()
        if label and len(label) <= 40:
            return owner, OWNER_LABELS.get(owner, label)
    return owner, OWNER_LABELS.get(owner, owner.replace("-", " ").title())


def estimate_per_thousand(price_in: float, price_out: float, role: str) -> dict:
    """{"typical": usd, "high": usd} for 1,000 images of one step with this model."""
    out = {}
    for case, tokens in COST_TOKENS[role].items():
        per_image = (tokens["in"] * price_in + tokens["out"] * price_out) / 1_000_000
        out[case] = round(per_image * 1000, 4)
    return out


def normalize_model(raw: dict) -> dict | None:
    """One catalog entry as the app uses it, or None for an entry VVL cannot use."""
    if not isinstance(raw, dict):
        return None
    slug = raw.get("id")
    if not isinstance(slug, str) or not slug or len(slug) > 200:
        return None
    arch = raw.get("architecture") if isinstance(raw.get("architecture"), dict) else {}
    inputs = [m for m in (arch.get("input_modalities") or []) if isinstance(m, str)]
    outputs = [m for m in (arch.get("output_modalities") or []) if isinstance(m, str)]
    pricing = raw.get("pricing") if isinstance(raw.get("pricing"), dict) else {}
    price_in = _price_per_million(pricing.get("prompt"))
    price_out = _price_per_million(pricing.get("completion"))
    if price_in is None or price_out is None:
        return None            # routers with a variable price
    if "text" not in inputs or outputs != ["text"]:
        return None            # image generators, embedding models, ...
    name = raw.get("name") if isinstance(raw.get("name"), str) and raw.get("name").strip() else slug
    owner, owner_label = _owner(slug, name)
    short = name.split(":", 1)[1].strip() if ":" in name and name.split(":", 1)[1].strip() else name
    params = [p for p in (raw.get("supported_parameters") or []) if isinstance(p, str)]
    top = raw.get("top_provider") if isinstance(raw.get("top_provider"), dict) else {}
    created = raw.get("created")
    description = raw.get("description") if isinstance(raw.get("description"), str) else ""
    if len(description) > DESCRIPTION_CHARS:
        description = description[:DESCRIPTION_CHARS].rsplit(" ", 1)[0] + "…"
    vision = "image" in inputs
    roles = ["ocr", "parse"] if vision else ["parse"]
    max_out = top.get("max_completion_tokens")
    return {
        "id": slug,
        "model_id": model_id_for(slug),
        "name": short,
        "full_name": name,
        "owner": owner,
        "owner_label": owner_label,
        "created": int(created) if isinstance(created, (int, float)) and created > 0 else None,
        "context_length": int(raw["context_length"]) if isinstance(raw.get("context_length"), (int, float)) else None,
        "max_output_tokens": int(max_out) if isinstance(max_out, (int, float)) and max_out > 0 else None,
        "input": inputs,
        "vision": vision,
        "roles": roles,
        "price_in": price_in,
        "price_out": price_out,
        "free": price_in == 0 and price_out == 0,
        "reasoning": "reasoning" in params,
        "structured_output": "structured_outputs" in params,
        "json_mode": "response_format" in params,
        "expires": raw.get("expiration_date") if isinstance(raw.get("expiration_date"), str) else None,
        "description": description,
        "cost_1k": {
            "ocr": estimate_per_thousand(price_in, price_out, "ocr") if vision else None,
            "parse": estimate_per_thousand(price_in, price_out, "parse"),
        },
    }


def private_index(payload) -> dict[str, list[str]]:
    """{model name: [provider names]} from OpenRouter's zero-data-retention endpoint list."""
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        raise ValueError("not an endpoint list")
    index: dict[str, list[str]] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        slug, provider = entry.get("model_id"), entry.get("provider_name")
        if not isinstance(slug, str) or not isinstance(provider, str) or not provider:
            continue
        names = index.setdefault(slug, [])
        if provider not in names and len(names) < MAX_PRIVATE_PROVIDERS:
            names.append(provider)
    return index


def apply_private(models: list[dict], index: dict[str, list[str]] | None) -> None:
    """Set `private` (True, False, or None when the list is not on hand) on every model."""
    for model in models:
        if index is None:
            model.setdefault("private", None)
            model.setdefault("private_providers", [])
        else:
            providers = index.get(model["id"], [])
            model["private"] = bool(providers)
            model["private_providers"] = list(providers)


def normalize_catalog(payload) -> list[dict]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise ValueError("not a model list")
    models = []
    seen = set()
    for raw in data[:MAX_CATALOG_MODELS]:
        model = normalize_model(raw)
        if model and model["id"] not in seen:
            seen.add(model["id"])
            models.append(model)
    return models


# --------------------------------------------------------------------- cache --

class Catalog:
    """The cached OpenRouter model list (memory, then disk, then the network)."""

    def __init__(self, cache_path: Path | None = None, *, now=time.time, http_get=None):
        self._path = cache_path
        self._now = now
        self._get = http_get or self._http_get
        self._lock = threading.Lock()
        self._fetch_lock = threading.Lock()
        self._models: list[dict] | None = None
        self._by_id: dict[str, dict] = {}
        self._fetched_at: float | None = None
        self._error: str | None = None
        self._disk_loaded = False
        self._private_retry_at = 0.0

    def cache_path(self) -> Path:
        return self._path or (paths.cache_dir() / "openrouter_models.json")

    @staticmethod
    def _http_get(url: str, timeout: float):
        # No key is sent: the model list is public. No redirects to other hosts.
        response = requests.get(url, headers={"Accept": "application/json", "User-Agent": "VoucherVisionLocal"},
                                timeout=timeout, allow_redirects=False, stream=True)
        try:
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")
            body = response.raw.read(CATALOG_MAX_BYTES + 1, decode_content=True)
            if len(body) > CATALOG_MAX_BYTES:
                raise RuntimeError("the list is too large")
            return json.loads(body.decode("utf-8"))
        finally:
            response.close()

    def _set(self, models: list[dict], fetched_at: float) -> None:
        self._models = models
        self._by_id = {m["id"]: m for m in models}
        self._fetched_at = fetched_at

    def _load_disk(self) -> None:
        if self._disk_loaded:
            return
        self._disk_loaded = True
        try:
            stored = json.loads(self.cache_path().read_text(encoding="utf-8"))
            models = [m for m in stored.get("models") or [] if isinstance(m, dict) and isinstance(m.get("id"), str)]
            fetched_at = float(stored.get("fetched_at"))
        except (OSError, ValueError, TypeError, AttributeError):
            return
        if models:
            self._set(models, fetched_at)

    def _save_disk(self) -> None:
        try:
            path = self.cache_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(path, json.dumps({"fetched_at": self._fetched_at, "models": self._models}),
                              mode=0o644)
        except OSError as exc:
            logger.info("OpenRouter model list could not be cached (%s).", type(exc).__name__)

    def _stale(self) -> bool:
        return self._fetched_at is None or self._now() - self._fetched_at > CATALOG_MAX_AGE_SECONDS

    def refresh(self, base_url: str) -> None:
        """Fetch the list now. A failure keeps the previous list and records the error."""
        with self._fetch_lock:
            try:
                payload = self._get(base_url.rstrip("/") + CATALOG_URL_PATH, CATALOG_TIMEOUT_SECONDS)
                models = normalize_catalog(payload)
                if not models:
                    raise ValueError("the list is empty")
                apply_private(models, self._fetch_private(base_url, models))
            except (requests.ConnectionError, requests.Timeout) as exc:
                logger.info("OpenRouter model list: network error (%s).", type(exc).__name__)
                error = "Could not reach OpenRouter. Check the internet connection and try again."
            except Exception as exc:  # noqa: BLE001 - any bad answer is the same to the user
                logger.info("OpenRouter model list: bad answer (%s).", type(exc).__name__)
                error = "OpenRouter's model list could not be read."
            else:
                with self._lock:
                    self._set(models, self._now())
                    self._error = None
                    self._save_disk()
                logger.info("OpenRouter model list: %d usable models.", len(models))
                return
            with self._lock:
                self._error = error

    def _fetch_private(self, base_url: str, models: list[dict]) -> dict[str, list[str]] | None:
        """The private-provider index; on failure, what the previous list said."""
        try:
            return private_index(self._get(base_url.rstrip("/") + PRIVATE_URL_PATH, CATALOG_TIMEOUT_SECONDS))
        except Exception as exc:  # noqa: BLE001
            logger.info("OpenRouter private-provider list unavailable (%s).", type(exc).__name__)
        with self._lock:
            previous = dict(self._by_id)
        if not any(m.get("private") is not None for m in previous.values()):
            return None
        return {m["id"]: list(previous[m["id"]].get("private_providers") or [])
                for m in models if m["id"] in previous}

    def snapshot(self, base_url: str | None = None, *, refresh: bool = False, fetch_if_stale: bool = True) -> dict:
        """{"models": [...], "fetched_at": epoch | None, "error": str | None, "stale": bool}."""
        with self._lock:
            self._load_disk()
            need = refresh or (fetch_if_stale and self._stale())
            # A list saved without private-provider data is fetched again (at most every 10 minutes).
            if (not need and fetch_if_stale and self._models
                    and all(m.get("private") is None for m in self._models)
                    and self._now() >= self._private_retry_at):
                self._private_retry_at = self._now() + 600
                need = True
        if need and base_url:
            self.refresh(base_url)
        with self._lock:
            return {"models": list(self._models or []), "fetched_at": self._fetched_at,
                    "error": self._error, "stale": self._stale()}

    def find(self, slug: str) -> dict | None:
        """A cached model, without touching the network."""
        with self._lock:
            self._load_disk()
            return self._by_id.get(slug)

    def loaded(self) -> bool:
        with self._lock:
            self._load_disk()
            return self._models is not None


# ------------------------------------------------------------------- limiter --

class RequestLimiter:
    """At most `limit` requests at a time; the limit can change between requests."""

    def __init__(self):
        self._cond = threading.Condition()
        self._active = 0
        self._waiting = 0

    @contextlib.contextmanager
    def slot(self, limit: int):
        limit = max(CONCURRENCY_MIN, min(CONCURRENCY_MAX, int(limit or CONCURRENCY_DEFAULT)))
        with self._cond:
            self._waiting += 1
            try:
                while self._active >= limit:
                    self._cond.wait()
            finally:
                self._waiting -= 1
            self._active += 1
        try:
            yield
        finally:
            with self._cond:
                self._active -= 1
                self._cond.notify_all()

    def status(self) -> dict:
        with self._cond:
            return {"active": self._active, "waiting": self._waiting}


# One catalog and one limiter per process (the app has one OpenRouter account).
catalog = Catalog()
limiter = RequestLimiter()


# ------------------------------------------------------------------ settings --

def settings_block(settings: dict | None) -> dict:
    block = (settings or {}).get("openrouter")
    block = block if isinstance(block, dict) else {}
    models = [m for m in (block.get("models") or []) if isinstance(m, str)]
    concurrency = block.get("concurrency")
    if not isinstance(concurrency, int) or isinstance(concurrency, bool):
        concurrency = CONCURRENCY_DEFAULT
    return {"models": models,
            "concurrency": max(CONCURRENCY_MIN, min(CONCURRENCY_MAX, concurrency)),
            "private_only": block.get("private_only") is True}


def concurrency_for(model_ids, settings: dict | None) -> int | None:
    """How many images a batch should send at once when any chosen model is on OpenRouter."""
    if any(is_openrouter_id(m) for m in model_ids or []):
        return settings_block(settings)["concurrency"]
    return None
