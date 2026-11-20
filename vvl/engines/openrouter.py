"""`api:openrouter:<model>` models: OpenRouter through the generic OpenAI-compatible engine.

Everything else is the same code path as a local model (prompts, TextCollage,
sanitizing, enrichment); only the two model calls go to
https://openrouter.ai/api/v1/chat/completions instead of llama.cpp.

    OCR      image (base64 data URL) + OCR prompt. The model must read images.
    parsing  the parsing prompt, with the prompt's JSON schema as
             `response_format` when the model lists `structured_outputs`. If
             OpenRouter has no endpoint that takes the schema, the request is
             sent once more without it (the prompt asks for JSON anyway).

Thinking level (low, medium, high) is sent as OpenRouter's unified
`reasoning: {effort}` for models that list `reasoning`; the reasoning text is
not returned. With `openrouter.private_only` on, every request carries
`provider: {"zdr": true}` (only providers that keep no copy). Requests wait for a free slot in vvl.openrouter.limiter, so at
most `openrouter.concurrency` (1 to 4) run at a time. HTTP 429 and 5xx are
retried with backoff, honoring Retry-After.

Cost is tokens x the catalog price, the same way VoucherVisionGO reports it.

Diagnostic switches (environment variables, read for every request; used by
tools/sweep_openrouter_jouvea.py, off in normal use):

    VVL_OPENROUTER_REASONING=default      send `reasoning: {"enabled": true}` (the model's own
                                          default effort) instead of the thinking level
    VVL_OPENROUTER_REASONING=off          ask for no reasoning (`effort: "none"`); when OpenRouter
                                          refuses (HTTP 400/404/422) try "minimal", then "low"
    VVL_OPENROUTER_INCLUDE_REASONING=1    ask for the reasoning text to be returned
    VVL_OPENROUTER_IMAGE_DETAIL=high      `detail` on the image part (OpenAI-style)
    VVL_OPENROUTER_LOG=<file.jsonl>       append every request (image data left out, never the
                                          key) and OpenRouter's full answer, one JSON object a line
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
import threading

from vvl import openrouter
from vvl.engines.base import (
    EngineCapabilities,
    EngineError,
    ModelNotAvailable,
    ModelRef,
    OCRRequest,
    ParseRequest,
    ResolveContext,
)
from vvl.engines.openai_compat import OpenAICompatConfig, OpenAICompatEngine, StepOptions

logger = logging.getLogger(__name__)

REASON_NO_KEY = "Add an OpenRouter key first (Keys tab)."
REASON_NO_VISION = "This model cannot read images, so it cannot be used for OCR."
NO_PRIVATE_PROVIDER = ("No private provider offers this model. Turn off \"Private providers only\" on the Models "
                       "tab, or choose another model.")
SERVICE_LABEL = "OpenRouter"
TIMEOUT_SECONDS = 600.0
MAX_RETRIES = 4
BACKOFF_SECONDS = (2.0, 5.0, 10.0, 20.0)
# Output token limits. OpenRouter checks that the account can pay for the whole
# limit before it runs a request, so these stay close to what a label needs.
MAX_TOKENS = {"ocr": 8192, "parse": 8192}
MAX_TOKENS_REASONING = {"ocr": 16384, "parse": 16384}
SCHEMA_FALLBACK_STATUS = (400, 404, 422)
# "Reasoning off" asks for these in turn until OpenRouter accepts one.
OFF_EFFORTS = ("none", "minimal", "low")

_log_lock = threading.Lock()


def _loggable(body: dict) -> dict:
    """The request body with image data replaced by its size."""
    out = dict(body)
    messages = []
    for message in body.get("messages") or []:
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                url = ((part.get("image_url") or {}).get("url") or "") if isinstance(part, dict) else ""
                if url.startswith("data:"):
                    kept = {k: v for k, v in part["image_url"].items() if k != "url"}
                    kept["url"] = f"<image, {len(url) * 3 // 4:,} bytes, left out of the log>"
                    part = dict(part, image_url=kept)
                parts.append(part)
            message = dict(message, content=parts)
        messages.append(message)
    out["messages"] = messages
    return out


def write_exchange(path: str, step: str, body: dict, status: int, answer) -> None:
    line = json.dumps({
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "step": step, "model": body.get("model"), "http_status": status,
        "request": _loggable(body), "response": answer,
    }, ensure_ascii=False)
    with _log_lock:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


FRIENDLY_ERRORS = {
    401: "OpenRouter rejected the saved key. Check it on the Keys tab.",
    402: "The OpenRouter account does not have enough credits for this request.",
    403: "OpenRouter refused this request (the model may be blocked by the account's privacy or moderation settings).",
    408: "OpenRouter timed out waiting for the model.",
    429: "OpenRouter is rate-limiting this account. Lower the number of concurrent requests, or wait a minute.",
}


def _key(context: ResolveContext) -> str | None:
    store = getattr(context, "secrets", None)
    if store is None:
        return None
    try:
        return store.get_value(openrouter.PROVIDER_ID, "api_key") or None
    except Exception:  # noqa: BLE001 - an unreadable credentials file means "no key"
        return None


def _base_url(context: ResolveContext) -> str:
    providers = getattr(context, "providers", None)
    for spec in getattr(providers, "providers", []) or []:
        if spec.id == openrouter.PROVIDER_ID:
            return spec.base_url
    return "https://openrouter.ai/api/v1"


def availability(ref: ModelRef, context: ResolveContext) -> tuple[bool, str | None]:
    if not _key(context):
        return False, REASON_NO_KEY
    return True, None


def capabilities_for(model: dict | None) -> EngineCapabilities:
    """What the catalog says about a model. An unknown model is assumed to read images."""
    if model is None:
        return EngineCapabilities(vision=True, structured_output=False, thinking_levels=())
    return EngineCapabilities(
        vision=bool(model.get("vision")),
        structured_output=bool(model.get("structured_output")),
        thinking_levels=openrouter.THINKING_LEVELS if model.get("reasoning") else (),
    )


class OpenRouterEngine:
    notice = None

    def __init__(self, ref: ModelRef, *, api_key: str, base_url: str, model: dict | None, settings: dict | None,
                 client=None, sleep=None, limiter: openrouter.RequestLimiter | None = None,
                 provider: dict | None = None):
        self.model_id = ref.model_id
        self.slug = ref.name
        self.model = model
        self.label = (model or {}).get("full_name") or ref.name
        self.capabilities = capabilities_for(model)
        self._api_key = api_key
        self._base_url = base_url
        self._settings = settings or {}
        self._client = client
        self._sleep = sleep
        self._limiter = limiter or openrouter.limiter
        # OpenRouter provider routing: {"zdr": true} for private only; callers may pin one
        # provider with {"only": ["amazon-bedrock"]}.
        self._provider = dict(provider or {})
        if openrouter.settings_block(self._settings)["private_only"]:
            self._provider["zdr"] = True

    # -- request building -------------------------------------------------

    def _step_options(self, step: str, thinking_level: str, max_output_tokens: int,
                      effort: str | None = None) -> StepOptions:
        reasons = bool(self.capabilities.thinking_levels)
        limit = (MAX_TOKENS_REASONING if reasons else MAX_TOKENS)[step]
        model_limit = (self.model or {}).get("max_output_tokens")
        if model_limit:
            limit = min(limit, int(model_limit))
        if max_output_tokens:
            limit = min(limit, int(max_output_tokens))
        extra: dict = {}
        if reasons:
            level = thinking_level if thinking_level in openrouter.THINKING_LEVELS else "low"
            include = os.environ.get("VVL_OPENROUTER_INCLUDE_REASONING") == "1"
            # exclude: the reasoning is paid for but its text is not sent back.
            if effort is not None:
                extra["reasoning"] = {"effort": effort, "exclude": not include}
            elif os.environ.get("VVL_OPENROUTER_REASONING") == "default":
                extra["reasoning"] = {"enabled": True, "exclude": not include}
            else:
                extra["reasoning"] = {"effort": level, "exclude": not include}
        detail = os.environ.get("VVL_OPENROUTER_IMAGE_DETAIL") if step == "ocr" else None
        return StepOptions(max_tokens=limit, sampling={}, image_first=True, extra_body=extra,
                           image_detail=detail if detail in ("high", "low", "auto") else None)

    def _engine(self, options: StepOptions, *, structured: bool, step: str = "") -> OpenAICompatEngine:
        log_path = os.environ.get("VVL_OPENROUTER_LOG")
        hook = None
        if log_path:
            hook = lambda body, status, answer: write_exchange(log_path, step, body, status, answer)  # noqa: E731
        caps = dataclasses.replace(self.capabilities, structured_output=structured)
        extra_body = {}
        provider = dict(self._provider)
        if structured:
            # Only route to an endpoint that honors the JSON schema.
            provider["require_parameters"] = True
        if provider:
            extra_body["provider"] = provider
        config = OpenAICompatConfig(
            base_url=self._base_url,
            api_key=self._api_key,
            model=self.slug,
            model_id=self.model_id,
            label=self.label,
            capabilities=caps,
            ocr=options,
            parse=options,
            timeout_seconds=TIMEOUT_SECONDS,
            max_retries=MAX_RETRIES,
            retry_backoff_seconds=BACKOFF_SECONDS,
            extra_body=extra_body,
            error_code="UPSTREAM_UNAVAILABLE",
            service_label=SERVICE_LABEL,
            on_exchange=hook,
        )
        kwargs = {"client": self._client}
        if self._sleep is not None:
            kwargs["sleep"] = self._sleep
        return OpenAICompatEngine(config, **kwargs)

    def _friendly(self, error: EngineError) -> EngineError:
        message = FRIENDLY_ERRORS.get(getattr(error, "upstream_status", None))
        if error.upstream_status == 404 and self._provider.get("zdr"):
            message = NO_PRIVATE_PROVIDER
        if not message:
            return error
        code = {401: "UPSTREAM_AUTH_FAILED", 402: "UPSTREAM_QUOTA_EXHAUSTED", 408: "UPSTREAM_TIMEOUT",
                429: "UPSTREAM_QUOTA_EXHAUSTED"}.get(error.upstream_status, error.code)
        friendly = EngineError(message, code=code, status=error.status)
        friendly.upstream_status = error.upstream_status
        return friendly

    def _priced(self, result, thinking_level: str):
        """Fill the cost fields from the catalog prices (USD per 1,000,000 tokens)."""
        model = self.model or {}
        rate_in = float(model.get("price_in") or 0.0)
        rate_out = float(model.get("price_out") or 0.0)
        thinking = min(int(result.thinking_tokens or 0), int(result.tokens_out or 0))
        result.cost_in = result.tokens_in * rate_in / 1_000_000
        # OpenRouter counts reasoning tokens inside completion_tokens.
        result.cost_out = (result.tokens_out - thinking) * rate_out / 1_000_000
        result.thinking_cost = thinking * rate_out / 1_000_000
        if hasattr(result, "rates_in"):
            result.rates_in = rate_in
            result.rates_out = rate_out
        if self.capabilities.thinking_levels:
            result.thinking_budget_requested = thinking_level
        return result

    def _concurrency(self) -> int:
        return openrouter.settings_block(self._settings)["concurrency"]

    # -- the two calls ------------------------------------------------------

    def _run(self, step: str, request, structured: bool):
        """Send one step. Variants are tried in turn while OpenRouter refuses the request
        fields (HTTP 400, 404, 422): the JSON schema, then, in "reasoning off" mode, the
        next more permissive reasoning setting. A refused request costs nothing."""
        efforts: tuple = (None,)
        if os.environ.get("VVL_OPENROUTER_REASONING") == "off" and self.capabilities.thinking_levels:
            efforts = OFF_EFFORTS
        variants = [(effort, use_schema) for effort in efforts for use_schema in ((True, False) if structured else (False,))]
        last = None
        with self._limiter.slot(self._concurrency()):
            for effort, use_schema in variants:
                options = self._step_options(step, request.thinking_level, request.max_output_tokens, effort)
                engine = self._engine(options, structured=use_schema, step=step)
                try:
                    return engine.ocr(request) if step == "ocr" else engine.parse(request)
                except EngineError as error:
                    last = error
                    if getattr(error, "upstream_status", None) not in SCHEMA_FALLBACK_STATUS:
                        break
                    logger.info("OpenRouter refused the request for %s (HTTP %s; reasoning %s, schema %s).",
                                self.slug, error.upstream_status, effort, use_schema)
        raise self._friendly(last) from None

    def ocr(self, request: OCRRequest):
        if not self.capabilities.vision:
            raise EngineError(REASON_NO_VISION, code="MODEL_NOT_AVAILABLE", status=409)
        return self._priced(self._run("ocr", request, False), request.thinking_level)

    def parse(self, request: ParseRequest):
        structured = bool(self.capabilities.structured_output and request.json_schema)
        return self._priced(self._run("parse", request, structured), request.thinking_level)


def create(ref: ModelRef, context: ResolveContext) -> OpenRouterEngine:
    api_key = _key(context)
    if not api_key:
        raise ModelNotAvailable(ref.model_id, REASON_NO_KEY)
    return OpenRouterEngine(ref, api_key=api_key, base_url=_base_url(context),
                            model=openrouter.catalog.find(ref.name), settings=getattr(context, "settings", None))
