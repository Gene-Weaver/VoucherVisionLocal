"""A generic OpenAI-compatible chat-completions engine.

Used for the managed llama.cpp `llama-server` and for hosted
OpenAI-compatible APIs (OpenRouter: vvl/engines/openrouter.py), so
nothing here knows about llama.cpp: it takes a base URL, an API key, a model
name, capability flags, and per-step request options (sampling fields are
sent as given, so a provider only receives the fields its config lists).

    OCR      one request: the image (base64 data URL) + the OCR prompt text
    parsing  one request: the parsing prompt, with `response_format` set to the
             prompt's JSON schema when the endpoint supports structured output

Errors become EngineError with a readable message (code `error_code`,
default ENGINE_ERROR, HTTP 502). Transient failures (connection refused or
reset, HTTP 429/502/503/504) are retried a bounded number of times; a read
timeout is not retried, because the request may still be running.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
import dataclasses
from dataclasses import dataclass, field

import httpx

from vvl.engines.base import (
    EngineCapabilities,
    EngineError,
    OCRRequest,
    OCRResult,
    ParseRequest,
    ParseResult,
)

logger = logging.getLogger(__name__)

RETRY_STATUS = {429, 502, 503, 504}
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)


@dataclass
class StepOptions:
    """Per-step request options (OCR or parsing)."""

    max_tokens: int | None = 4096
    sampling: dict = field(default_factory=dict)    # temperature, top_p, ... sent as-is
    image_first: bool = True                         # OCR: image part before the prompt text
    extra_body: dict = field(default_factory=dict)   # e.g. {"chat_template_kwargs": {...}}
    prompt_suffix: str = ""                          # appended to the end of the user text
    image_detail: str | None = None                  # OCR: "high" | "low" | "auto" on the image part


@dataclass
class OpenAICompatConfig:
    base_url: str                     # ends with /v1 (".../v1/chat/completions" is appended)
    model: str                        # the `model` field sent in requests
    model_id: str                     # VVL model id, for results
    label: str
    api_key: str | None = None
    capabilities: EngineCapabilities = field(default_factory=lambda: EngineCapabilities(
        vision=True, structured_output=True, thinking_levels=()))
    ocr: StepOptions = field(default_factory=StepOptions)
    parse: StepOptions = field(default_factory=StepOptions)
    timeout_seconds: float = 600.0
    connect_timeout_seconds: float = 10.0
    max_retries: int = 2
    retry_backoff_seconds: tuple[float, ...] = (1.0, 3.0, 6.0)
    extra_headers: dict = field(default_factory=dict)
    extra_body: dict = field(default_factory=dict)   # provider-specific request fields
    schema_name: str = "vvl_record"
    error_code: str = "ENGINE_ERROR"
    error_status: int = 502
    service_label: str = "The model server"
    # Called once per HTTP answer with (request body, HTTP status, parsed JSON or text).
    # For diagnostic logs; it must not raise.
    on_exchange: object = None


def image_data_url(path: str) -> str:
    with open(path, "rb") as handle:
        data = handle.read()
    return f"data:{_sniff_mime(data)};base64,{base64.b64encode(data).decode('ascii')}"


def _sniff_mime(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _content_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("text", None)]
        content = "".join(parts)
    if not isinstance(content, str):
        content = ""
    return _THINK_RE.sub("", content).strip()


def parse_json_reply(text: str) -> dict | None:
    """The JSON object in a reply: plain, fenced, or the outermost {...} block."""
    if not text:
        return None
    candidates = [text.strip()]
    fenced = _FENCE_RE.match(text.strip())
    if fenced:
        candidates.append(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


class OpenAICompatEngine:
    def __init__(self, config: OpenAICompatConfig, *, client: httpx.Client | None = None,
                 sleep=time.sleep):
        self.config = config
        self.model_id = config.model_id
        self.label = config.label
        self.capabilities = config.capabilities
        self.notice = None
        self._client = client
        self._sleep = sleep

    # -- transport ---------------------------------------------------------

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        headers.update(self.config.extra_headers)
        return headers

    def _error(self, message: str) -> EngineError:
        return EngineError(message, code=self.config.error_code, status=self.config.error_status)

    def _post(self, body: dict) -> dict:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        timeout = httpx.Timeout(self.config.timeout_seconds, connect=self.config.connect_timeout_seconds)
        client = self._client or httpx.Client(timeout=timeout)
        who = self.config.service_label
        try:
            attempt = 0
            while True:
                try:
                    response = client.post(url, json=body, headers=self._headers(), timeout=timeout)
                except httpx.ReadTimeout:
                    raise self._error(f"{who} did not answer within {int(self.config.timeout_seconds)} seconds.") from None
                except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError,
                        httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
                    if attempt < self.config.max_retries:
                        self._backoff(attempt)
                        attempt += 1
                        continue
                    raise self._error(f"Could not reach {who.lower() if who.startswith('The ') else who} "
                                      f"({type(exc).__name__}).") from None
                except httpx.HTTPError as exc:
                    raise self._error(f"The request to {who.lower()} failed ({type(exc).__name__}).") from None
                self._report(body, response)
                if response.status_code in RETRY_STATUS and attempt < self.config.max_retries:
                    self._backoff(attempt, response)
                    attempt += 1
                    continue
                if response.status_code != 200:
                    error = self._error(self._describe_http_error(response))
                    error.upstream_status = response.status_code
                    raise error
                try:
                    data = response.json()
                except ValueError:
                    raise self._error(f"{who} sent a reply that is not JSON.") from None
                if not isinstance(data, dict) or not isinstance(data.get("choices"), list) or not data["choices"]:
                    # Some hosted services report a failure inside an HTTP 200 reply.
                    inner = data.get("error") if isinstance(data, dict) else None
                    if isinstance(inner, dict) and inner.get("message"):
                        error = self._error(f"{who} reported an error: {str(inner['message'])[:300]}.")
                        code = inner.get("code")
                        error.upstream_status = code if isinstance(code, int) else None
                        raise error
                    raise self._error(f"{who} sent a reply without any choices.")
                return data
        finally:
            if self._client is None:
                client.close()

    def _report(self, body: dict, response: httpx.Response) -> None:
        hook = self.config.on_exchange
        if hook is None:
            return
        try:
            try:
                answer = response.json()
            except ValueError:
                answer = response.text[:4000]
            hook(body, response.status_code, answer)
        except Exception:  # noqa: BLE001 - a log must never break a request
            logger.debug("Exchange hook failed.", exc_info=True)

    def _backoff(self, attempt: int, response: httpx.Response | None = None) -> None:
        delays = self.config.retry_backoff_seconds or (1.0,)
        delay = delays[min(attempt, len(delays) - 1)]
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = min(30.0, float(retry_after))
        self._sleep(delay)

    def _describe_http_error(self, response: httpx.Response) -> str:
        who = self.config.service_label
        detail = None
        try:
            payload = response.json()
            err = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(err, dict):
                detail = err.get("message")
                if err.get("type") == "exceed_context_size_error" or "context" in str(detail).lower():
                    n_ctx = err.get("n_ctx")
                    n_prompt = err.get("n_prompt_tokens")
                    size = f" ({n_prompt} tokens needed, {n_ctx} available)" if n_ctx and n_prompt else ""
                    return (f"{who} could not fit this request into its context window{size}. Raise the context "
                            "size on the Engine tab, or use a smaller image.")
            elif isinstance(err, str):
                detail = err
        except ValueError:
            detail = None
        text = f"{who} answered HTTP {response.status_code}"
        if detail:
            text += f": {str(detail)[:300]}"
        return text + "."

    # -- requests ------------------------------------------------------------

    def _body(self, messages: list, step: StepOptions, *, response_format: dict | None = None) -> dict:
        body = {"model": self.config.model, "messages": messages, "stream": False}
        if step.max_tokens:
            body["max_tokens"] = int(step.max_tokens)
        body.update(step.sampling or {})
        body.update(self.config.extra_body or {})
        body.update(step.extra_body or {})
        if response_format is not None:
            body["response_format"] = response_format
        return body

    @staticmethod
    def _usage(data: dict) -> tuple[int, int, int]:
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        tokens_in = int(usage.get("prompt_tokens") or 0)
        tokens_out = int(usage.get("completion_tokens") or 0)
        details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) else {}
        thinking = int(details.get("reasoning_tokens") or 0)
        return tokens_in, tokens_out, thinking

    def ocr(self, request: OCRRequest) -> OCRResult:
        if not self.capabilities.vision:
            raise self._error(f"{self.label} cannot read images.")
        try:
            url = image_data_url(request.image_path)
        except OSError as exc:
            raise self._error(f"The image could not be read ({type(exc).__name__}).") from None
        step = self.config.ocr
        image_part = {"type": "image_url", "image_url": {"url": url}}
        if step.image_detail:
            image_part["image_url"]["detail"] = step.image_detail
        text_part = {"type": "text", "text": request.prompt_text + (step.prompt_suffix or "")}
        parts = [image_part, text_part] if step.image_first else [text_part, image_part]
        messages = [{"role": "user", "content": parts}]
        if request.max_output_tokens and step.max_tokens:
            step = dataclasses.replace(step, max_tokens=min(step.max_tokens, request.max_output_tokens))
        data = self._post(self._body(messages, step))
        choice = data["choices"][0] if isinstance(data["choices"][0], dict) else {}
        text = _content_text(choice.get("message") or {})
        if choice.get("finish_reason") == "length":
            logger.warning("OCR reply from %s stopped at the output token limit.", self.model_id)
        tokens_in, tokens_out, thinking = self._usage(data)
        return OCRResult(text=text, model_id=self.model_id, tokens_in=tokens_in, tokens_out=tokens_out,
                         thinking_tokens=thinking)

    def parse(self, request: ParseRequest) -> ParseResult:
        step = self.config.parse
        messages = [{"role": "user", "content": request.prompt_text + (step.prompt_suffix or "")}]
        response_format = None
        if self.capabilities.structured_output and request.json_schema:
            response_format = {"type": "json_schema", "json_schema": {
                "name": self.config.schema_name, "strict": True, "schema": request.json_schema}}
        if request.max_output_tokens and step.max_tokens:
            step = dataclasses.replace(step, max_tokens=min(step.max_tokens, request.max_output_tokens))
        data = self._post(self._body(messages, step, response_format=response_format))
        choice = data["choices"][0] if isinstance(data["choices"][0], dict) else {}
        text = _content_text(choice.get("message") or {})
        record = parse_json_reply(text)
        if record is None:
            why = " (it stopped at the output token limit)" if choice.get("finish_reason") == "length" else ""
            logger.warning("Parsing reply from %s was not a JSON object%s.", self.model_id, why)
        tokens_in, tokens_out, thinking = self._usage(data)
        return ParseResult(record=record, model_id=self.model_id, raw_text=text, tokens_in=tokens_in,
                           tokens_out=tokens_out, thinking_tokens=thinking)
