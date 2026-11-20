"""The engine seam: what the pipeline asks of a model, and what it gets back.

The pipeline (vvl/pipeline.py) does everything that needs no model itself:
validation, resizing, TextCollage, prompt building, key alignment, sanitizing,
enrichment, and response assembly. An engine only does the two model calls:

    ocr(OCRRequest)   -> OCRResult    image + OCR prompt -> transcription text
    parse(ParseRequest) -> ParseResult  parse prompt (+ JSON schema) -> one JSON object

Engines:
  * demo (mock.py): fixed sample output.
  * Phase 2, local (local.py): the managed llama.cpp `llama-server` through the
    generic OpenAI-compatible engine (openai_compat.py): image + prompt -> text
    via /v1/chat/completions; parse with `response_format` set to the JSON
    schema in ParseRequest.json_schema.
  * OpenRouter (openrouter.py): every hosted model family through one key, also
    on the generic OpenAI-compatible engine.
  * Later: native Gemini, Anthropic, and OpenAI adapters, registered by
    provider `kind` (see vvl/engines/resolver.py).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

ALLOWED_THINKING_LEVELS = ("low", "medium", "high")
DEFAULT_THINKING_LEVEL = "low"


@dataclass(frozen=True)
class EngineCapabilities:
    vision: bool = True
    structured_output: bool = False
    # Levels the model accepts; empty means "hide the thinking-level control".
    thinking_levels: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "vision": self.vision,
            "thinking_levels": list(self.thinking_levels),
            "structured_output": self.structured_output,
        }


@dataclass
class OCRRequest:
    image_path: str            # the TextCollage image (or the original when skipped)
    prompt_text: str           # exact OCR prompt text (vvl.prompting.ocr_prompt_text)
    prompt_key: str            # OCR prompt catalog key, e.g. "verbatim_with_annotations"
    thinking_level: str = DEFAULT_THINKING_LEVEL
    max_output_tokens: int = 32768


@dataclass
class OCRResult:
    text: str
    model_id: str
    tokens_in: int = 0
    tokens_out: int = 0
    thinking_tokens: int = 0
    cost_in: float = 0.0
    cost_out: float = 0.0
    thinking_cost: float = 0.0
    rates_in: float = 0.0
    rates_out: float = 0.0
    thinking_budget_requested: str = ""

    @property
    def total_cost(self) -> float:
        # Same composition as VoucherVisionGO's OCR engine: in + out + thinking.
        return self.cost_in + self.cost_out + self.thinking_cost

    def to_packet(self) -> OrderedDict:
        """One `ocr_info[<model>]` entry, in VoucherVisionGO's key order."""
        return OrderedDict([
            ("ocr_text", self.text),
            ("cost_in", self.cost_in),
            ("cost_out", self.cost_out),
            ("total_cost", self.total_cost),
            ("rates_in", self.rates_in),
            ("rates_out", self.rates_out),
            ("tokens_in", self.tokens_in),
            ("tokens_out", self.tokens_out),
            ("thinking_tokens", self.thinking_tokens),
            ("thinking_cost", self.thinking_cost),
            ("thinking_budget_requested", self.thinking_budget_requested),
        ])


@dataclass
class ParseRequest:
    prompt_text: str                 # exact text VoucherVisionGO would send
    template: dict[str, str]         # field -> "" in prompt order (JSON_dict_structure)
    json_schema: dict                # vvl.engines.schema.json_schema_from_rules(rules)
    thinking_level: str = DEFAULT_THINKING_LEVEL
    max_output_tokens: int = 32768


@dataclass
class ParseResult:
    record: dict | None              # one JSON object, or None when the model gave none
    model_id: str
    raw_text: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    thinking_tokens: int = 0
    cost_in: float = 0.0
    cost_out: float = 0.0
    thinking_cost: float = 0.0
    thinking_budget_requested: str = ""


@runtime_checkable
class Engine(Protocol):
    model_id: str
    label: str
    capabilities: EngineCapabilities
    # Shown to the user with every result from this engine (e.g. the demo
    # notice); None for real engines.
    notice: str | None

    def ocr(self, request: OCRRequest) -> OCRResult: ...

    def parse(self, request: ParseRequest) -> ParseResult: ...


@dataclass(frozen=True)
class ModelRef:
    """A parsed VVL model id (docs/API.md section 4)."""

    model_id: str
    source: str                      # demo | local | api
    name: str                        # "mock", a build id, or a provider model name
    provider: str | None = None      # api only

    def __str__(self) -> str:
        return self.model_id


class ModelIdError(ValueError):
    """The id is malformed or names nothing VVL knows (HTTP 400)."""

    def __init__(self, model_id: str, message: str):
        super().__init__(message)
        self.model_id = model_id


class ModelNotAvailable(Exception):
    """The id names a real model that cannot run yet (HTTP 409 MODEL_NOT_AVAILABLE)."""

    def __init__(self, model_id: str, reason: str):
        super().__init__(reason)
        self.model_id = model_id
        self.reason = reason


class EngineError(RuntimeError):
    """An engine call failed. `code`/`status` map onto the error envelope."""

    def __init__(self, message: str, *, code: str = "UPSTREAM_UNAVAILABLE", status: int = 502):
        super().__init__(message)
        self.code = code
        self.status = status
        # The HTTP status the model service answered with, when there was one.
        self.upstream_status: int | None = None


@dataclass
class ResolveContext:
    """What engine factories may need. Built per request by the caller."""

    providers: object = None         # vvl.registry.loader.ProvidersFile
    local_models: object = None      # vvl.registry.loader.LocalModelsFile
    secrets: object = None           # vvl.secrets.store.SecretStore
    settings: dict = field(default_factory=dict)
    runtime: object = None           # vvl.runtime.Runtime (jobs, model store, engine), when in the app
