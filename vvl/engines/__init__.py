"""Engines: the model-calling half of the pipeline. See base.py for the seam."""

from vvl.engines.base import (
    ALLOWED_THINKING_LEVELS,
    DEFAULT_THINKING_LEVEL,
    Engine,
    EngineCapabilities,
    EngineError,
    ModelIdError,
    ModelNotAvailable,
    ModelRef,
    OCRRequest,
    OCRResult,
    ParseRequest,
    ParseResult,
    ResolveContext,
)
from vvl.engines.resolver import (
    API_UNAVAILABLE_REASON,
    DEMO_MODEL_ID,
    LOCAL_UNAVAILABLE_REASON,
    availability,
    parse_model_id,
    register_factory,
    resolve_engine,
    unregister_factory,
)
from vvl.engines.schema import json_schema_from_prompt, json_schema_from_rules

__all__ = [
    "ALLOWED_THINKING_LEVELS", "DEFAULT_THINKING_LEVEL", "Engine", "EngineCapabilities",
    "EngineError", "ModelIdError", "ModelNotAvailable", "ModelRef", "OCRRequest", "OCRResult",
    "ParseRequest", "ParseResult", "ResolveContext", "API_UNAVAILABLE_REASON", "DEMO_MODEL_ID",
    "LOCAL_UNAVAILABLE_REASON", "availability", "parse_model_id", "register_factory",
    "resolve_engine", "unregister_factory", "json_schema_from_prompt", "json_schema_from_rules",
]
