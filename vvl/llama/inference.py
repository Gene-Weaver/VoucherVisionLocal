"""Per-model engine and sampling settings, from `inference:` blocks in models_local.yaml.

A family may carry an `inference:` mapping; a build may carry its own, which
overrides the family's key by key (the `ocr` / `parse` sub-blocks and their
`sampling` mappings merge one level deeper). Anything not set falls back to
`DEFAULTS` below. This is the one place to change sampling for a model.

Keys:
  context_size         per-request context window in tokens, used when the
                       engine setting context_size is "auto" (default: the
                       build's context_default)
  image_min_tokens     lower bound on vision tokens per image (null = engine default)
  image_max_tokens     upper bound on vision tokens per image (null = engine default)
  kv_bytes_per_token   KV-cache bytes per context token, for the GPU-fit check
                       when choosing GPUs (null = vvl.system.policy value)
  request_timeout_seconds  how long one model request may take
  server_args          extra llama-server arguments (list of strings)
  ocr.max_tokens / parse.max_tokens   output token cap per request
  ocr.image_first      put the image before the prompt text (default true)
  ocr.sampling / parse.sampling       request sampling fields sent as-is
                       (temperature, top_p, top_k, min_p, repeat_penalty,
                       presence_penalty, frequency_penalty, seed, ...)
"""

from __future__ import annotations

import copy

DEFAULTS: dict = {
    "context_size": None,
    "image_min_tokens": None,
    "image_max_tokens": None,
    "kv_bytes_per_token": None,
    "request_timeout_seconds": 900,
    "server_args": [],
    "ocr": {
        "max_tokens": 4096,
        "sampling": {"temperature": 0.1, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.05},
    },
    "parse": {
        "max_tokens": 4096,
        "sampling": {"temperature": 0.1, "top_p": 0.8, "top_k": 20},
    },
}

_NESTED = ("ocr", "parse")


def _merge(base: dict, override) -> dict:
    out = copy.deepcopy(base)
    if not isinstance(override, dict):
        return out
    for key, value in override.items():
        if key in _NESTED and isinstance(value, dict):
            block = dict(out.get(key) or {})
            for sub, sub_value in value.items():
                if sub == "sampling" and isinstance(sub_value, dict):
                    sampling = dict(block.get("sampling") or {})
                    sampling.update(sub_value)
                    block["sampling"] = {k: v for k, v in sampling.items() if v is not None}
                else:
                    block[sub] = sub_value
            out[key] = block
        else:
            out[key] = copy.deepcopy(value)
    return out


def _block(obj) -> dict | None:
    value = getattr(obj, "inference", None)
    if value is None and isinstance(obj, dict):
        value = obj.get("inference")
    if hasattr(value, "model_dump"):
        value = value.model_dump(exclude_none=True)
    return value if isinstance(value, dict) else None


def settings_for(family, build) -> dict:
    """Merged inference settings for one build."""
    merged = _merge(DEFAULTS, _block(family))
    merged = _merge(merged, _block(build))
    if not merged.get("context_size"):
        merged["context_size"] = int(getattr(build, "context_default", None) or 8192)
    return merged
