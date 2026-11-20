"""Per-step model settings: reasoning, output length, image detail, context window.

The data lives in `vvl/registry/model_settings.json` (what each model family
supports, the literal values each simple option maps to, and the model card's
own sampling defaults). The user's choices live in settings (`generation`):

    generation:
      ocr:   {reasoning: off|on, strength: low|medium|high|xhigh,
              output_length: standard|long, image_detail: standard|high}
      parse: {reasoning: off|on, strength: ..., output_length: standard|long}
      context: {<model id>: tokens}      # a model not listed uses the automatic size

Strength applies only to families whose reasoning lists `levels` (Muse), while
reasoning is on. A family with `can_turn_off: false` still reasons when
reasoning is off, so its output token limits are the reasoning-on ones.

This module is the one place that turns those into what is sent:

- `step_request(model_id, step, settings)`: per request (no engine restart):
  max_tokens, sampling (model card defaults), chat_template_kwargs, prompt suffix.
- `engine_params(model_id, settings)`: per engine start: image token limits.
- `context_rule(model_id, settings)`: how large the context window should be:
  a fixed number, or "as large as fits" (reasoning on, automatic size).
- `describe(...)`: the controls and literal values the VVL tab shows.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import struct
import threading
from pathlib import Path

from vvl import paths

logger = logging.getLogger(__name__)

CONTEXT_MIN = 8192
CONTEXT_STEP = 1024
OCR_PROMPT_TOKENS = 400        # the OCR prompt text, with room to spare
PARSE_PROMPT_TOKENS = 2000     # the parsing prompt (rules + JSON instructions), with room to spare

_lock = threading.Lock()
_cache: dict[str, tuple[float, dict]] = {}
_header_cache: dict[tuple[str, float], dict] = {}

# llama-server's names for sampling fields that differ from the model cards' names.
SAMPLING_NAMES = {"repetition_penalty": "repeat_penalty"}


# ---------------------------------------------------------------- the data file --

def settings_path() -> Path:
    return paths.registry_dir() / "model_settings.json"


def load(path: str | os.PathLike | None = None) -> dict:
    """model_settings.json, re-read when the file changes."""
    path = Path(path) if path else settings_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    key = str(path)
    with _lock:
        cached = _cache.get(key)
        if cached and cached[0] == mtime:
            return cached[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("model_settings.json could not be read (%s).", exc)
        data = {}
    with _lock:
        _cache[key] = (mtime, data)
    return data


def entry_for(model_id: str, data: dict | None = None, registry=None) -> dict | None:
    """The settings entry for a local model id (family entry, or a per-build entry)."""
    if not isinstance(model_id, str) or not model_id.startswith("local:"):
        return None
    data = data if data is not None else load()
    build_id = model_id.split(":", 1)[1]
    family_id = None
    if registry is None:
        try:
            from vvl.registry.loader import load_local_models

            registry = load_local_models()
        except Exception:  # noqa: BLE001
            registry = None
    if registry is not None:
        found = registry.find_build(build_id)
        if found:
            family_id = found[0].id
    families = data.get("families") or {}
    candidates = [family_id] if family_id else list(families)
    for fid in candidates:
        fam = families.get(fid)
        if not isinstance(fam, dict):
            continue
        builds = fam.get("builds")
        if isinstance(builds, dict):
            if build_id in builds:
                entry = builds[build_id]
                if isinstance(entry, dict) and entry.get("same_as") in builds:
                    entry = builds[entry["same_as"]]
                return entry
            continue
        if family_id:
            return fam
    return None


# ------------------------------------------------------------------- choices --

def choices(settings: dict) -> dict:
    gen = (settings or {}).get("generation") or {}
    defaults = load().get("defaults") or {}
    out = {}
    for step in ("ocr", "parse"):
        merged = dict(defaults.get(step) or {})
        merged.update(gen.get(step) or {})
        out[step] = merged
    out["context"] = dict(gen.get("context") or {})
    return out


def steps_using(model_id: str, settings: dict) -> list[str]:
    sel = (settings or {}).get("selection") or {}
    steps = []
    if model_id in (sel.get("ocr") or []):
        steps.append("ocr")
    if sel.get("parse") == model_id:
        steps.append("parse")
    return steps


def reasoning_supported(entry: dict | None) -> bool:
    return bool(entry and (entry.get("reasoning") or {}).get("supported"))


def image_detail_supported(entry: dict | None) -> bool:
    return bool(entry and (entry.get("image_detail") or {}).get("supported"))


def reasoning_state(model_id: str, step: str, settings: dict, entry: dict | None = None) -> str | None:
    """"on" / "off" for a model that can reason, None for one that cannot."""
    entry = entry if entry is not None else entry_for(model_id)
    if not reasoning_supported(entry):
        return None
    return "on" if choices(settings)[step].get("reasoning") == "on" else "off"


def can_turn_off(entry: dict | None) -> bool:
    """False for a model that keeps reasoning (at its lowest level) when reasoning is off."""
    return ((entry or {}).get("reasoning") or {}).get("can_turn_off", True) is not False


def token_reasoning(entry: dict | None, reasoning: str | None) -> str | None:
    """The reasoning state that sets the output token limit: "on" whenever the model reasons."""
    if reasoning == "off" and not can_turn_off(entry):
        return "on"
    return reasoning


def strength_levels(entry: dict | None) -> list[str]:
    """Reasoning strength levels, lowest first; [] for a model without levels."""
    if not reasoning_supported(entry):
        return []
    levels = (entry.get("reasoning") or {}).get("levels")
    return [str(level) for level in levels] if isinstance(levels, list) else []


def strength_for(entry: dict | None, chosen: dict) -> str | None:
    """The strength level to send while reasoning is on (the choice, else the model default)."""
    levels = strength_levels(entry)
    if not levels:
        return None
    if chosen.get("strength") in levels:
        return chosen["strength"]
    default = (entry.get("reasoning") or {}).get("default_level")
    return default if default in levels else levels[-1]


def _fill_strength(value, strength: str):
    if isinstance(value, dict):
        return {k: _fill_strength(v, strength) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill_strength(v, strength) for v in value]
    if isinstance(value, str) and "{strength}" in value:
        return value.replace("{strength}", strength)
    return value


def max_tokens(step: str, output_length: str, reasoning: str | None, data: dict | None = None) -> int:
    data = data if data is not None else load()
    table = data.get("output_length_tokens") or {}
    block = table.get("reasoning_on" if reasoning == "on" else "reasoning_off") or {}
    row = block.get(output_length) or block.get("standard") or {}
    fallback = {"ocr": 3072, "parse": 2048}[step]
    return int(row.get(step) or fallback)


def _sampling(entry: dict | None, reasoning: str | None) -> dict:
    defaults = (entry or {}).get("sampling_defaults") or {}
    values = defaults.get("values")
    if values is None:
        by = defaults.get("values_by_reasoning") or {}
        values = by.get(reasoning or "off") or by.get("off") or {}
    return {SAMPLING_NAMES.get(k, k): v for k, v in (values or {}).items()}


def step_request(model_id: str, step: str, settings: dict, registry=None) -> dict:
    """What one OCR or parsing request to `model_id` sends, from the user's choices.

    {"max_tokens": int, "sampling": {...}, "extra_body": {...}, "prompt_suffix": str,
     "reasoning": "on" | "off" | None, "strength": level | None}
    """
    data = load()
    entry = entry_for(model_id, data, registry)
    chosen = choices(settings)[step]
    reasoning = reasoning_state(model_id, step, settings, entry)
    levels = strength_levels(entry)
    strength = None
    if levels:
        strength = strength_for(entry, chosen) if reasoning == "on" else levels[0]
    out = {
        "max_tokens": max_tokens(step, chosen.get("output_length", "standard"),
                                 token_reasoning(entry, reasoning), data),
        "sampling": _sampling(entry, reasoning),
        "extra_body": {},
        "prompt_suffix": "",
        "reasoning": reasoning,
        "strength": strength,
    }
    if reasoning is not None:
        literal = ((entry.get("reasoning") or {}).get("literal") or {}).get(reasoning) or {}
        request = literal.get("request") or {}
        if request:
            out["extra_body"] = _fill_strength(copy.deepcopy(request), strength or "")
        if literal.get("prompt_suffix"):
            out["prompt_suffix"] = str(literal["prompt_suffix"])
        # Extra request fields for one strength level (reasoning.level_requests), e.g. a
        # reasoning budget for the lowest level of a model that has no levels of its own.
        if reasoning == "on" and strength:
            extra = ((entry.get("reasoning") or {}).get("level_requests") or {}).get(strength) or {}
            for key, value in copy.deepcopy(extra).items():
                if isinstance(value, dict) and isinstance(out["extra_body"].get(key), dict):
                    out["extra_body"][key].update(value)
                else:
                    out["extra_body"][key] = value
    return out


def engine_params(model_id: str, settings: dict, registry=None) -> dict:
    """Engine start options that follow the user's choices: image token limits."""
    data = load()
    entry = entry_for(model_id, data, registry)
    if not image_detail_supported(entry):
        return {}
    level = choices(settings)["ocr"].get("image_detail", "standard")
    tokens = (data.get("image_detail_tokens") or {}).get(level) or {}
    return {k: int(v) for k, v in tokens.items() if k in ("image_min_tokens", "image_max_tokens") and v}


# ------------------------------------------------------------------- context --

def _round_up(tokens: int) -> int:
    return -(-int(tokens) // CONTEXT_STEP) * CONTEXT_STEP


def context_needed(model_id: str, settings: dict, registry=None, image_max_tokens: int | None = None) -> int:
    """Smallest context window per request that holds the chosen settings (at least 8,192)."""
    data = load()
    steps = steps_using(model_id, settings) or ["ocr", "parse"]
    need = CONTEXT_MIN
    params = engine_params(model_id, settings, registry)
    image = params.get("image_max_tokens") or image_max_tokens or 4096
    entry = entry_for(model_id, data, registry)
    ocr_out = max_tokens("ocr", choices(settings)["ocr"].get("output_length", "standard"),
                         token_reasoning(entry, reasoning_state(model_id, "ocr", settings, entry)), data)
    if "ocr" in steps:
        need = max(need, image + OCR_PROMPT_TOKENS + ocr_out)
    if "parse" in steps:
        parse_out = max_tokens("parse", choices(settings)["parse"].get("output_length", "standard"),
                               token_reasoning(entry, reasoning_state(model_id, "parse", settings, entry)),
                               data)
        need = max(need, PARSE_PROMPT_TOKENS + ocr_out + parse_out)
    return _round_up(need)


def context_rule(model_id: str, settings: dict, registry=None, model_max: int | None = None,
                 image_max_tokens: int | None = None) -> dict:
    """How large the context window should be.

    {"mode": "fixed" | "auto" | "auto_max", "tokens": int | None, "needed": int, "max": int | None}
      fixed     the user chose a size (clamped to the model's maximum)
      auto      the smallest size that holds the chosen settings
      auto_max  reasoning is on: as large as fits in memory, up to the model's maximum
    """
    needed = context_needed(model_id, settings, registry, image_max_tokens)
    chosen = choices(settings)["context"].get(model_id)
    if chosen is None:
        # A size set for every model on the Engine tab applies when none is set for this model.
        engine_size = ((settings or {}).get("engine") or {}).get("context_size")
        if isinstance(engine_size, int) and not isinstance(engine_size, bool):
            chosen = engine_size
    if model_max:
        needed = min(needed, model_max)
    if isinstance(chosen, int):
        tokens = max(CONTEXT_MIN, chosen)
        if model_max:
            tokens = min(tokens, model_max)
        return {"mode": "fixed", "tokens": tokens, "needed": needed, "max": model_max}
    entry = entry_for(model_id, registry=registry)
    reasoning_on = any(reasoning_state(model_id, step, settings, entry) == "on"
                       for step in (steps_using(model_id, settings) or ["ocr", "parse"]))
    if reasoning_on:
        return {"mode": "auto_max", "tokens": None, "needed": needed, "max": model_max}
    return {"mode": "auto", "tokens": needed, "needed": needed, "max": model_max}


# --------------------------------------------------------------- GGUF headers --

_GGUF_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_GGUF_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def gguf_header(path: str | os.PathLike) -> dict:
    """Scalar metadata of a GGUF file (arrays are skipped). Cached by path and modification time."""
    path = Path(path)
    try:
        key = (str(path), path.stat().st_mtime)
    except OSError:
        return {}
    with _lock:
        if key in _header_cache:
            return _header_cache[key]
    out: dict = {}
    try:
        with open(path, "rb") as handle:
            read = handle.read
            if read(4) != b"GGUF":
                return {}
            struct.unpack("<I", read(4))
            _tensors, count = struct.unpack("<QQ", read(16))

            def text() -> str:
                (length,) = struct.unpack("<Q", read(8))
                return read(length).decode("utf-8", "replace")

            for _ in range(count):
                name = text()
                (kind,) = struct.unpack("<I", read(4))
                if kind == 8:
                    value = text()
                    if len(value) > 4096:
                        value = None      # chat templates etc. are not needed here
                elif kind == 9:
                    (inner,) = struct.unpack("<I", read(4))
                    (n,) = struct.unpack("<Q", read(8))
                    if inner == 8:
                        for _ in range(n):
                            (length,) = struct.unpack("<Q", read(8))
                            handle.seek(length, 1)
                    else:
                        handle.seek(_GGUF_SIZES.get(inner, 0) * n, 1)
                    continue
                else:
                    (value,) = struct.unpack(_GGUF_FMT[kind], read(_GGUF_SIZES[kind]))
                if value is not None:
                    out[name] = value
    except (OSError, struct.error, KeyError) as exc:
        logger.info("GGUF header of %s could not be read (%s).", path.name, exc)
        return {}
    with _lock:
        _header_cache[key] = out
    return out


def model_max_context(models_dir: str, build) -> int | None:
    """The model's own maximum context window, from its file header."""
    from vvl.local_models import file_path

    model_file = next((f for f in build.files if f.role == "model"), None)
    if model_file is None:
        return None
    header = gguf_header(file_path(models_dir, build, model_file))
    arch = header.get("general.architecture")
    value = header.get(f"{arch}.context_length") if arch else None
    return int(value) if isinstance(value, int) and value > 0 else None


# ------------------------------------------------------------------ describe --

def describe_step(model_id: str, step: str, settings: dict, registry=None) -> dict:
    """The controls one model offers for one step, and what the current choice sends."""
    data = load()
    entry = entry_for(model_id, data, registry)
    chosen = choices(settings)[step]
    reasoning = reasoning_state(model_id, step, settings, entry)
    controls = {}
    if reasoning_supported(entry):
        block = entry.get("reasoning") or {}
        controls["reasoning"] = {
            "options": ["off", "on"],
            "model_default": block.get("model_default"),
            "can_turn_off": can_turn_off(entry),
            "literal": copy.deepcopy(block.get("literal") or {}),
            "note": block.get("_note"),
        }
        levels = strength_levels(entry)
        if levels:
            controls["strength"] = {"options": levels, "default": block.get("default_level")}
    controls["output_length"] = {
        "options": ["standard", "long"],
        "max_tokens": {length: max_tokens(step, length, token_reasoning(entry, reasoning), data)
                       for length in ("standard", "long")},
    }
    if step == "ocr" and image_detail_supported(entry):
        controls["image_detail"] = {
            "options": ["standard", "high"],
            "tokens": copy.deepcopy(data.get("image_detail_tokens") or {}),
            "needs_engine_restart": True,
        }
    request = step_request(model_id, step, settings, registry)
    sent = {"max_tokens": request["max_tokens"], **request["sampling"]}
    sent.update(request["extra_body"])
    if request["prompt_suffix"]:
        sent["prompt_suffix"] = request["prompt_suffix"]
    if step == "ocr":
        sent.update(engine_params(model_id, settings, registry))
    return {"controls": controls, "values": dict(chosen), "sent": sent}
