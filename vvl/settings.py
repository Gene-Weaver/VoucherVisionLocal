"""Machine-local settings (settings.local.yaml from source, else <VVL_HOME>/settings.yaml).

Only the allowlisted keys below can be read or changed through the API. Keys
found in the file that the app does not manage are preserved on save. Writes
are atomic (temp file + rename).

Shape (docs/API.md, GET/PUT /v1/settings):
    models_dir: absolute path
    selection:  {ocr: [model_id, ...], parse: model_id}
    enrichment: {elevation: bool, wfo_source: off | vvgo | local}
    ui:         {theme: light | dark | system}
    engine:     {variant: "auto" | <variant id>, context_size: "auto" | int,
                 parallel: "auto" | int (request slots per engine),
                 gpu_devices: "auto" | [int, ...],
                 idle_stop_minutes: int (0 = never),
                 staged_batches: bool (batches run all OCR first, then all parsing)}
"""

from __future__ import annotations

import copy
import datetime
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable

import yaml

from vvl import paths
from vvl.fileutil import atomic_write_text

logger = logging.getLogger(__name__)

DEMO_MODEL_ID = "demo:mock"
WFO_SOURCES = ("off", "vvgo", "local")
THEMES = ("light", "dark", "system")
MAX_OCR_ENGINES = 4

ALLOWED_KEYS = ("models_dir", "selection", "enrichment", "ui", "engine", "generation", "openrouter")
NESTED_KEYS = {
    "selection": ("ocr", "parse"),
    "enrichment": ("elevation", "wfo_source"),
    "ui": ("theme",),
    "engine": ("variant", "context_size", "parallel", "gpu_devices", "idle_stop_minutes", "staged_batches"),
    "generation": ("ocr", "parse", "context"),
    "openrouter": ("models", "concurrency", "private_only"),
}

# OpenRouter (vvl/openrouter.py): the models added from the gallery, and how many
# requests run at once.
OPENROUTER_CONCURRENCY_MIN = 1
OPENROUTER_CONCURRENCY_MAX = 4
OPENROUTER_MAX_MODELS = 200
_OPENROUTER_SLUG = re.compile(r"^[A-Za-z0-9~][A-Za-z0-9._:/@+~-]{0,199}$")

# Per-step model settings chosen on the VVL tab (vvl/generation.py, model_settings.json).
STRENGTH_LEVELS = ("low", "medium", "high", "xhigh")
GENERATION_STEP_KEYS = {
    "ocr": {"reasoning": ("off", "on"), "strength": STRENGTH_LEVELS, "output_length": ("standard", "long"),
            "image_detail": ("standard", "high")},
    "parse": {"reasoning": ("off", "on"), "strength": STRENGTH_LEVELS, "output_length": ("standard", "long")},
}
MAX_CONTEXT_OVERRIDES = 64

# Engine limits (docs/API.md section 9, `PUT /v1/settings` engine object).
ENGINE_CONTEXT_MIN = 8192
ENGINE_CONTEXT_MAX = 262144
ENGINE_PARALLEL_MAX = 16
ENGINE_IDLE_MAX_MINUTES = 7 * 24 * 60
ENGINE_MAX_GPU_INDEX = 63

DEFAULTS: dict[str, Any] = {
    "models_dir": None,  # None means <VVL_HOME>/models
    "selection": {"ocr": [DEMO_MODEL_ID], "parse": DEMO_MODEL_ID},
    "enrichment": {"elevation": True, "wfo_source": "off"},
    "ui": {"theme": "dark"},
    "engine": {"variant": "auto", "context_size": "auto", "parallel": "auto", "gpu_devices": "auto",
               "idle_stop_minutes": 15, "staged_batches": False},
    # context: {model id: number of tokens}; a model that is not listed uses the automatic size.
    "generation": {
        "ocr": {"reasoning": "off", "strength": "high", "output_length": "standard", "image_detail": "standard"},
        "parse": {"reasoning": "off", "strength": "high", "output_length": "standard"},
        "context": {},
    },
    "openrouter": {"models": [], "concurrency": 4, "private_only": False},
}

_HEADER = """\
# Machine-local settings for this computer (gitignored).
# Written by VoucherVisionLocal; edits made while the app is running may be
# overwritten. Keep models_dir OUTSIDE any synced folder (Dropbox, OneDrive,
# iCloud): model files are tens of GB.
"""

_lock = threading.RLock()


class SettingsError(ValueError):
    """A settings update was rejected. `details` is safe to return to the client."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


def _read_file(path: Path) -> dict:
    """Raw mapping from the settings file; {} when it is missing or unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Settings file %s could not be read (%s); using defaults.", path, type(exc).__name__)
        return {}
    return data if isinstance(data, dict) else {}


def _merge(defaults: dict, stored: dict) -> dict:
    result = copy.deepcopy(defaults)
    for key in ALLOWED_KEYS:
        if key not in stored:
            continue
        value = stored[key]
        if key in NESTED_KEYS:
            if isinstance(value, dict):
                for sub in NESTED_KEYS[key]:
                    if sub in value:
                        result[key][sub] = value[sub]
        else:
            result[key] = value
    return result


def _sanitize_loaded(settings: dict) -> dict:
    """Fall back to defaults for any stored value that does not validate."""
    clean = copy.deepcopy(settings)
    try:
        clean["models_dir"] = _validate_models_dir(clean.get("models_dir")) if clean.get("models_dir") else None
    except SettingsError:
        clean["models_dir"] = None
    sel = clean.get("selection") or {}
    ocr = sel.get("ocr")
    if isinstance(ocr, str):
        ocr = [ocr]
    if not (isinstance(ocr, list) and ocr and all(isinstance(m, str) and m for m in ocr)):
        ocr = list(DEFAULTS["selection"]["ocr"])
    parse = sel.get("parse")
    if not (isinstance(parse, str) and parse):
        parse = DEFAULTS["selection"]["parse"]
    clean["selection"] = {"ocr": ocr[:MAX_OCR_ENGINES], "parse": parse}
    enr = clean.get("enrichment") or {}
    if not isinstance(enr.get("elevation"), bool):
        enr["elevation"] = DEFAULTS["enrichment"]["elevation"]
    if enr.get("wfo_source") not in WFO_SOURCES:
        enr["wfo_source"] = DEFAULTS["enrichment"]["wfo_source"]
    clean["enrichment"] = enr
    ui = clean.get("ui") or {}
    if ui.get("theme") not in THEMES:
        ui["theme"] = DEFAULTS["ui"]["theme"]
    clean["ui"] = ui
    eng = clean.get("engine") or {}
    defaults = DEFAULTS["engine"]
    for key in NESTED_KEYS["engine"]:
        try:
            eng[key] = _validate_engine_value(key, eng.get(key, defaults[key]))
        except SettingsError:
            eng[key] = defaults[key]
    clean["engine"] = eng
    gen = clean.get("generation") if isinstance(clean.get("generation"), dict) else {}
    out = {}
    for step in ("ocr", "parse"):
        try:
            out[step] = _validate_generation_step(step, gen.get(step) or {}, fill=True)
        except SettingsError:
            out[step] = dict(DEFAULTS["generation"][step])
    try:
        out["context"] = _validate_context_map(gen.get("context") or {})
    except SettingsError:
        out["context"] = {}
    clean["generation"] = out
    block = clean.get("openrouter") if isinstance(clean.get("openrouter"), dict) else {}
    orouter = {}
    for key in NESTED_KEYS["openrouter"]:
        try:
            orouter[key] = _validate_openrouter_value(key, block.get(key, DEFAULTS["openrouter"][key]))
        except SettingsError:
            orouter[key] = copy.deepcopy(DEFAULTS["openrouter"][key])
    clean["openrouter"] = orouter
    return clean


def _validate_openrouter_value(key: str, value: Any):
    if key == "concurrency":
        if not (_is_int(value) and OPENROUTER_CONCURRENCY_MIN <= value <= OPENROUTER_CONCURRENCY_MAX):
            raise SettingsError(
                f"openrouter.concurrency must be a number from {OPENROUTER_CONCURRENCY_MIN} to "
                f"{OPENROUTER_CONCURRENCY_MAX}.", {"field": "openrouter.concurrency"})
        return int(value)
    if key == "private_only":
        if not isinstance(value, bool):
            raise SettingsError("openrouter.private_only must be true or false.",
                                {"field": "openrouter.private_only"})
        return value
    if key == "models":
        if not isinstance(value, list) or len(value) > OPENROUTER_MAX_MODELS:
            raise SettingsError(f"openrouter.models must be a list of at most {OPENROUTER_MAX_MODELS} model names.",
                                {"field": "openrouter.models"})
        out = []
        for slug in value:
            if not (isinstance(slug, str) and _OPENROUTER_SLUG.match(slug)):
                raise SettingsError("openrouter.models has a name that is not an OpenRouter model.",
                                    {"field": "openrouter.models"})
            if slug not in out:
                out.append(slug)
        return out
    raise SettingsError("Unknown openrouter key.", {"unknown_keys": [key]})


def _validate_generation_step(step: str, value: Any, *, fill: bool = False) -> dict:
    """One step's choices ({"reasoning": "off", ...}). With `fill`, missing keys take the defaults."""
    if not isinstance(value, dict):
        raise SettingsError(f"generation.{step} must be an object.", {"field": f"generation.{step}"})
    allowed = GENERATION_STEP_KEYS[step]
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise SettingsError(f"Unknown generation.{step} keys.", {"unknown_keys": unknown})
    out = dict(DEFAULTS["generation"][step]) if fill else {}
    for key, options in allowed.items():
        if key in value:
            if value[key] not in options:
                raise SettingsError(f"generation.{step}.{key} must be one of: {', '.join(options)}.",
                                    {"field": f"generation.{step}.{key}", "allowed": list(options)})
            out[key] = value[key]
    return out


def _validate_context_map(value: Any) -> dict:
    """{model id: tokens}. "auto" (or null) removes a model's entry."""
    if not isinstance(value, dict):
        raise SettingsError("generation.context must be an object of model id -> tokens.",
                            {"field": "generation.context"})
    if len(value) > MAX_CONTEXT_OVERRIDES:
        raise SettingsError("Too many context sizes.", {"field": "generation.context"})
    out = {}
    for model_id, tokens in value.items():
        if not (isinstance(model_id, str) and model_id.startswith("local:") and len(model_id) <= 200):
            raise SettingsError("generation.context keys must be local model ids.", {"field": "generation.context"})
        if tokens in (None, "auto"):
            continue
        if not (_is_int(tokens) and ENGINE_CONTEXT_MIN <= tokens <= ENGINE_CONTEXT_MAX):
            raise SettingsError(
                f"A context size must be \"auto\" or a number from {ENGINE_CONTEXT_MIN} to {ENGINE_CONTEXT_MAX}.",
                {"field": f"generation.context.{model_id}"})
        out[model_id] = int(tokens)
    return out


def load(path: Path | None = None) -> dict:
    """Current settings with defaults applied. `models_dir` is always a resolved string."""
    path = path or paths.settings_path()
    with _lock:
        settings = _sanitize_loaded(_merge(DEFAULTS, _read_file(path)))
    if not settings["models_dir"]:
        settings["models_dir"] = str(paths.default_models_dir())
    return settings


def models_dir(path: Path | None = None) -> str:
    return load(path)["models_dir"]


def _validate_models_dir(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SettingsError("models_dir must be a non-empty string.", {"field": "models_dir"})
    expanded = os.path.expanduser(value.strip())
    if not os.path.isabs(expanded):
        raise SettingsError("models_dir must be an absolute path.", {"field": "models_dir"})
    if "\x00" in expanded or len(expanded) > 1024:
        raise SettingsError("models_dir is not a valid path.", {"field": "models_dir"})
    return os.path.normpath(expanded)


def _validate_selection(value: Any, validate_ids: Callable[[list[str], str], None] | None) -> dict:
    if not isinstance(value, dict):
        raise SettingsError("selection must be an object with 'ocr' and/or 'parse'.", {"field": "selection"})
    unknown = sorted(set(value) - set(NESTED_KEYS["selection"]))
    if unknown:
        raise SettingsError("Unknown selection keys.", {"unknown_keys": unknown})
    out = {}
    if "ocr" in value:
        ocr = value["ocr"]
        if isinstance(ocr, str):
            ocr = [ocr]
        if not isinstance(ocr, list) or not ocr or not all(isinstance(m, str) and m.strip() for m in ocr):
            raise SettingsError("selection.ocr must be a non-empty list of model ids.", {"field": "selection.ocr"})
        ocr = [m.strip() for m in ocr]
        if len(set(ocr)) != len(ocr):
            raise SettingsError("selection.ocr lists the same model twice.", {"field": "selection.ocr"})
        if len(ocr) > MAX_OCR_ENGINES:
            raise SettingsError(f"selection.ocr allows at most {MAX_OCR_ENGINES} models.", {"field": "selection.ocr"})
        if validate_ids:
            validate_ids(ocr, "ocr")
        out["ocr"] = ocr
    if "parse" in value:
        parse = value["parse"]
        if not isinstance(parse, str) or not parse.strip():
            raise SettingsError("selection.parse must be a model id.", {"field": "selection.parse"})
        parse = parse.strip()
        if validate_ids:
            validate_ids([parse], "parse")
        out["parse"] = parse
    return out


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_engine_value(key: str, value: Any):
    """One engine setting, normalized. Raises SettingsError."""
    field = f"engine.{key}"
    if key == "variant":
        if value == "auto":
            return value
        if isinstance(value, str) and value.strip() and len(value) <= 64 and value.replace("-", "").replace("_", "").isalnum():
            return value.strip()
        raise SettingsError("engine.variant must be \"auto\" or a build id.", {"field": field})
    if key == "context_size":
        if value == "auto":
            return value
        if _is_int(value) and ENGINE_CONTEXT_MIN <= value <= ENGINE_CONTEXT_MAX:
            return value
        raise SettingsError(
            f"engine.context_size must be \"auto\" or a number from {ENGINE_CONTEXT_MIN} to {ENGINE_CONTEXT_MAX}.",
            {"field": field})
    if key == "parallel":
        if value == "auto":
            return value
        if _is_int(value) and 1 <= value <= ENGINE_PARALLEL_MAX:
            return value
        raise SettingsError(f"engine.parallel must be \"auto\" or a number from 1 to {ENGINE_PARALLEL_MAX}.",
                            {"field": field})
    if key == "staged_batches":
        if isinstance(value, bool):
            return value
        raise SettingsError("engine.staged_batches must be true or false.", {"field": field})
    if key == "gpu_devices":
        if value == "auto":
            return value
        if (isinstance(value, list) and value and len(value) <= 16
                and all(_is_int(v) and 0 <= v <= ENGINE_MAX_GPU_INDEX for v in value)
                and len(set(value)) == len(value)):
            return list(value)
        raise SettingsError("engine.gpu_devices must be \"auto\" or a list of GPU indexes.", {"field": field})
    if key == "idle_stop_minutes":
        if _is_int(value) and 0 <= value <= ENGINE_IDLE_MAX_MINUTES:
            return value
        raise SettingsError(
            f"engine.idle_stop_minutes must be a number from 0 (never) to {ENGINE_IDLE_MAX_MINUTES}.",
            {"field": field})
    raise SettingsError("Unknown engine key.", {"unknown_keys": [key]})


def validate_update(update: Any, validate_ids: Callable[[list[str], str], None] | None = None) -> dict:
    """Check a partial update and return the normalized changes. Raises SettingsError."""
    if not isinstance(update, dict):
        raise SettingsError("The request body must be a JSON object.")
    unknown = sorted(set(update) - set(ALLOWED_KEYS))
    if unknown:
        raise SettingsError(
            f"Unknown settings key(s): {', '.join(unknown)}.",
            {"unknown_keys": unknown, "allowed_keys": list(ALLOWED_KEYS)},
        )
    changes: dict[str, Any] = {}
    if "models_dir" in update:
        changes["models_dir"] = _validate_models_dir(update["models_dir"])
    if "selection" in update:
        changes["selection"] = _validate_selection(update["selection"], validate_ids)
    if "enrichment" in update:
        enr = update["enrichment"]
        if not isinstance(enr, dict):
            raise SettingsError("enrichment must be an object.", {"field": "enrichment"})
        unknown = sorted(set(enr) - set(NESTED_KEYS["enrichment"]))
        if unknown:
            raise SettingsError("Unknown enrichment keys.", {"unknown_keys": unknown})
        out = {}
        if "elevation" in enr:
            if not isinstance(enr["elevation"], bool):
                raise SettingsError("enrichment.elevation must be true or false.", {"field": "enrichment.elevation"})
            out["elevation"] = enr["elevation"]
        if "wfo_source" in enr:
            if enr["wfo_source"] not in WFO_SOURCES:
                raise SettingsError(
                    f"enrichment.wfo_source must be one of: {', '.join(WFO_SOURCES)}.",
                    {"field": "enrichment.wfo_source", "allowed": list(WFO_SOURCES)},
                )
            out["wfo_source"] = enr["wfo_source"]
        changes["enrichment"] = out
    if "ui" in update:
        ui = update["ui"]
        if not isinstance(ui, dict):
            raise SettingsError("ui must be an object.", {"field": "ui"})
        unknown = sorted(set(ui) - set(NESTED_KEYS["ui"]))
        if unknown:
            raise SettingsError("Unknown ui keys.", {"unknown_keys": unknown})
        out = {}
        if "theme" in ui:
            if ui["theme"] not in THEMES:
                raise SettingsError(
                    f"ui.theme must be one of: {', '.join(THEMES)}.",
                    {"field": "ui.theme", "allowed": list(THEMES)},
                )
            out["theme"] = ui["theme"]
        changes["ui"] = out
    if "engine" in update:
        eng = update["engine"]
        if not isinstance(eng, dict):
            raise SettingsError("engine must be an object.", {"field": "engine"})
        unknown = sorted(set(eng) - set(NESTED_KEYS["engine"]))
        if unknown:
            raise SettingsError("Unknown engine keys.", {"unknown_keys": unknown})
        changes["engine"] = {key: _validate_engine_value(key, value) for key, value in eng.items()}
    if "generation" in update:
        gen = update["generation"]
        if not isinstance(gen, dict):
            raise SettingsError("generation must be an object.", {"field": "generation"})
        unknown = sorted(set(gen) - set(NESTED_KEYS["generation"]))
        if unknown:
            raise SettingsError("Unknown generation keys.", {"unknown_keys": unknown})
        out = {}
        for step in ("ocr", "parse"):
            if step in gen:
                out[step] = _validate_generation_step(step, gen[step], fill=True)
        if "context" in gen:
            out["context"] = _validate_context_map(gen["context"])
        changes["generation"] = out
    if "openrouter" in update:
        block = update["openrouter"]
        if not isinstance(block, dict):
            raise SettingsError("openrouter must be an object.", {"field": "openrouter"})
        unknown = sorted(set(block) - set(NESTED_KEYS["openrouter"]))
        if unknown:
            raise SettingsError("Unknown openrouter keys.", {"unknown_keys": unknown})
        changes["openrouter"] = {key: _validate_openrouter_value(key, value) for key, value in block.items()}
    return changes


def save(changes: dict, path: Path | None = None) -> dict:
    """Apply validated `changes` (from validate_update) and write the file atomically."""
    path = path or paths.settings_path()
    with _lock:
        raw = _read_file(path)
        if path.exists() and not raw:
            _set_aside_unreadable(path)
        for key, value in changes.items():
            if key in NESTED_KEYS:
                current = raw.get(key) if isinstance(raw.get(key), dict) else {}
                current = dict(current)
                current.update(value)
                raw[key] = current
            else:
                raw[key] = value
        text = _HEADER + "\n" + yaml.safe_dump(raw, sort_keys=False, allow_unicode=True, default_flow_style=False)
        atomic_write_text(path, text, mode=0o644)
        logger.info("Settings saved (%s).", ", ".join(sorted(changes)) or "no changes")
    return load(path)


def _set_aside_unreadable(path: Path) -> None:
    """Keep a copy of a settings file that could not be parsed before replacing it."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    if _only_comments(text):
        return
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.unreadable-{stamp}")
    try:
        os.replace(path, backup)
        logger.warning("Settings file could not be parsed; kept it as %s.", backup.name)
    except OSError:
        pass


def _only_comments(text: str) -> bool:
    return all(not line.strip() or line.lstrip().startswith("#") for line in text.splitlines())


def warnings_for(settings: dict, changed: dict | None = None) -> list[dict]:
    """Advisory warnings for the PUT /v1/settings response."""
    warnings = []
    models_dir = settings.get("models_dir")
    if models_dir:
        hint = paths.synced_folder_hint(models_dir)
        if hint:
            warnings.append({
                "code": "MODELS_DIR_SYNCED",
                "message": (
                    f"The models folder is inside a synced folder ({hint}). Model files are tens "
                    "of GB; choose a folder outside Dropbox, OneDrive, or iCloud."
                ),
            })
        if changed and "models_dir" in changed and not os.path.isdir(models_dir):
            warnings.append({
                "code": "MODELS_DIR_MISSING",
                "message": "The models folder does not exist yet. It is created on first download.",
            })
    return warnings
