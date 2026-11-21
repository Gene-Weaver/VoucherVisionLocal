"""Prompt templates on disk: built-in (read-only) and user (imported).

Names are bare file names matching `^[A-Za-z0-9._-]+\\.ya?ml$` and must resolve
inside their folder (docs/API.md section 2.7). Validation of imported prompts
uses the rules VoucherVisionGO applies to user prompt uploads
(`_validate_user_prompt_yaml`), and `extract_prompt_info` /
`extract_prompt_details` / the text formatters are VoucherVisionGO's helpers
for the legacy `GET /prompts` route.
"""

from __future__ import annotations

import json
import logging
import os
import re
import textwrap
from pathlib import Path

import yaml

from vvl import paths
from vvl.errors import ApiError
from vvl.fileutil import atomic_write_bytes

logger = logging.getLogger(__name__)

NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+\.ya?ml$")
PROMPT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
SOURCES = ("builtin", "user")
DEFAULT_PROMPT = "SLTPvM_full.yaml"
USER_PROMPTS_MAX_BYTES = 256 * 1024

CANONICAL_JSON_FORMATTING_INSTRUCTIONS = (
    "This section provides rules for formatting each JSON value organized by the JSON key. "
    "Correct minor typos introduced by OCR errors."
)

# Top-level YAML keys VoucherVisionGO requires uploaded user prompts to define.
REQUIRED_USER_PROMPT_KEYS = (
    "prompt_name",
    "prompt_version",
    "prompt_author",
    "prompt_description",
    "instructions",
    "rules",
    "mapping",
    "json_formatting_instructions",
)

REQUIRED_MAPPING_CATEGORIES = ("GEOGRAPHY", "TAXONOMY", "COLLECTING")
MAX_MAPPING_CATEGORIES = 6
CATEGORY_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
FIELD_KEY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class UserPromptValidationError(ValueError):
    """Raised when a prompt YAML fails schema validation."""


class _StrictYamlLoader(yaml.SafeLoader):
    """YAML SafeLoader that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader: yaml.SafeLoader, node, deep: bool = False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise UserPromptValidationError(f"All keys must be text (found {type(key).__name__}).")
        if key in mapping:
            raise UserPromptValidationError(f"Duplicate key '{key}'.")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictYamlLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _assert_strict_json_compatible(data: dict) -> None:
    """Verify `data` serializes cleanly as strict JSON (no dates, NaN, Infinity, non-string keys)."""
    try:
        json.dumps(data, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise UserPromptValidationError(f"Prompt is not valid JSON-compatible data: {exc}") from None


def _validate_import_prompt_yaml(name: str, raw_bytes: bytes) -> dict:
    """Strict validation for user-imported prompts.

    Ensures valid YAML/JSON structure (no duplicate keys, strictly JSON-serializable),
    all required fields present and populated, valid `prompt_name` matching the file name,
    valid `rules` keys, and complete `mapping` covering every field.
    """
    if not raw_bytes or not raw_bytes.strip():
        raise UserPromptValidationError("Uploaded file is empty.")
    try:
        parsed = yaml.load(raw_bytes, Loader=_StrictYamlLoader)
    except UserPromptValidationError:
        raise
    except yaml.YAMLError as e:
        raise UserPromptValidationError(f"Invalid YAML/JSON syntax: {e}") from None

    if not isinstance(parsed, dict):
        raise UserPromptValidationError("Top-level document must be a JSON/YAML object (key: value pairs).")

    _assert_strict_json_compatible(parsed)

    missing = [k for k in REQUIRED_USER_PROMPT_KEYS if k not in parsed]
    if missing:
        raise UserPromptValidationError(f"Missing required field(s): {', '.join(missing)}.")

    for req_str in (
        "prompt_name",
        "prompt_version",
        "prompt_author",
        "prompt_description",
        "instructions",
        "json_formatting_instructions",
    ):
        val = parsed.get(req_str)
        if not isinstance(val, str) or not val.strip():
            raise UserPromptValidationError(f"Required field '{req_str}' must be a non-empty string.")

    prompt_name = parsed["prompt_name"].strip()
    if not PROMPT_NAME_PATTERN.match(prompt_name):
        raise UserPromptValidationError(
            "Invalid `prompt_name`: use only letters, numbers, '_' and '-' (no spaces)."
        )
    file_stem = Path(name).stem
    if file_stem != prompt_name:
        raise UserPromptValidationError(
            f"File name '{name}' must match prompt_name ('{prompt_name}.yaml')."
        )

    # Rules
    raw_rules = parsed.get("rules")
    if not isinstance(raw_rules, dict) or not raw_rules:
        raise UserPromptValidationError("`rules` must be a non-empty object mapping field names to rules.")
    seen_lower: set[str] = set()
    for raw_k, raw_v in raw_rules.items():
        if not isinstance(raw_k, str) or not FIELD_KEY_PATTERN.match(raw_k.strip()):
            raise UserPromptValidationError(
                f"Invalid field name '{raw_k}' in `rules`. Use letters, numbers, and underscores."
            )
        low = raw_k.strip().lower()
        if low in seen_lower:
            raise UserPromptValidationError(f"Duplicate field name '{raw_k}' in `rules`.")
        seen_lower.add(low)
        if raw_v is not None and not isinstance(raw_v, str):
            raise UserPromptValidationError(f"Rule for '{raw_k}' must be text.")
    rule_keys = {k.strip() for k in raw_rules}

    # Mapping
    raw_mapping = parsed.get("mapping")
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise UserPromptValidationError("`mapping` must be a non-empty object.")
    if len(raw_mapping) > MAX_MAPPING_CATEGORIES:
        raise UserPromptValidationError(f"`mapping` supports at most {MAX_MAPPING_CATEGORIES} categories.")
    for cat, members in raw_mapping.items():
        if not isinstance(cat, str) or not CATEGORY_PATTERN.match(cat.strip()):
            raise UserPromptValidationError(f"Mapping category '{cat}' must be ALL CAPS.")
        if not isinstance(members, list):
            raise UserPromptValidationError(f"Mapping category '{cat}' must be a list of field names.")
    missing_cats = [c for c in REQUIRED_MAPPING_CATEGORIES if c not in raw_mapping]
    if missing_cats:
        raise UserPromptValidationError(f"Missing required mapping category: {', '.join(missing_cats)}.")

    assigned: dict[str, str] = {}
    for cat, members in raw_mapping.items():
        for m in members:
            if not isinstance(m, str) or m.strip() not in rule_keys:
                raise UserPromptValidationError(f"Mapping category '{cat}' references unknown field '{m}'.")
            mk = m.strip()
            if mk in assigned:
                raise UserPromptValidationError(f"Field '{mk}' is mapped more than once ({assigned[mk]}, {cat}).")
            assigned[mk] = cat
    unmapped = [k for k in raw_rules if k.strip() not in assigned]
    if unmapped:
        raise UserPromptValidationError(f"Every field in `rules` must be mapped: {', '.join(unmapped)}.")

    # Optional editor sections
    if "checklist" in parsed and parsed["checklist"] is not None:
        if not isinstance(parsed["checklist"], list) or not all(isinstance(x, str) for x in parsed["checklist"]):
            raise UserPromptValidationError("`checklist` must be a list of strings.")
    if "field_default_values" in parsed and parsed["field_default_values"] is not None:
        defs = parsed["field_default_values"]
        if not isinstance(defs, dict):
            raise UserPromptValidationError("`field_default_values` must be an object.")
        for fk, fvals in defs.items():
            if fk not in rule_keys:
                raise UserPromptValidationError(f"`field_default_values` references unknown field '{fk}'.")
            if not isinstance(fvals, list) or not all(isinstance(v, str) for v in fvals):
                raise UserPromptValidationError(f"Default values for '{fk}' must be a list of strings.")
    if "review_not_required" in parsed and parsed["review_not_required"] is not None:
        rnr = parsed["review_not_required"]
        if not isinstance(rnr, list):
            raise UserPromptValidationError("`review_not_required` must be a list of field names.")
        for item in rnr:
            if not isinstance(item, str) or item.strip() not in rule_keys:
                raise UserPromptValidationError(f"`review_not_required` references unknown field '{item}'.")
    if "custom_flags" in parsed and parsed["custom_flags"] is not None:
        if not isinstance(parsed["custom_flags"], list) or not all(isinstance(x, str) for x in parsed["custom_flags"]):
            raise UserPromptValidationError("`custom_flags` must be a list of strings.")

    return parsed


def _validate_user_prompt_yaml(raw_bytes: bytes) -> dict:
    """Parse + schema-check prompt YAML (VoucherVisionGO). Returns the parsed dict.

    Never logs `raw_bytes`. Raises UserPromptValidationError with a user-safe message.
    """
    if not raw_bytes:
        raise UserPromptValidationError("Uploaded file is empty.")
    try:
        parsed = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as e:
        raise UserPromptValidationError(f"YAML parse error: {e}") from None

    if not isinstance(parsed, dict):
        raise UserPromptValidationError("Top-level YAML must be a mapping (key: value pairs).")

    missing = [k for k in REQUIRED_USER_PROMPT_KEYS if k not in parsed]
    if missing:
        raise UserPromptValidationError(f"Missing required top-level key(s): {', '.join(missing)}")

    if not isinstance(parsed.get("rules"), dict):
        raise UserPromptValidationError("`rules` must be a mapping.")
    if not isinstance(parsed.get("mapping"), dict):
        raise UserPromptValidationError("`mapping` must be a mapping.")
    if not isinstance(parsed.get("instructions"), str) or not parsed["instructions"].strip():
        raise UserPromptValidationError("`instructions` must be a non-empty string.")
    if not isinstance(parsed.get("json_formatting_instructions"), str):
        raise UserPromptValidationError("`json_formatting_instructions` must be a string.")
    if not isinstance(parsed.get("prompt_name"), str) or not parsed["prompt_name"].strip():
        raise UserPromptValidationError("`prompt_name` must be a non-empty string.")

    # VVL additions: the pipeline needs at least one field, and field names
    # become JSON keys.
    if not parsed["rules"]:
        raise UserPromptValidationError("`rules` must list at least one field.")
    bad = [k for k in parsed["rules"] if not isinstance(k, str) or not k.strip()]
    if bad:
        raise UserPromptValidationError("Every key under `rules` must be a non-empty field name.")
    return parsed


# ------------------------------------------------------------- name rules --

def valid_name(name) -> bool:
    return isinstance(name, str) and bool(NAME_PATTERN.match(name)) and not name.startswith(".")


def source_dir(source: str) -> Path:
    if source == "builtin":
        return paths.prompts_builtin_dir()
    if source == "user":
        return paths.prompts_user_dir()
    raise ApiError("INVALID_INPUT", f"Unknown prompt source '{source}'. Use builtin or user.", status=400)


def _list_names(folder: Path) -> list[str]:
    try:
        entries = sorted(os.listdir(folder), key=str.lower)
    except FileNotFoundError:
        return []
    names = []
    for name in entries:
        if not valid_name(name):
            continue
        full = folder / name
        if full.is_file() and paths.is_within(full, folder):
            names.append(name)
    return names


def builtin_names() -> list[str]:
    return _list_names(paths.prompts_builtin_dir())


def user_names() -> list[str]:
    return _list_names(paths.prompts_user_dir())


def collides_with_builtin(name: str) -> bool:
    lowered = name.lower()
    return any(b.lower() == lowered for b in builtin_names())


def _filename_collision(name: str, *, exclude_user: str | None = None) -> str | None:
    """Return a user-facing error if `name` overlaps any existing built-in or user prompt filename."""
    lowered = name.lower()
    if any(b.lower() == lowered for b in builtin_names()):
        return "A built-in prompt with that file name already exists. Pick a different name."
    ex = exclude_user.lower() if exclude_user else None
    if any(u.lower() == lowered and u.lower() != ex for u in user_names()):
        return "A custom prompt with that file name already exists. Pick a different name."
    return None


def _prompt_name_collision(prompt_name: str, *, exclude_user: str | None = None) -> str | None:
    """Return a user-facing error if `prompt_name` overlaps any other prompt's display name."""
    target = prompt_name.strip().lower()
    if not target:
        return None
    ex = exclude_user.lower() if exclude_user else None
    for source in SOURCES:
        names = builtin_names() if source == "builtin" else user_names()
        for fname in names:
            if source == "user" and ex and fname.lower() == ex:
                continue
            _, data, _ = _load(source_dir(source) / fname)
            existing = _str_or_none((data or {}).get("prompt_name")) or ""
            if existing.strip().lower() == target:
                kind = "built-in" if source == "builtin" else "custom"
                return f"Another {kind} prompt is already named '{existing.strip()}'. Pick a different name."
    return None


def prompt_path(source: str, name: str) -> Path:
    """Confined path of an existing prompt. Raises ApiError (400/404)."""
    if not valid_name(name):
        raise ApiError(
            "INVALID_INPUT",
            "Prompt names must look like 'My_prompt-1.yaml' (letters, digits, '.', '_', '-'; ending in .yaml or .yml).",
            status=400,
        )
    folder = source_dir(source)
    candidate = folder / name
    if not paths.is_within(candidate, folder) or not candidate.is_file():
        raise ApiError("PROMPT_NOT_FOUND", f"Prompt '{name}' was not found in the {source} prompts.", status=404)
    return candidate


def resolve(name: str | None) -> tuple[Path, str]:
    """(path, source) for a prompt named in a processing request; built-ins first."""
    name = name or DEFAULT_PROMPT
    if not valid_name(name):
        raise ApiError("INVALID_INPUT", f"'{name}' is not a valid prompt name.", status=400)
    for source in SOURCES:
        folder = source_dir(source)
        candidate = folder / name
        if candidate.is_file() and paths.is_within(candidate, folder):
            return candidate, source
    raise ApiError("PROMPT_NOT_FOUND", "Prompt not found or not accessible.", status=404)


# ------------------------------------------------------------- summaries --

def _load(path: Path) -> tuple[str, dict | None, str | None]:
    """(yaml_text, parsed-or-None, error-or-None) using the upload validation rules."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return "", None, f"Could not read the file ({type(exc).__name__})."
    # Several built-ins use CRLF; the UI gets plain "\n" line endings.
    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
    try:
        parsed = _validate_user_prompt_yaml(raw)
        return text, parsed, None
    except UserPromptValidationError as exc:
        try:
            loose = yaml.safe_load(raw)
        except yaml.YAMLError:
            loose = None
        return text, loose if isinstance(loose, dict) else None, str(exc)


def _str_or_none(value):
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _summary_from(name: str, source: str, data: dict | None, error: str | None) -> dict:
    data = data or {}
    rules = data.get("rules")
    return {
        "name": name,
        "source": source,
        "prompt_name": _str_or_none(data.get("prompt_name")),
        "prompt_author": _str_or_none(data.get("prompt_author")),
        "prompt_author_institution": _str_or_none(data.get("prompt_author_institution")),
        "prompt_version": _str_or_none(data.get("prompt_version")),
        "prompt_description": _str_or_none(data.get("prompt_description")),
        "llm": _str_or_none(data.get("LLM")),
        "n_fields": len(rules) if isinstance(rules, dict) else 0,
        "valid": error is None,
        "error": error,
    }


def summary(source: str, name: str) -> dict:
    _, data, error = _load(prompt_path(source, name))
    return _summary_from(name, source, data, error)


def list_all() -> list[dict]:
    out = []
    for source in SOURCES:
        names = builtin_names() if source == "builtin" else user_names()
        for name in names:
            _, data, error = _load(source_dir(source) / name)
            out.append(_summary_from(name, source, data, error))
    return out


def _str_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        text = _str_or_none(item)
        if text is not None and text.strip():
            out.append(text.strip())
    return out


def _defaults_map(value) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, list[str]] = {}
    for k, v in value.items():
        key = _str_or_none(k)
        if not key or not key.strip():
            continue
        items = _str_list(v) if isinstance(v, list) else ([_str_or_none(v).strip()] if _str_or_none(v) and _str_or_none(v).strip() else [])
        if items:
            out[key.strip()] = items
    return out


def detail(source: str, name: str) -> dict:
    text, data, error = _load(prompt_path(source, name))
    base = _summary_from(name, source, data, error)
    data = data or {}
    rules = data.get("rules") if isinstance(data.get("rules"), dict) else {}
    mapping = data.get("mapping") if isinstance(data.get("mapping"), dict) else {}
    group_of = {}
    for group, members in mapping.items():
        if isinstance(members, list):
            for member in members:
                group_of.setdefault(str(member), str(group))
    fields = [
        {"name": str(field), "rule": _str_or_none(rule) or "", "group": group_of.get(str(field))}
        for field, rule in rules.items()
    ]
    return {
        "name": base["name"],
        "source": base["source"],
        "prompt_name": base["prompt_name"],
        "prompt_author": base["prompt_author"],
        "prompt_author_institution": base["prompt_author_institution"],
        "prompt_version": base["prompt_version"],
        "prompt_description": base["prompt_description"],
        "llm": base["llm"],
        "instructions": _str_or_none(data.get("instructions")),
        "json_formatting_instructions": _str_or_none(data.get("json_formatting_instructions")),
        "fields": fields,
        "mapping": {str(k): [str(x) for x in v] if isinstance(v, list) else v for k, v in mapping.items()},
        "checklist": _str_list(data.get("checklist")),
        "field_default_values": _defaults_map(data.get("field_default_values")),
        "review_not_required": _str_list(data.get("review_not_required")),
        "custom_flags": _str_list(data.get("custom_flags")),
        "yaml_text": text,
        "valid": base["valid"],
        "error": base["error"],
    }


# -------------------------------------------------------- designer save ---

def _clean_str(value, field_label: str, *, required: bool = False) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.replace("\r\n", "\n").strip()
    else:
        raise UserPromptValidationError(f"`{field_label}` must be text.")
    if required and not text:
        raise UserPromptValidationError(f"`{field_label}` is required.")
    return text


def normalize_designer_prompt(payload: dict) -> dict:
    """Validate and normalize a structured Designer prompt into canonical dict order.

    Enforces all invariants needed by VoucherVision and VVGO-Editor so invalid
    prompts cannot be saved or exported.
    """
    if not isinstance(payload, dict):
        raise UserPromptValidationError("Prompt data must be an object.")

    prompt_author = _clean_str(payload.get("prompt_author"), "prompt_author", required=True)
    prompt_author_institution = _clean_str(payload.get("prompt_author_institution"), "prompt_author_institution")
    prompt_name = _clean_str(payload.get("prompt_name"), "prompt_name", required=True)
    if not PROMPT_NAME_PATTERN.match(prompt_name):
        raise UserPromptValidationError(
            "Prompt name can only use letters, numbers, '_' and '-' (no spaces)."
        )
    prompt_version = _clean_str(payload.get("prompt_version"), "prompt_version", required=True)
    prompt_description = _clean_str(payload.get("prompt_description"), "prompt_description", required=True)
    llm = _clean_str(payload.get("LLM", payload.get("llm")), "LLM")
    instructions = _clean_str(payload.get("instructions"), "instructions", required=True)
    # Formatting instructions are locked to the canonical standard text.
    json_formatting_instructions = CANONICAL_JSON_FORMATTING_INSTRUCTIONS

    # Rules: accept either {name: rule} dict or [{name, rule}] list to preserve order.
    raw_rules = payload.get("rules")
    if raw_rules is None and isinstance(payload.get("fields"), list):
        raw_rules = payload.get("fields")
    rules: dict[str, str] = {}
    if isinstance(raw_rules, dict):
        items = list(raw_rules.items())
    elif isinstance(raw_rules, list):
        items = []
        for entry in raw_rules:
            if not isinstance(entry, dict) or "name" not in entry:
                raise UserPromptValidationError("Each field must have a name.")
            items.append((entry.get("name"), entry.get("rule", "")))
    else:
        raise UserPromptValidationError("`rules` must be a mapping of field names to rules.")

    if not items:
        raise UserPromptValidationError("Add at least one field.")
    seen_lower: set[str] = set()
    for raw_key, raw_rule in items:
        if not isinstance(raw_key, str) or not FIELD_KEY_PATTERN.match(raw_key.strip()):
            raise UserPromptValidationError(
                f"Invalid field name '{raw_key}'. Use letters, digits, and underscores (starting with a letter)."
            )
        key = raw_key.strip()
        if key in rules or key.lower() in seen_lower:
            raise UserPromptValidationError(f"Duplicate field name '{key}'.")
        seen_lower.add(key.lower())
        rules[key] = _clean_str(raw_rule, f"rules.{key}")

    # Mapping: 3 required categories + up to 3 extra ALL-CAPS categories (6 total).
    raw_mapping = payload.get("mapping")
    if not isinstance(raw_mapping, dict):
        raise UserPromptValidationError("`mapping` must be a mapping of categories to field lists.")
    if len(raw_mapping) > MAX_MAPPING_CATEGORIES:
        raise UserPromptValidationError(f"Mapping supports at most {MAX_MAPPING_CATEGORIES} categories.")

    mapping: dict[str, list[str]] = {}
    for raw_cat, raw_members in raw_mapping.items():
        if not isinstance(raw_cat, str) or not CATEGORY_PATTERN.match(raw_cat.strip()):
            raise UserPromptValidationError(
                f"Category '{raw_cat}' must be ALL CAPS (letters, digits, underscores)."
            )
        cat = raw_cat.strip()
        if cat in mapping:
            raise UserPromptValidationError(f"Duplicate category '{cat}'.")
        if not isinstance(raw_members, list):
            raise UserPromptValidationError(f"Category '{cat}' must list field names.")
        members: list[str] = []
        for m in raw_members:
            if not isinstance(m, str) or m.strip() not in rules:
                raise UserPromptValidationError(f"Category '{cat}' references unknown field '{m}'.")
            members.append(m.strip())
        mapping[cat] = members

    missing_cats = [c for c in REQUIRED_MAPPING_CATEGORIES if c not in mapping]
    if missing_cats:
        raise UserPromptValidationError(f"Missing required mapping category: {', '.join(missing_cats)}.")

    # Canonical category order: required 3 first, then custom categories.
    ordered_mapping: dict[str, list[str]] = {}
    for c in REQUIRED_MAPPING_CATEGORIES:
        ordered_mapping[c] = mapping[c]
    for c, members in mapping.items():
        if c not in ordered_mapping:
            ordered_mapping[c] = members

    assigned: dict[str, str] = {}
    for cat, members in ordered_mapping.items():
        for m in members:
            if m in assigned:
                raise UserPromptValidationError(f"Field '{m}' is mapped more than once ({assigned[m]}, {cat}).")
            assigned[m] = cat
    unmapped = [k for k in rules if k not in assigned]
    if unmapped:
        raise UserPromptValidationError(f"Every field must be in a mapping category: {', '.join(unmapped)}.")

    # Checklist
    raw_checklist = payload.get("checklist", [])
    if raw_checklist is None:
        raw_checklist = []
    if not isinstance(raw_checklist, list):
        raise UserPromptValidationError("`checklist` must be a list of steps.")
    checklist: list[str] = []
    for item in raw_checklist:
        if not isinstance(item, str):
            raise UserPromptValidationError("Each checklist item must be text.")
        cleaned = item.strip()
        if cleaned:
            checklist.append(cleaned)

    # Default values
    raw_defaults = payload.get("field_default_values", {})
    if raw_defaults is None:
        raw_defaults = {}
    if not isinstance(raw_defaults, dict):
        raise UserPromptValidationError("`field_default_values` must be a mapping.")
    field_default_values: dict[str, list[str]] = {}
    for raw_k, raw_vals in raw_defaults.items():
        if not isinstance(raw_k, str) or raw_k.strip() not in rules:
            raise UserPromptValidationError(f"Default values reference unknown field '{raw_k}'.")
        if not isinstance(raw_vals, list):
            raise UserPromptValidationError(f"Default values for '{raw_k}' must be a list.")
        vals = [v.strip() for v in raw_vals if isinstance(v, str) and v.strip()]
        if vals:
            field_default_values[raw_k.strip()] = list(dict.fromkeys(vals))

    # Review not required
    raw_skip = payload.get("review_not_required", [])
    if raw_skip is None:
        raw_skip = []
    if not isinstance(raw_skip, list):
        raise UserPromptValidationError("`review_not_required` must be a list of field names.")
    review_not_required: list[str] = []
    for item in raw_skip:
        if not isinstance(item, str) or item.strip() not in rules:
            raise UserPromptValidationError(f"`review_not_required` references unknown field '{item}'.")
        k = item.strip()
        if k not in review_not_required:
            review_not_required.append(k)

    # Custom flags
    raw_flags = payload.get("custom_flags", [])
    if raw_flags is None:
        raw_flags = []
    if not isinstance(raw_flags, list):
        raise UserPromptValidationError("`custom_flags` must be a list of strings.")
    custom_flags: list[str] = []
    for item in raw_flags:
        if not isinstance(item, str):
            raise UserPromptValidationError("Each custom flag must be text.")
        cleaned = item.strip()
        if cleaned:
            custom_flags.append(cleaned)

    return {
        "prompt_author": prompt_author,
        "prompt_author_institution": prompt_author_institution,
        "prompt_name": prompt_name,
        "prompt_version": prompt_version,
        "prompt_description": prompt_description,
        "LLM": llm,
        "instructions": instructions,
        "json_formatting_instructions": json_formatting_instructions,
        "rules": rules,
        "mapping": ordered_mapping,
        "checklist": checklist,
        "field_default_values": field_default_values,
        "review_not_required": review_not_required,
        "custom_flags": custom_flags,
    }


class _PromptYamlDumper(yaml.SafeDumper):
    """YAML dumper that uses block style for multiline strings and keeps clean formatting."""


def _str_representer(dumper: yaml.SafeDumper, value: str):
    if "\n" in value:
        return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", value)


_PromptYamlDumper.add_representer(str, _str_representer)


def render_designer_yaml(payload: dict) -> tuple[dict, str]:
    """Validate `payload` and return `(normalized_dict, yaml_text)`."""
    normalized = normalize_designer_prompt(payload)
    yaml_text = yaml.dump(
        normalized,
        Dumper=_PromptYamlDumper,
        sort_keys=False,
        allow_unicode=True,
        width=1000,
    )
    raw = yaml_text.encode("utf-8")
    if len(raw) > USER_PROMPTS_MAX_BYTES:
        raise UserPromptValidationError(f"Prompt exceeds maximum size of {USER_PROMPTS_MAX_BYTES} bytes.")
    # Double-check round-trip with the standard validator.
    _validate_user_prompt_yaml(raw)
    return normalized, yaml_text


def save_user_prompt(name: str, payload: dict, *, overwrite: bool) -> dict:
    """Create (`overwrite=False`) or update (`overwrite=True`) a custom prompt.

    File name and `prompt_name` always match (`<prompt_name>.yaml`). Updating a
    custom prompt with a new `prompt_name` cleanly renames the file on disk.
    """
    if not valid_name(name):
        raise ApiError(
            "INVALID_INPUT",
            "Prompt names must use letters, numbers, '_' and '-' and end in .yaml or .yml.",
            status=400,
        )
    if collides_with_builtin(name):
        raise ApiError(
            "FORBIDDEN" if overwrite else "CONFLICT",
            "Built-in prompts cannot be modified. Save a copy under a new name.",
            status=403 if overwrite else 409,
        )
    try:
        normalized, yaml_text = render_designer_yaml(payload)
    except UserPromptValidationError as exc:
        raise ApiError("INVALID_INPUT", str(exc), status=400) from None

    expected_stem = normalized["prompt_name"]
    if overwrite:
        ext = ".yml" if name.lower().endswith(".yml") and Path(name).stem == expected_stem else ".yaml"
        target_name = f"{expected_stem}{ext}"
    else:
        if Path(name).stem != expected_stem:
            raise ApiError(
                "INVALID_INPUT",
                f"File name '{name}' must match prompt_name ('{expected_stem}.yaml').",
                status=400,
            )
        target_name = name

    exclude = name if overwrite else None
    fname_err = _filename_collision(target_name, exclude_user=exclude)
    if fname_err:
        raise ApiError("CONFLICT", fname_err, status=409)
    pname_err = _prompt_name_collision(expected_stem, exclude_user=exclude)
    if pname_err:
        raise ApiError("CONFLICT", pname_err, status=409)

    folder = paths.ensure_dir(paths.prompts_user_dir())
    old_path = folder / name
    target = folder / target_name
    if not paths.is_within(old_path, folder) or not paths.is_within(target, folder):
        raise ApiError("INVALID_INPUT", "That prompt name is not allowed.", status=400)
    if overwrite and not old_path.is_file():
        raise ApiError("PROMPT_NOT_FOUND", f"Prompt '{name}' was not found in custom prompts.", status=404)
    if not overwrite and target.exists():
        raise ApiError("CONFLICT", "A custom prompt with that file name already exists.", status=409)

    raw = yaml_text.encode("utf-8")
    atomic_write_bytes(target, raw, mode=0o644)
    if overwrite and target_name != name and old_path.is_file():
        old_path.unlink()
    logger.info("User prompt %s: %s (%d bytes)", "updated" if overwrite else "created", target_name, len(raw))
    return detail("user", target_name)


# -------------------------------------------------------- import / delete --

def import_prompt(name: str, raw: bytes) -> dict:
    """Strictly validate and save a user prompt. Returns its summary. Raises ApiError."""
    if not valid_name(name):
        raise ApiError(
            "INVALID_INPUT",
            "Prompt file names must use letters, numbers, '_' and '-' and end in .yaml or .yml.",
            status=400,
        )
    if len(raw) > USER_PROMPTS_MAX_BYTES:
        raise ApiError(
            "PAYLOAD_TOO_LARGE", f"File exceeds maximum size of {USER_PROMPTS_MAX_BYTES} bytes.", status=413,
        )
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ApiError("INVALID_INPUT", "Invalid prompt file: the file must be UTF-8 text.", status=400) from None
    try:
        parsed = _validate_import_prompt_yaml(name, raw)
    except UserPromptValidationError as exc:
        raise ApiError("INVALID_INPUT", f"Invalid prompt: {exc}", status=400) from None
    fname_err = _filename_collision(name)
    if fname_err:
        raise ApiError("CONFLICT", fname_err, status=409)
    pname_err = _prompt_name_collision(parsed["prompt_name"])
    if pname_err:
        raise ApiError("CONFLICT", pname_err, status=409)
    folder = paths.ensure_dir(paths.prompts_user_dir())
    target = folder / name
    if not paths.is_within(target, folder):
        raise ApiError("INVALID_INPUT", "That prompt name is not allowed.", status=400)
    if target.exists():
        raise ApiError(
            "CONFLICT", "A user prompt with that filename already exists. Delete it first or pick another name.",
            status=409,
        )
    atomic_write_bytes(target, raw, mode=0o644)
    logger.info("User prompt imported: %s (%d bytes)", name, len(raw))
    return _summary_from(name, "user", parsed, None)


def delete_user_prompt(name: str) -> None:
    path = prompt_path("user", name)
    path.unlink()
    logger.info("User prompt deleted: %s", name)


# ------------------------------------------ legacy GET /prompts (VVGO shape) --

def extract_prompt_info(prompt_file: Path) -> dict:
    """Basic information from a prompt file for the /prompts listing."""
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            content = f.read()

        info = {
            "filename": prompt_file.name,
            "description": "No description provided",
            "version": "Unknown",
            "author": "Unknown",
            "institution": "Unknown",
            "name": os.path.splitext(prompt_file.name)[0],
            "full_path": str(prompt_file.absolute()),
        }

        try:
            data = yaml.safe_load(content)
            if isinstance(data, dict):
                field_mapping = {
                    "prompt_name": "name",
                    "prompt_version": "version",
                    "prompt_author": "author",
                    "prompt_author_institution": "institution",
                    "prompt_description": "description",
                }
                for yaml_field, info_field in field_mapping.items():
                    if yaml_field in data and data[yaml_field]:
                        info[info_field] = data[yaml_field]
        except Exception as e:
            logger.warning(f"YAML parsing failed for info extraction: {e}, using regex")
            patterns = {
                "name": r"prompt_name:\s*(.*?)(?=\n\w+:|$)",
                "version": r"prompt_version:\s*(.*?)(?=\n\w+:|$)",
                "author": r"prompt_author:\s*(.*?)(?=\n\w+:|$)",
                "institution": r"prompt_author_institution:\s*(.*?)(?=\n\w+:|$)",
                "description": r"prompt_description:\s*(.*?)(?=\n\w+:|$)",
            }
            for field, pattern in patterns.items():
                matches = re.findall(pattern, content, re.DOTALL)
                if matches:
                    info[field] = " ".join(line.strip() for line in matches[0].strip().split("\n"))

        return info

    except Exception as e:
        logger.error(f"Error extracting info from {prompt_file}: {e}")
        return {
            "filename": prompt_file.name,
            "description": f"Error reading file: {str(e)}",
            "version": "Unknown",
            "author": "Unknown",
            "institution": "Unknown",
            "name": os.path.splitext(prompt_file.name)[0],
            "full_path": str(prompt_file.absolute()),
        }


def extract_prompt_details(prompt_file: Path) -> dict:
    """Detailed content of a prompt file (raw text + parsed YAML)."""
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            content = f.read()
        details = {"raw_content": content}
        try:
            data = yaml.safe_load(content)
            if isinstance(data, dict):
                details["parsed_data"] = data
            else:
                logger.warning(f"YAML parsing produced non-dictionary: {type(data)}")
                details["parsed_data"] = {"content": data}
        except Exception as e:
            logger.warning(f"YAML parsing failed for {prompt_file}: {e}")
            details["parse_error"] = str(e)
        return details
    except Exception as e:
        logger.error(f"Error extracting details from {prompt_file}: {e}")
        return {"error": str(e), "raw_content": ""}


def format_prompts_as_text_table(prompt_list: list[dict]) -> str:
    """Plain-text table of prompts for terminal display (VVGO used `tabulate`)."""
    headers = ["#", "Filename", "Description", "Version", "Author", "Institution"]
    rows = []
    for i, info in enumerate(prompt_list, 1):
        rows.append([
            str(i),
            str(info.get("filename", "")),
            textwrap.shorten(str(info.get("description", "")), width=50, placeholder="..."),
            str(info.get("version", "Unknown")),
            str(info.get("author", "Unknown")),
            str(info.get("institution", "Unknown")),
        ])
    widths = [max(len(h), *(len(r[c]) for r in rows)) if rows else len(h) for c, h in enumerate(headers)]

    def line(cells):
        return "| " + " | ".join(cell.ljust(widths[c]) for c, cell in enumerate(cells)) + " |"

    rule = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    table = "\n".join([rule, line(headers), rule.replace("-", "=")] + [line(r) for r in rows] + [rule])
    return f"Available Prompt Templates:\n\n{table}\n\nTotal: {len(prompt_list)} prompt file(s) found"


def format_prompt_as_text(filename: str, details: dict) -> str:
    """One prompt's details as plain text for terminal display."""
    lines = ["=" * 80, f"PROMPT FILE: {filename}", "=" * 80, ""]
    data = details.get("parsed_data") or {}
    metadata_fields = {
        "prompt_name": "Name",
        "prompt_description": "Description",
        "prompt_version": "Version",
        "prompt_author": "Author",
        "prompt_author_institution": "Institution",
        "LLM": "LLM Type",
    }
    if data:
        metadata = {label: data[key] for key, label in metadata_fields.items() if data.get(key)}
        if metadata:
            lines.append("METADATA:")
            for label, value in metadata.items():
                if isinstance(value, str):
                    lines.append(f"{label}: {textwrap.fill(value, width=76, subsequent_indent='    ')}")
                else:
                    lines.append(f"{label}: {value}")
            lines.append("")
        priority = [
            ("instructions", "INSTRUCTIONS"),
            ("json_formatting_instructions", "JSON FORMATTING INSTRUCTIONS"),
            ("rules", "RULES"),
            ("mapping", "MAPPING"),
            ("examples", "EXAMPLES"),
        ]
        shown = {k for k, _ in priority} | set(metadata_fields)
        sections = [(k, h) for k, h in priority if data.get(k)]
        sections += [(k, k.replace("_", " ").upper()) for k in data if k not in shown]
        for key, heading in sections:
            value = data[key]
            lines.append(heading + ":")
            if isinstance(value, str):
                lines.append(textwrap.fill(value, width=76, subsequent_indent="  "))
            else:
                for yaml_line in yaml.dump(value, default_flow_style=False).split("\n"):
                    lines.append("  " + yaml_line)
            lines.append("")
    elif "raw_content" in details:
        lines.append("RAW CONTENT:")
        lines.append(details["raw_content"])
    lines.append("=" * 80)
    return "\n".join(lines)
