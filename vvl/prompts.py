"""Prompt templates on disk: built-in (read-only) and user (imported).

Names are bare file names matching `^[A-Za-z0-9._-]+\\.ya?ml$` and must resolve
inside their folder (docs/API.md section 2.7). Validation of imported prompts
uses the rules VoucherVisionGO applies to user prompt uploads
(`_validate_user_prompt_yaml`), and `extract_prompt_info` /
`extract_prompt_details` / the text formatters are VoucherVisionGO's helpers
for the legacy `GET /prompts` route.
"""

from __future__ import annotations

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
SOURCES = ("builtin", "user")
DEFAULT_PROMPT = "SLTPvM_full.yaml"
USER_PROMPTS_MAX_BYTES = 256 * 1024

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


class UserPromptValidationError(ValueError):
    """Raised when a prompt YAML fails schema validation."""


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
        "mapping": {str(k): list(v) if isinstance(v, list) else v for k, v in mapping.items()},
        "yaml_text": text,
        "valid": base["valid"],
        "error": base["error"],
    }


# -------------------------------------------------------- import / delete --

def import_prompt(name: str, raw: bytes) -> dict:
    """Validate and save a user prompt. Returns its summary. Raises ApiError."""
    if not valid_name(name):
        raise ApiError(
            "INVALID_INPUT",
            "Prompt names must look like 'My_prompt-1.yaml' (letters, digits, '.', '_', '-'; ending in .yaml or .yml).",
            status=400,
        )
    if len(raw) > USER_PROMPTS_MAX_BYTES:
        raise ApiError(
            "PAYLOAD_TOO_LARGE", f"File exceeds maximum size of {USER_PROMPTS_MAX_BYTES} bytes.", status=413,
        )
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ApiError("INVALID_INPUT", "Invalid prompt YAML: the file must be UTF-8 text.", status=400) from None
    try:
        parsed = _validate_user_prompt_yaml(raw)
    except UserPromptValidationError as exc:
        raise ApiError("INVALID_INPUT", f"Invalid prompt YAML: {exc}", status=400) from None
    if collides_with_builtin(name):
        raise ApiError(
            "CONFLICT", "A built-in prompt with that filename already exists. Pick a different name.", status=409,
        )
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
