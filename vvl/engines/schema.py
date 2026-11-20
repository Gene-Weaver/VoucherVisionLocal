"""JSON schema from a prompt YAML's `rules`, for schema-constrained output.

A VoucherVision prompt's `rules` is a mapping of output field -> instruction.
Every field is a string in the parsed record (VoucherVisionGO's template is
`{field: ""}`), so the schema is a flat object of required strings with the
rule text as each field's description. llama.cpp (`response_format` /
`json_schema`), OpenAI structured outputs, and Gemini `response_schema` all
accept this subset.
"""

from __future__ import annotations

import json
from typing import Any

MAX_DESCRIPTION_CHARS = 2000


def _description(rule: Any) -> str | None:
    if rule is None:
        return None
    if isinstance(rule, str):
        text = rule.strip()
    else:
        text = json.dumps(rule, ensure_ascii=False, sort_keys=False)
    if not text:
        return None
    return text[:MAX_DESCRIPTION_CHARS]


def json_schema_from_rules(rules: Any, *, title: str | None = None) -> dict:
    """Build {"type": "object", ...} with one required string property per rule.

    Field order follows the YAML. Raises ValueError for a missing, empty, or
    malformed rules mapping.
    """
    if not isinstance(rules, dict) or not rules:
        raise ValueError("rules must be a non-empty mapping of field name -> rule")
    properties: dict[str, dict] = {}
    for name, rule in rules.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"rule field names must be non-empty strings (got {name!r})")
        prop: dict[str, Any] = {"type": "string"}
        description = _description(rule)
        if description:
            prop["description"] = description
        properties[name] = prop
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    if title:
        schema["title"] = title
    return schema


def json_schema_from_prompt(prompt_yaml: dict) -> dict:
    """Schema for a parsed prompt YAML (uses its `rules` and `prompt_name`)."""
    if not isinstance(prompt_yaml, dict):
        raise ValueError("prompt must be a YAML mapping")
    title = prompt_yaml.get("prompt_name")
    return json_schema_from_rules(prompt_yaml.get("rules"), title=title if isinstance(title, str) else None)
