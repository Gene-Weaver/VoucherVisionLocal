"""PRIVATE.yaml: the only place credentials live.

File layout (see PRIVATE.example.yaml):

    version: 1
    credentials:
      <provider_id>:
        <field_name>: "<secret value>"

Rules (docs/API.md section 2.6 and PUT /v1/keys):
  * Values are write-only for the API. Nothing in this module returns a value
    to a route except `get_value()`, which the key-test and model-list code
    uses to call the provider itself.
  * Values are never logged. Log lines name the provider and field only.
  * Writes are atomic (temp file in the same folder + rename) with mode 0600,
    and keep every other provider's entries and any other top-level keys.
  * An empty string removes a field. A provider left with no fields is removed.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Iterable

import yaml

from vvl import paths
from vvl.fileutil import atomic_write_text

logger = logging.getLogger(__name__)

MASK_DOTS = "••••"  # "••••"
MASK_MIN_LENGTH = 12
MAX_VALUE_LENGTH = 8192
FILE_MODE = 0o600

_HEADER = """\
# VoucherVisionLocal credentials. Written by the app (Keys tab).
# This file is gitignored and never leaves this computer. Keys are sent only
# to the service they belong to, and only when a request uses that service.
"""

_lock = threading.RLock()


class SecretsFileError(RuntimeError):
    """PRIVATE.yaml exists but cannot be parsed. The message never includes its content."""


class SecretValueError(ValueError):
    """A submitted value or field name was rejected. The message never includes the value."""


def mask(value: str | None) -> str | None:
    """'••••' + last 4 characters, or just '••••' for values shorter than 12."""
    if not value:
        return None
    if len(value) < MASK_MIN_LENGTH:
        return MASK_DOTS
    return MASK_DOTS + value[-4:]


class SecretStore:
    def __init__(self, path: Path | None = None):
        self._path = Path(path) if path else None

    @property
    def path(self) -> Path:
        return self._path or paths.private_path()

    # -- reading ---------------------------------------------------------

    def _read(self) -> dict:
        path = self.path
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as exc:
            raise SecretsFileError(f"{path.name} could not be opened ({type(exc).__name__}).") from None
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            # No `from exc`: a YAML error message can quote the offending line.
            raise SecretsFileError(
                f"{path.name} is not valid YAML. Fix it by hand or move it aside; "
                "the app will not overwrite it."
            ) from None
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise SecretsFileError(f"{path.name} must be a YAML mapping.")
        return data

    def _credentials(self, data: dict) -> dict:
        creds = data.get("credentials")
        return creds if isinstance(creds, dict) else {}

    def _provider_values(self, provider_id: str) -> dict[str, str]:
        with _lock:
            entry = self._credentials(self._read()).get(provider_id)
        if not isinstance(entry, dict):
            return {}
        return {str(k): v for k, v in entry.items() if isinstance(v, str) and v != ""}

    def get_value(self, provider_id: str, field: str) -> str | None:
        """The saved value, for calling the provider. Never return this from a route."""
        return self._provider_values(provider_id).get(field) or None

    def field_status(self, provider_id: str, field_names: Iterable[str]) -> dict[str, dict]:
        """{field: {"saved": bool, "masked": str|None}} with no values."""
        values = self._provider_values(provider_id)
        return {
            name: {"saved": bool(values.get(name)), "masked": mask(values.get(name))}
            for name in field_names
        }

    def is_configured(self, provider_id: str, field_names: Iterable[str]) -> bool:
        values = self._provider_values(provider_id)
        names = list(field_names)
        return bool(names) and all(values.get(n) for n in names)

    # -- writing ---------------------------------------------------------

    def _write(self, data: dict) -> None:
        data.setdefault("version", 1)
        body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)
        atomic_write_text(self.path, _HEADER + "\n" + body, mode=FILE_MODE)

    def set_fields(self, provider_id: str, fields: dict, allowed_fields: Iterable[str]) -> None:
        """Save or remove fields for one provider. '' removes a field."""
        allowed = set(allowed_fields)
        if not isinstance(fields, dict) or not fields:
            raise SecretValueError("Send at least one field, e.g. {\"fields\": {\"api_key\": \"...\"}}.")
        unknown = sorted(str(k) for k in fields if k not in allowed)
        if unknown:
            raise SecretValueError(f"Unknown field(s) for this provider: {', '.join(unknown)}.")
        cleaned: dict[str, str] = {}
        for name, value in fields.items():
            if value is None:
                value = ""
            if not isinstance(value, str):
                raise SecretValueError(f"Field '{name}' must be a string.")
            value = value.strip()
            if len(value) > MAX_VALUE_LENGTH:
                raise SecretValueError(f"Field '{name}' is too long.")
            if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
                raise SecretValueError(f"Field '{name}' contains control characters or line breaks.")
            cleaned[name] = value
        with _lock:
            data = self._read()
            creds = dict(self._credentials(data))
            entry = creds.get(provider_id)
            entry = dict(entry) if isinstance(entry, dict) else {}
            for name, value in cleaned.items():
                if value:
                    entry[name] = value
                else:
                    entry.pop(name, None)
            entry = {k: v for k, v in entry.items() if v not in ("", None)}
            if entry:
                creds[provider_id] = entry
            else:
                creds.pop(provider_id, None)
            data["credentials"] = creds
            self._write(data)
        changed = ", ".join(f"{n}={'set' if v else 'removed'}" for n, v in cleaned.items())
        logger.info("Credentials updated for %s (%s).", provider_id, changed)

    def delete_provider(self, provider_id: str) -> None:
        with _lock:
            data = self._read()
            creds = dict(self._credentials(data))
            if provider_id not in creds:
                return
            creds.pop(provider_id, None)
            data["credentials"] = creds
            self._write(data)
        logger.info("Credentials removed for %s.", provider_id)
