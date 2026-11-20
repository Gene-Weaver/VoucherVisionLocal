"""Saved OCR results for batches that run in two passes.

A staged batch runs OCR for every image first and parsing afterwards, so the
engine swaps models once per batch instead of twice per image. Between the two
passes each image's OCR result (text, token counts, the label collage) waits
here, one small JSON file per image under `<VVL_HOME>/cache/stages/`.

`save()` returns a token; the parsing pass sends it back and `load()` returns
the saved state. A file is removed when its image is finished, and anything
older than `MAX_AGE_SECONDS` is pruned.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import time
from pathlib import Path

from vvl import paths
from vvl.fileutil import atomic_write_text

logger = logging.getLogger(__name__)

MAX_AGE_SECONDS = 3 * 24 * 3600
_TOKEN = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
_last_prune = 0.0


class StageNotFound(Exception):
    """The token is malformed, unknown, or its saved result has expired."""


def _dir() -> Path:
    return paths.ensure_dir(paths.cache_dir() / "stages")


def _path(token: str) -> Path:
    if not isinstance(token, str) or not _TOKEN.match(token):
        raise StageNotFound("The stage token is not valid.")
    return _dir() / f"{token}.json"


def save(state: dict) -> str:
    token = secrets.token_urlsafe(18)
    atomic_write_text(_path(token), json.dumps(state), mode=0o600)
    prune()
    return token


def load(token: str) -> dict:
    path = _path(token)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise StageNotFound("This image's OCR result is no longer saved. Run the image again.") from None
    except (OSError, ValueError):
        raise StageNotFound("This image's saved OCR result could not be read. Run the image again.") from None


def discard(token: str) -> None:
    try:
        _path(token).unlink(missing_ok=True)
    except (OSError, StageNotFound):
        pass


def prune(max_age_seconds: float = MAX_AGE_SECONDS, *, force: bool = False) -> int:
    """Remove saved results older than `max_age_seconds`. Runs at most once a minute."""
    global _last_prune
    now = time.time()
    if not force and now - _last_prune < 60:
        return 0
    _last_prune = now
    removed = 0
    try:
        for path in _dir().glob("*.json"):
            try:
                if now - path.stat().st_mtime > max_age_seconds:
                    path.unlink()
                    removed += 1
            except OSError:
                continue
    except OSError:
        pass
    return removed
