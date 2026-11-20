"""Small file helpers shared by the settings, secrets, and prompt stores."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_bytes(path: Path, data: bytes, *, mode: int | None = None) -> None:
    """Write `data` to `path` atomically (temp file in the same folder + rename).

    `mode` is applied to the temp file before the rename, so the final file
    never exists with looser permissions. On Windows chmod only toggles the
    read-only bit; the file then inherits the folder's ACLs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        if mode is not None:
            try:
                os.chmod(tmp_name, mode)
            except OSError:
                pass
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    if mode is not None:
        try:
            os.chmod(path, mode)
        except OSError:
            pass


def atomic_write_text(path: Path, text: str, *, mode: int | None = None) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), mode=mode)
