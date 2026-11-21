"""Where VoucherVisionLocal keeps its files.

Every function reads the environment at call time, so tests and the desktop
shell can point the app somewhere else without reloading modules.

Environment variables (see docs/API.md section 1):
    VVL_HOME              runtime data folder (logs, cache, temp)
    VVL_PRIVATE_PATH      credentials file
    VVL_SETTINGS_PATH     settings file
    VVL_PROMPTS_USER_DIR  user prompt folder (not in the contract; used by tests
                          and scratch runs so they never touch the real folder)
"""

from __future__ import annotations

import os
from pathlib import Path

APP_DIR_NAME = "VoucherVisionLocal"


def app_root() -> Path:
    """The folder that holds app.py (the repo root when running from source)."""
    return Path(__file__).resolve().parent.parent


def from_source() -> bool:
    """True when a pyproject.toml sits beside app.py (a source checkout)."""
    root = app_root()
    return (root / "pyproject.toml").is_file() and (root / "app.py").is_file()


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    return Path(os.path.expanduser(value)).resolve()


def home() -> Path:
    """Runtime data folder. Not created here; see ensure_dir()."""
    override = _env_path("VVL_HOME")
    if override is not None:
        return override
    try:
        from platformdirs import user_data_dir

        return Path(user_data_dir(APP_DIR_NAME, appauthor=False, roaming=False))
    except Exception:
        return Path.home() / f".{APP_DIR_NAME.lower()}"


def private_path() -> Path:
    """The credentials file. Only the path; nothing here reads it."""
    override = _env_path("VVL_PRIVATE_PATH")
    if override is not None:
        return override
    if from_source():
        return app_root() / "PRIVATE.yaml"
    return home() / "PRIVATE.yaml"


def settings_path() -> Path:
    override = _env_path("VVL_SETTINGS_PATH")
    if override is not None:
        return override
    if from_source():
        return app_root() / "settings.local.yaml"
    return home() / "settings.yaml"


def prompts_builtin_dir() -> Path:
    return app_root() / "prompts" / "builtin"


def prompts_user_dir() -> Path:
    override = _env_path("VVL_PROMPTS_USER_DIR")
    if override is not None:
        return override
    if from_source():
        return app_root() / "prompts" / "user"
    return home() / "prompts" / "user"


def ui_dir() -> Path:
    return app_root() / "ui"


def registry_dir() -> Path:
    return Path(__file__).resolve().parent / "registry"


def default_models_dir() -> Path:
    return home() / "models"


def logs_dir() -> Path:
    return home() / "logs"


def temp_dir() -> Path:
    return home() / "tmp"


def cache_dir() -> Path:
    return home() / "cache"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_within(child: Path, parent: Path) -> bool:
    """True when `child` resolves to a location inside `parent` (symlinks followed)."""
    try:
        child_real = Path(os.path.realpath(child))
        parent_real = Path(os.path.realpath(parent))
    except OSError:
        return False
    return child_real == parent_real or parent_real in child_real.parents


# Folder names used by common sync clients. A models folder inside one of these
# would upload tens of gigabytes.
_SYNCED_MARKERS = (
    "dropbox",          # also "Dropbox (Personal)"
    "onedrive",         # also "OneDrive - Company"
    "icloud drive",
    "mobile documents",  # macOS iCloud Drive storage
    "google drive",
    "googledrive",
    "my drive",
    "box sync",
    "pcloud",
    "nextcloud",
    "owncloud",
    "megasync",
)


def synced_folder_hint(path: str | os.PathLike) -> str | None:
    """The path component that looks like a sync-client folder, or None."""
    for part in Path(path).parts:
        low = part.lower()
        if any(low.startswith(marker) for marker in _SYNCED_MARKERS):
            return part
    return None
