"""Find the models folder without importing the backend at import time.

Order: explicit argument -> ``vvl.paths`` (if that module exists) -> the
settings YAML (``VVL_SETTINGS_PATH`` or ``<repo>/settings.local.yaml`` when run
from source, else ``<VVL_HOME>/settings.yaml``) -> ``<VVL_HOME>/models``, where
VVL_HOME defaults to the OS user-data folder "VoucherVisionLocal". Every step
is guarded; this never raises.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import sys

APP_NAME = "VoucherVisionLocal"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _norm(path, base=None):
    p = os.path.expandvars(os.path.expanduser(str(path).strip()))
    if base and not os.path.isabs(p):
        p = os.path.join(base, p)
    return os.path.normpath(os.path.abspath(p))


def _usable(value):
    return isinstance(value, (str, os.PathLike)) and str(value).strip() != ""


def from_vvl_paths():
    try:
        mod = sys.modules.get("vvl.paths")
        if mod is None:
            if importlib.util.find_spec("vvl.paths") is None:
                return None
            mod = importlib.import_module("vvl.paths")
    except Exception:
        return None
    for attr in ("models_dir", "get_models_dir", "resolve_models_dir", "MODELS_DIR"):
        try:
            val = getattr(mod, attr, None)
            if callable(val):
                val = val()
            if _usable(val):
                return _norm(val)
        except Exception:
            continue
    for attr in ("get_paths", "paths", "resolve_paths", "PATHS"):
        try:
            val = getattr(mod, attr, None)
            if callable(val):
                val = val()
            if isinstance(val, dict):
                val = val.get("models_dir")
            else:
                val = getattr(val, "models_dir", None)
            if callable(val):
                val = val()
            if _usable(val):
                return _norm(val)
        except Exception:
            continue
    return None


def vvl_home(env=None):
    env = os.environ if env is None else env
    if env.get("VVL_HOME"):
        return _norm(env["VVL_HOME"])
    try:
        import platformdirs

        return os.path.normpath(platformdirs.user_data_dir(APP_NAME, appauthor=False))
    except Exception:
        return _norm(os.path.join("~", "." + APP_NAME))


def settings_path(env=None):
    env = os.environ if env is None else env
    if env.get("VVL_SETTINGS_PATH"):
        return _norm(env["VVL_SETTINGS_PATH"])
    if os.path.isfile(os.path.join(REPO_ROOT, "pyproject.toml")):
        return os.path.join(REPO_ROOT, "settings.local.yaml")
    return os.path.join(vvl_home(env), "settings.yaml")


def from_settings(env=None):
    path = settings_path(env)
    # Never open a credentials file, even if a variable points at one.
    if re.match(r"private.*\.ya?ml$", os.path.basename(path), re.I):
        return None
    try:
        if not os.path.isfile(path) or os.path.getsize(path) > 1_000_000:
            return None
        import yaml

        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            data = yaml.safe_load(fh)
        val = data.get("models_dir") if isinstance(data, dict) else None
        if _usable(val):
            return _norm(val, base=os.path.dirname(path))
    except Exception:
        return None
    return None


def resolve(explicit=None, env=None):
    """(models_dir, source) where source is argument | vvl.paths | settings | default."""
    try:
        if _usable(explicit):
            return _norm(explicit), "argument"
    except Exception:
        pass
    found = from_vvl_paths()
    if found:
        return found, "vvl.paths"
    found = from_settings(env)
    if found:
        return found, "settings"
    try:
        return os.path.join(vvl_home(env), "models"), "default"
    except Exception:
        return None, None
