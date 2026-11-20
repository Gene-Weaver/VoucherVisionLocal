"""Load and validate vvl/llama/engine.lock.yaml (the pinned llama.cpp release)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from vvl.registry.loader import RegistryError, _check_url

DEFAULT_LOCK_PATH = Path(__file__).resolve().parent / "engine.lock.yaml"


class _Model(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)


class Asset(_Model):
    kind: Literal["engine", "runtime"]
    name: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    url: str
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("url")
    @classmethod
    def _url(cls, value):
        return _check_url(value)

    @property
    def archive_type(self) -> str:
        lower = self.name.lower()
        if lower.endswith((".tar.gz", ".tgz")):
            return "tar.gz"
        if lower.endswith(".zip"):
            return "zip"
        return "unknown"


class Requires(_Model):
    nvidia_driver_min: str | None = None
    glibc_min: str | None = None      # Linux builds: lowest glibc the binaries run with


class Variant(_Model):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    os: Literal["linux", "macos", "windows"]
    arch: Literal["x86_64", "arm64"]
    backend: Literal["cuda", "vulkan", "metal", "cpu"]
    label: str
    requires: Requires = Field(default_factory=Requires)
    assets: list[Asset] = Field(min_length=1)

    @property
    def download_size_bytes(self) -> int:
        return sum(a.size_bytes for a in self.assets)


class Release(_Model):
    tag: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    stable_alias: str | None = None
    repository: str | None = None
    published_at: str | None = None
    license: str | None = None


class EngineLock(_Model):
    version: Literal[1]
    release: Release
    variants: list[Variant] = Field(min_length=1)

    def find(self, variant_id: str) -> Variant | None:
        return next((v for v in self.variants if v.id == variant_id), None)


_cache: dict[str, tuple[float, EngineLock]] = {}


def load_lock(path: str | os.PathLike | None = None) -> EngineLock:
    path = Path(path) if path else DEFAULT_LOCK_PATH
    try:
        mtime = os.path.getmtime(path)
    except OSError as exc:
        raise RegistryError(f"Engine lock file not found: {path}") from exc
    cached = _cache.get(str(path))
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        lock = EngineLock.model_validate(raw)
    except (yaml.YAMLError, ValidationError) as exc:
        raise RegistryError(f"{path.name} is not valid: {exc}") from exc
    ids = [v.id for v in lock.variants]
    if len(ids) != len(set(ids)):
        raise RegistryError(f"{path.name} lists a variant id twice")
    _cache[str(path)] = (mtime, lock)
    return lock
