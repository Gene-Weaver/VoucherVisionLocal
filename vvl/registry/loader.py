"""Load and validate the two registry files.

    vvl/registry/providers.yaml     credentials and API providers (Keys tab, API models)
    vvl/registry/models_local.yaml  verified local model builds (Local models)

Both are validated with pydantic. A broken registry is a packaging bug, so the
loaders raise RegistryError with a readable message instead of guessing.
Results are cached per (path, modification time).
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from vvl import paths

ID_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"
BUILD_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"
FIELD_NAME_PATTERN = r"^[a-z][a-z0-9_]*$"
AUTH_PATTERN = re.compile(r"^(bearer|header:[A-Za-z0-9-]+|query:[A-Za-z0-9_]+)$")
SHA256_PATTERN = r"^[0-9a-f]{64}$"
COLOR_PATTERN = r"^#[0-9a-fA-F]{6}$"
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class RegistryError(ValueError):
    """A registry file is missing or does not match the expected structure."""


def _check_url(value: str, *, allow_template: bool = False) -> str:
    """https everywhere; plain http only for loopback (local test servers)."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("must be a non-empty URL")
    probe = value.replace("{base_url}", "https://example.invalid") if allow_template else value
    parsed = urlparse(probe)
    if parsed.scheme == "https" and parsed.hostname:
        return value
    if parsed.scheme == "http" and parsed.hostname in _LOOPBACK_HOSTS:
        return value
    raise ValueError(f"must be an https URL (got {value!r})")


class _Model(BaseModel):
    # Unknown keys are kept, so the registry can grow without code changes.
    model_config = ConfigDict(extra="allow", frozen=True)


# ---------------------------------------------------------------- providers --

class CredentialField(_Model):
    name: str = Field(pattern=FIELD_NAME_PATTERN)
    label: str
    placeholder: str = ""


class RequestSpec(_Model):
    method: Literal["GET", "POST"] = "GET"
    url: str
    auth: str
    extra_headers: dict[str, str] = Field(default_factory=dict)
    ok_status: list[int] = Field(default_factory=lambda: [200])

    @field_validator("url")
    @classmethod
    def _url(cls, value):
        return _check_url(value, allow_template=True)

    @field_validator("auth")
    @classmethod
    def _auth(cls, value):
        if not AUTH_PATTERN.match(value or ""):
            raise ValueError("auth must be 'bearer', 'header:<Name>', or 'query:<name>'")
        return value


class ModelsSpec(_Model):
    source: Literal["list_endpoint", "catalog", "none"] = "none"
    url: str | None = None
    auth: str | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)
    id_path: str | None = None
    strip_prefix: str = ""

    @model_validator(mode="after")
    def _complete(self):
        if self.source == "list_endpoint":
            missing = [k for k in ("url", "auth", "id_path") if not getattr(self, k)]
            if missing:
                raise ValueError(f"list_endpoint needs {', '.join(missing)}")
            _check_url(self.url, allow_template=True)
            if not AUTH_PATTERN.match(self.auth):
                raise ValueError("auth must be 'bearer', 'header:<Name>', or 'query:<name>'")
        return self


class ProviderSpec(_Model):
    id: str = Field(pattern=ID_PATTERN)
    label: str
    role: Literal["model_provider", "enrichment", "downloads"]
    kind: Literal["openai_compat", "openrouter", "gemini", "anthropic", "vvgo", "huggingface"]
    purpose: str = ""
    sends_images: bool
    optional: bool = True
    base_url: str
    key_url: str | None = None
    docs_url: str | None = None
    fields: list[CredentialField] = Field(min_length=1)
    test: RequestSpec | None = None
    models: ModelsSpec = Field(default_factory=ModelsSpec)
    enabled: bool = True
    # The company's brand color; the model pickers tint this provider's group with it.
    brand_color: str | None = Field(default=None, pattern=COLOR_PATTERN)

    @field_validator("base_url")
    @classmethod
    def _base_url(cls, value):
        return _check_url(value).rstrip("/")

    @model_validator(mode="after")
    def _unique_fields(self):
        names = [f.name for f in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("field names must be unique")
        return self

    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]


class ProvidersFile(_Model):
    version: Literal[1]
    providers: list[ProviderSpec]

    @model_validator(mode="after")
    def _unique_ids(self):
        ids = [p.id for p in self.providers]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate provider ids: {', '.join(dupes)}")
        return self


# ------------------------------------------------------------- local models --

# A file inside the model repository, optionally in a subfolder ("BF16/x.gguf").
# Every segment starts with a letter or digit, so "." and ".." cannot occur.
FILE_PATH_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)*$"
SPLIT_PART = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


class ModelFile(_Model):
    role: Literal["model", "mmproj"]
    path: str = Field(pattern=FILE_PATH_PATTERN)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=SHA256_PATTERN)


class InferenceStep(BaseModel):
    """ocr / parse options inside an `inference:` block (see vvl/llama/inference.py)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    max_tokens: int | None = Field(default=None, gt=0)
    sampling: dict[str, float | int | bool | str | None] | None = None
    image_first: bool | None = None      # ocr only: image before the prompt text (default true)


class InferenceSpec(BaseModel):
    """Optional per-family / per-build engine and sampling settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    context_size: int | None = Field(default=None, ge=512)
    image_min_tokens: int | None = Field(default=None, gt=0)
    image_max_tokens: int | None = Field(default=None, gt=0)
    kv_bytes_per_token: int | None = Field(default=None, gt=0)
    request_timeout_seconds: int | None = Field(default=None, gt=0)
    server_args: list[str] | None = None
    ocr: InferenceStep | None = None
    parse: InferenceStep | None = None


class LocalBuild(_Model):
    id: str = Field(pattern=BUILD_ID_PATTERN)
    label: str
    params: str
    quant: str
    tier: Literal["small", "medium", "large"]
    hf_repo: str = Field(pattern=r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
    hf_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    context_default: int = Field(gt=0)
    files: list[ModelFile] = Field(min_length=1)
    default_for_test: bool = False
    inference: InferenceSpec | None = None

    @model_validator(mode="after")
    def _one_model_file(self):
        roles = [f.role for f in self.files]
        models = [f for f in self.files if f.role == "model"]
        if len(models) > 1:
            # A model split into parts ("x-00001-of-00003.gguf", ...): listed in
            # order, all in one folder. The engine is given the first part.
            parts = [SPLIT_PART.search(f.path) for f in models]
            if not all(parts):
                raise ValueError("a build has one 'model' file, or the numbered parts of one split model")
            count = int(parts[0].group(2))
            numbers = [int(m.group(1)) for m in parts]
            folders = {f.path.rsplit("/", 1)[0] if "/" in f.path else "" for f in models}
            if numbers != list(range(1, count + 1)) or any(int(m.group(2)) != count for m in parts) \
                    or len(folders) != 1:
                raise ValueError("split model parts must be complete, in order, and in one folder")
        if not models:
            raise ValueError("a build needs a file with role 'model'")
        if roles.count("mmproj") > 1:
            raise ValueError("a build has at most one 'mmproj' file")
        return self

    @property
    def total_size_bytes(self) -> int:
        return sum(f.size_bytes for f in self.files)


class LocalFamily(_Model):
    id: str = Field(pattern=ID_PATTERN)
    label: str
    developer: str = ""
    license: str | None = None
    status: Literal["verified", "planned"]
    roles: list[Literal["ocr", "parse"]] = Field(min_length=1)
    summary: str = ""
    homepage: str | None = None
    # The developer's brand color; the model pickers tint this family's group with it.
    brand_color: str | None = Field(default=None, pattern=COLOR_PATTERN)
    # The family listed first in the model pickers, with its group open.
    primary_family: bool = False
    inference: InferenceSpec | None = None
    builds: list[LocalBuild] = Field(default_factory=list)

    @model_validator(mode="after")
    def _verified_has_builds(self):
        if self.status == "verified" and not self.builds:
            raise ValueError(f"verified family {self.id!r} has no builds")
        return self


class LocalModelsFile(_Model):
    version: Literal[1]
    families: list[LocalFamily]

    @model_validator(mode="after")
    def _unique_ids(self):
        fam_ids = [f.id for f in self.families]
        build_ids = [b.id for f in self.families for b in f.builds]
        for label, ids in (("family", fam_ids), ("build", build_ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            if dupes:
                raise ValueError(f"duplicate {label} ids: {', '.join(dupes)}")
        defaults = [b for f in self.families for b in f.builds if b.default_for_test]
        if len(defaults) > 1:
            raise ValueError("at most one build may set default_for_test")
        if sum(1 for f in self.families if f.primary_family) > 1:
            raise ValueError("at most one family may set primary_family")
        return self

    def builds(self) -> list[tuple[LocalFamily, LocalBuild]]:
        return [(f, b) for f in self.families for b in f.builds]

    def find_build(self, build_id: str) -> tuple[LocalFamily, LocalBuild] | None:
        for family, build in self.builds():
            if build.id == build_id:
                return family, build
        return None


# ------------------------------------------------------------------ loading --

DEFAULT_PROVIDERS_PATH = paths.registry_dir() / "providers.yaml"
DEFAULT_LOCAL_MODELS_PATH = paths.registry_dir() / "models_local.yaml"
DEFAULT_API_COST_PATH = paths.registry_dir() / "api_cost.yaml"

_cache: dict[tuple[str, str], tuple[float, Any]] = {}
_cache_lock = threading.Lock()


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    lines = []
    for err in exc.errors()[:10]:
        where = ".".join(str(p) for p in err.get("loc", ())) or "(top level)"
        lines.append(f"  {where}: {err.get('msg')}")
    return f"{path.name} is not valid:\n" + "\n".join(lines)


def _load(kind: str, path: Path, model: type[BaseModel]):
    path = Path(path)
    try:
        mtime = os.path.getmtime(path)
    except OSError as exc:
        raise RegistryError(f"Registry file not found: {path}") from exc
    key = (kind, str(path))
    with _cache_lock:
        cached = _cache.get(key)
        if cached and cached[0] == mtime:
            return cached[1]
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise RegistryError(f"{path.name} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise RegistryError(f"{path.name} must be a YAML mapping")
    try:
        parsed = model.model_validate(raw)
    except ValidationError as exc:
        raise RegistryError(_format_validation_error(path, exc)) from exc
    with _cache_lock:
        _cache[key] = (mtime, parsed)
    return parsed


def load_providers(path: str | os.PathLike | None = None) -> ProvidersFile:
    return _load("providers", Path(path) if path else DEFAULT_PROVIDERS_PATH, ProvidersFile)


def load_local_models(path: str | os.PathLike | None = None) -> LocalModelsFile:
    return _load("models_local", Path(path) if path else DEFAULT_LOCAL_MODELS_PATH, LocalModelsFile)


def enabled_providers(path: str | os.PathLike | None = None) -> list[ProviderSpec]:
    return [p for p in load_providers(path).providers if p.enabled]


def find_provider(provider_id: str, path: str | os.PathLike | None = None) -> ProviderSpec | None:
    for provider in enabled_providers(path):
        if provider.id == provider_id:
            return provider
    return None


def load_api_costs(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else DEFAULT_API_COST_PATH
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise RegistryError(f"{path.name} must be a YAML mapping")
    return data


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()
