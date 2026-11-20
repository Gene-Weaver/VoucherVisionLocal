"""Helpers for the Phase 2 tests (downloads, engine install, supervisor).

Nothing here touches the network or real model files: registries point at the
loopback FakeServer, the "engine" is tests/fakes/fake_llama_server.py, and
model files are a few hundred kilobytes of random bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures"
FAKE_LLAMA = REPO / "tests" / "fakes" / "fake_llama_server.py"
WORKSTATION_REPORT = FIXTURES / "system_reports" / "linux_workstation_2x_rtx6000ada.json"
REVISION = "0123456789abcdef0123456789abcdef01234567"
REPO_ID = "TestOrg/Test-VL-GGUF"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def blob(size: int, seed: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < size:
        out += hashlib.sha256(f"{seed}-{counter}".encode()).digest()
        counter += 1
    return bytes(out[:size])


def load_report(path: Path = WORKSTATION_REPORT) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class FakeModels:
    """Two builds that share one vision projector, served by a FakeServer."""

    def __init__(self, tmp_path: Path, server=None, *, model_a=300_000, model_b=200_000, mmproj=150_000):
        self.data = {
            "model-a.gguf": blob(model_a, 1),
            "model-b.gguf": blob(model_b, 2),
            "mmproj-shared.gguf": blob(mmproj, 3),
        }
        self.registry_path = tmp_path / "models_local.yaml"
        registry = {
            "version": 1,
            "families": [{
                "id": "testvl", "label": "TestVL", "developer": "Tests", "license": "mit", "status": "verified",
                "roles": ["ocr", "parse"], "summary": "Fake models for tests.",
                "inference": {"image_max_tokens": 1024, "ocr": {"max_tokens": 512, "sampling": {"temperature": 0.0}}},
                "builds": [
                    self._build("test-a", "Test A", "model-a.gguf"),
                    self._build("test-b", "Test B", "model-b.gguf"),
                ],
            }],
        }
        self.registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
        if server is not None:
            self.serve(server)

    def _build(self, build_id, label, model_name):
        return {
            "id": build_id, "label": label, "params": "1B", "quant": "Q8_0", "tier": "small",
            "hf_repo": REPO_ID, "hf_revision": REVISION, "context_default": 4096,
            "files": [
                {"role": "model", "path": model_name, "size_bytes": len(self.data[model_name]),
                 "sha256": sha(self.data[model_name])},
                {"role": "mmproj", "path": "mmproj-shared.gguf", "size_bytes": len(self.data["mmproj-shared.gguf"]),
                 "sha256": sha(self.data["mmproj-shared.gguf"])},
            ],
        }

    def url_path(self, name: str) -> str:
        return f"/{REPO_ID}/resolve/{REVISION}/{name}"

    def serve(self, server, **route_options):
        from conftest import FileRoute

        for name, data in self.data.items():
            server.files[self.url_path(name)] = FileRoute(data, **route_options.get(name, {}))

    def total(self, build_id: str) -> int:
        model = "model-a.gguf" if build_id == "test-a" else "model-b.gguf"
        return len(self.data[model]) + len(self.data["mmproj-shared.gguf"])

    def install_on_disk(self, models_dir: Path, build_id: str) -> None:
        """Put verified files on disk without downloading (for supervisor tests)."""
        from vvl.local_models import record_verified

        folder = Path(models_dir) / "models" / REPO_ID
        folder.mkdir(parents=True, exist_ok=True)
        names = ["model-a.gguf" if build_id == "test-a" else "model-b.gguf", "mmproj-shared.gguf"]
        for name in names:
            path = folder / name
            path.write_bytes(self.data[name])
            record_verified(models_dir, path, sha(self.data[name]))


def write_settings(path: Path, models_dir: Path, **engine) -> None:
    data = {"models_dir": str(models_dir)}
    if engine:
        data["engine"] = engine
    Path(path).write_text(yaml.safe_dump(data), encoding="utf-8")


def make_runtime(settings_path: Path, registry_path: Path, *, hf_base_url: str | None = None,
                 report: dict | None = None, **extra):
    from vvl.runtime import Runtime

    config = {
        "VVL_SETTINGS_PATH": str(settings_path),
        "VVL_LOCAL_MODELS_PATH": str(registry_path),
        "VVL_SYSTEM_REPORT": (lambda refresh=False: report if report is not None else load_report()),
    }
    if hf_base_url:
        config["VVL_HF_BASE_URL"] = hf_base_url
    config.update(extra)
    return Runtime(config)


def wait_job(job, timeout: float = 20.0):
    end = time.monotonic() + timeout
    while job.active and time.monotonic() < end:
        time.sleep(0.02)
    assert not job.active, f"job {job.id} still {job.state}"
    return job.to_dict()


def wait_until(predicate, timeout: float = 15.0, interval: float = 0.02):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError("condition not met in time")


def fake_engine_install(models_dir: Path, *, tag: str = "b12359", variant: str = "linux-x64-cuda12",
                        backend: str = "cuda", state_dir: Path | None = None, mode: str = "ready",
                        env: dict | None = None) -> Path:
    """An 'installed' engine whose llama-server is the fake script. Returns its path."""
    folder = Path(models_dir) / "engine" / tag / variant / f"llama-{tag}"
    folder.mkdir(parents=True, exist_ok=True)
    server = folder / "llama-server"
    exports = {"FAKE_LLAMA_MODE": mode}
    if state_dir is not None:
        exports["FAKE_LLAMA_STATE_DIR"] = str(state_dir)
    exports.update(env or {})
    lines = ["#!/bin/sh"]
    for key, value in exports.items():
        lines.append(f": \"${{{key}:={value}}}\"")
        lines.append(f"export {key}")
    lines.append(f'exec "{sys.executable}" -I "{FAKE_LLAMA}" "$@"')
    server.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(server, 0o755)
    manifest = {
        "tag": tag, "variant": variant, "label": f"Fake {variant}", "backend": backend, "os": "linux",
        "arch": "x86_64", "server": f"llama-{tag}/llama-server", "installed_at": "2026-11-19T00:00:00Z",
        "version": "b12359 (abc1234)",
        "devices": [{"name": "CUDA0", "description": "NVIDIA RTX 6000 Ada Generation"},
                    {"name": "CUDA1", "description": "NVIDIA RTX 6000 Ada Generation"}],
        "assets": [],
    }
    (folder.parent / "vvl-install.json").write_text(json.dumps(manifest), encoding="utf-8")
    return server
