#!/usr/bin/env python3
"""A VoucherVisionLocal backend for UI work that never downloads anything real.

It starts two things:

  * a loopback file server that plays the part of Hugging Face and of the
    llama.cpp release page, serving small random files slowly enough that a
    progress bar is visible for several seconds;
  * the normal VVL server (the same code path as `vvl serve`), pointed at a
    scratch models folder, a scratch model list, and a scratch engine lock
    whose "llama-server" is tests/fakes/fake_llama_server.py.

Every path VVL writes to (home, credentials, settings, user prompts, models)
is inside --root, so the real files are never created or changed.

    uv run python tools/dev_fake_backend.py --root /some/scratch/dir

What the fake catalog contains

  Models (family "Demo-VL")
    demo-small     downloads and verifies normally
    demo-medium    a larger download (good for cancel and resume)
    demo-badsum    the served bytes do not match the checksum (CHECKSUM_MISMATCH)
    demo-huge      claims to be enormous: the download is refused (INSUFFICIENT_DISK_SPACE)

  Engine builds (for Linux x64, macOS, and Windows alike: the fake is a script)
    *-cuda12       installs, then fails its self-test; the message names the next build
    *-vulkan       installs and runs
    *-cpu          installs and runs

  Label detector
    one 8 MB random file in place of the real model; it downloads at first start
    (at --rate) and then a fixed-box stand-in (tests/fakes/fake_detector.py)
    finds the "labels"

Useful switches: --preinstall-engine and --preinstall-model put a working
engine / the demo-small model on disk at start; --rate sets the download
speed; --load-seconds and --reply-seconds slow the fake engine down so the
"loading" and "reading the label" states can be seen.
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import io
import json
import os
import platform
import stat
import sys
import tarfile
import threading
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FAKE_LLAMA = REPO / "tests" / "fakes" / "fake_llama_server.py"
HF_REPO = "DemoOrg/Demo-VL-GGUF"
HF_REVISION = "0123456789abcdef0123456789abcdef01234567"
ENGINE_TAG = "b12359"
MB = 1000 * 1000


def blob(size: int, seed: int) -> bytes:
    """Deterministic, incompressible bytes."""
    out = bytearray()
    counter = 0
    while len(out) < size:
        out += hashlib.sha256(f"{seed}-{counter}".encode()).digest()
        counter += 1
    return bytes(out[:size])


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- file server

class ThrottledFiles:
    """Serves in-memory files with Range support at a fixed speed."""

    def __init__(self, port: int, rate_bytes_per_second: int):
        self.files: dict[str, bytes] = {}
        self.rate = max(1, int(rate_bytes_per_second))
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                path = self.path.partition("?")[0]
                data = outer.files.get(path)
                if data is None:
                    body = b'{"error": "not found"}'
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                start = 0
                header = self.headers.get("Range")
                if header and header.startswith("bytes="):
                    try:
                        start = int(header[6:].split("-", 1)[0])
                    except ValueError:
                        start = 0
                    if start >= len(data):
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{len(data)}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    self.send_response(200)
                body = data[start:]
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                step = max(4096, outer.rate // 10)   # ten pieces a second
                sent = 0
                try:
                    while sent < len(body):
                        piece = body[sent:sent + step]
                        self.wfile.write(piece)
                        self.wfile.flush()
                        sent += len(piece)
                        time.sleep(len(piece) / outer.rate)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            do_HEAD = do_GET

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"

    def start(self):
        threading.Thread(target=self.httpd.serve_forever, name="fake-files", daemon=True).start()
        return self


# ---------------------------------------------------------------- fake models

def model_url_path(name: str) -> str:
    return f"/{HF_REPO}/resolve/{HF_REVISION}/{name}"


def build_models(files: ThrottledFiles, registry_path: Path) -> dict:
    data = {
        "demo-small.gguf": blob(24 * MB, 1),
        "demo-medium.gguf": blob(72 * MB, 2),
        "demo-badsum.gguf": blob(12 * MB, 3),
        "mmproj-demo.gguf": blob(6 * MB, 4),
    }
    for name, content in data.items():
        files.files[model_url_path(name)] = content

    def entry(role, name, *, sha256=None, size=None):
        return {"role": role, "path": name, "size_bytes": size or len(data[name]),
                "sha256": sha256 or sha(data[name])}

    def build(build_id, label, params, tier, model_file):
        return {"id": build_id, "label": label, "params": params, "quant": "Q8_0", "tier": tier,
                "hf_repo": HF_REPO, "hf_revision": HF_REVISION, "context_default": 4096,
                "files": [model_file, entry("mmproj", "mmproj-demo.gguf")]}

    registry = {
        "version": 1,
        "families": [{
            "id": "demo-vl", "label": "Demo-VL", "developer": "Fake models for interface work", "license": "mit",
            "status": "verified", "roles": ["ocr", "parse"],
            "summary": "Small random files that stand in for real models. Nothing here can transcribe a label.",
            "inference": {"image_max_tokens": 1024, "ocr": {"max_tokens": 512, "sampling": {"temperature": 0.0}}},
            "builds": [
                build("demo-small", "Demo-VL Small (Q8)", "1B", "small", entry("model", "demo-small.gguf")),
                build("demo-medium", "Demo-VL Medium (Q8)", "4B", "medium", entry("model", "demo-medium.gguf")),
                # The checksum on record belongs to other bytes: the download fails verification.
                build("demo-badsum", "Demo-VL Bad Checksum (Q8)", "2B", "small",
                      entry("model", "demo-badsum.gguf", sha256=sha(b"not these bytes"))),
                # Claims 4 PB, so the free-space check refuses it before any byte moves.
                build("demo-huge", "Demo-VL Huge (Q8)", "900B", "large",
                      entry("model", "demo-huge.gguf", sha256=sha(b"never served"), size=4 * 10 ** 15)),
            ],
        }],
    }
    import yaml

    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    return data


def preinstall_model(models_dir: Path, data: dict) -> None:
    from vvl.local_models import record_verified

    folder = models_dir / "models" / HF_REPO
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("demo-small.gguf", "mmproj-demo.gguf"):
        path = folder / name
        path.write_bytes(data[name])
        record_verified(str(models_dir), path, sha(data[name]))


# ---------------------------------------------------------------- fake engine

def server_script(mode: str, load_seconds: float, reply_seconds: float) -> str:
    lines = ["#!/bin/sh"]
    for key, value in (("FAKE_LLAMA_MODE", mode), ("FAKE_LLAMA_LOAD_SECONDS", load_seconds),
                       ("FAKE_LLAMA_REPLY_DELAY", reply_seconds)):
        lines.append(f': "${{{key}:={value}}}"')
        lines.append(f"export {key}")
    lines.append(f'exec "{sys.executable}" -I "{FAKE_LLAMA}" "$@"')
    return "\n".join(lines) + "\n"


def tar_bytes(members: list[tuple]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, kind, payload, mode in members:
            info = tarfile.TarInfo(name)
            info.mode = mode
            if kind == "dir":
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            else:
                raw = payload if isinstance(payload, bytes) else payload.encode()
                info.size = len(raw)
                tar.addfile(info, io.BytesIO(raw))
    return buffer.getvalue()


def zip_bytes(members: list[tuple]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, _kind, payload, mode in members:
            info = zipfile.ZipInfo(name)
            info.external_attr = (stat.S_IFREG | mode) << 16
            archive.writestr(info, payload)
    return buffer.getvalue()


def host_os_arch() -> tuple[str, str]:
    system = {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}.get(platform.system(), "linux")
    arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x86_64"
    return system, arch


def fake_detector_factory():
    sys.path.insert(0, str(REPO / "tests"))
    from fakes.fake_detector import FakeDetector

    return FakeDetector


def build_detector_lock(files: ThrottledFiles, lock_path: Path) -> None:
    """A copy of the real detector lock whose every format is one small random file on the fake server."""
    import yaml

    from vvl.label_detector import load_lock

    lock = load_lock()
    data = blob(8 * MB, 5)
    for fmt in lock["formats"]:
        entry = {"path": f"{fmt}/fake-detector.bin", "size_bytes": len(data), "sha256": sha(data)}
        lock["formats"][fmt] = {"files": [entry]}
        files.files[f"/{lock['repo']}/resolve/{lock['revision']}/{entry['path']}"] = data
    lock_path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")


def build_engine_lock(files: ThrottledFiles, lock_path: Path, load_seconds: float, reply_seconds: float) -> None:
    system, arch = host_os_arch()
    prefix = f"{system}-{'x64' if arch == 'x86_64' else 'arm64'}"
    folder = f"llama-{ENGINE_TAG}"
    filler = blob(18 * MB, 9)    # makes the archive big enough to watch it download

    def engine_tar(mode):
        return tar_bytes([
            (folder, "dir", None, 0o755),
            (f"{folder}/llama-server", "file", server_script(mode, load_seconds, reply_seconds), 0o755),
            (f"{folder}/libfake.so", "file", filler, 0o644),
        ])

    archives = {
        "fake-engine-cuda.tar.gz": engine_tar("selftest_glibc"),
        "fake-cudart.tar.gz": tar_bytes([("libcudart.so.12", "file", blob(6 * MB, 10), 0o644)]),
        "fake-engine-vulkan.tar.gz": engine_tar("ready"),
        "fake-engine-cpu.zip": zip_bytes([
            ("llama-server", "file", server_script("ready", load_seconds, reply_seconds).encode(), 0o755),
            ("libfake.so", "file", blob(8 * MB, 11), 0o644),
        ]),
    }
    for name, content in archives.items():
        files.files[f"/release/{name}"] = content

    def asset(kind, name):
        return {"kind": kind, "name": name, "url": f"{files.base_url}/release/{name}",
                "size_bytes": len(archives[name]), "sha256": sha(archives[name])}

    lock = {
        "version": 1,
        "release": {"tag": ENGINE_TAG, "stable_alias": "v1.0.0", "repository": "https://example.invalid/llama.cpp"},
        "variants": [
            {"id": f"{prefix}-cuda12", "os": system, "arch": arch, "backend": "cuda",
             "label": "Fake build, NVIDIA CUDA 12",
             "assets": [asset("engine", "fake-engine-cuda.tar.gz"), asset("runtime", "fake-cudart.tar.gz")]},
            {"id": f"{prefix}-vulkan", "os": system, "arch": arch, "backend": "vulkan",
             "label": "Fake build, Vulkan", "assets": [asset("engine", "fake-engine-vulkan.tar.gz")]},
            {"id": f"{prefix}-cpu", "os": system, "arch": arch, "backend": "cpu",
             "label": "Fake build, CPU only", "assets": [asset("engine", "fake-engine-cpu.zip")]},
        ],
    }
    import yaml

    lock_path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")


def preinstall_engine(models_dir: Path, load_seconds: float, reply_seconds: float) -> None:
    system, arch = host_os_arch()
    variant = f"{system}-{'x64' if arch == 'x86_64' else 'arm64'}-vulkan"
    folder = models_dir / "engine" / ENGINE_TAG / variant / f"llama-{ENGINE_TAG}"
    folder.mkdir(parents=True, exist_ok=True)
    server = folder / "llama-server"
    server.write_text(server_script("ready", load_seconds, reply_seconds), encoding="utf-8")
    os.chmod(server, 0o755)
    manifest = {
        "tag": ENGINE_TAG, "variant": variant, "label": "Fake build, Vulkan", "backend": "vulkan", "os": system,
        "arch": arch, "server": f"llama-{ENGINE_TAG}/llama-server", "installed_at": "2026-11-19T00:00:00Z",
        "version": f"{ENGINE_TAG} (abc1234)",
        "devices": [{"name": "Vulkan0", "description": "Fake GPU A"}, {"name": "Vulkan1", "description": "Fake GPU B"}],
        "assets": [],
    }
    (folder.parent / "vvl-install.json").write_text(json.dumps(manifest), encoding="utf-8")


# ---------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, help="Scratch folder for everything this run writes.")
    parser.add_argument("--port", type=int, default=8653, help="VVL port (default 8653).")
    parser.add_argument("--files-port", type=int, default=8654, help="Fake download server port (default 8654).")
    parser.add_argument("--token", default="dev-fake-token", help="Session token (default dev-fake-token).")
    parser.add_argument("--rate", type=float, default=4.0, help="Download speed in MB per second (default 4).")
    parser.add_argument("--load-seconds", type=float, default=5.0, help="How long the fake engine 'loads' a model.")
    parser.add_argument("--reply-seconds", type=float, default=4.0, help="How long each fake engine reply takes.")
    parser.add_argument("--preinstall-engine", action="store_true", help="Start with a working fake engine installed.")
    parser.add_argument("--preinstall-model", action="store_true", help="Start with demo-small installed.")
    parser.add_argument("--models-dir", default=None, help="Models folder (default <root>/models_dir).")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    real_repo_files = {REPO / "PRIVATE.yaml", REPO / "settings.local.yaml"}
    if root == REPO or REPO in root.parents:
        print("--root must be outside the repository.", file=sys.stderr)
        return 2
    root.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir).resolve() if args.models_dir else root / "models_dir"
    models_dir.mkdir(parents=True, exist_ok=True)
    settings_path = root / "settings.yaml"
    private_path = root / "PRIVATE.yaml"
    assert settings_path not in real_repo_files and private_path not in real_repo_files

    # The four path overrides are read from the environment (vvl/paths.py); set
    # them before anything from the backend is imported.
    os.environ.update({
        "VVL_HOME": str(root / "home"),
        "VVL_DETECTOR_GPU": "0",          # never the real GPU runtime download from the fake backend
        "VVL_PRIVATE_PATH": str(private_path),
        "VVL_SETTINGS_PATH": str(settings_path),
        "VVL_PROMPTS_USER_DIR": str(root / "prompts_user"),
        "VVL_TOKEN": args.token,
        "VVL_PORT": str(args.port),
    })
    os.environ.pop("VVL_OWNER_PID", None) if not os.environ.get("VVL_KEEP_OWNER_PID") else None

    import yaml

    current = {}
    if settings_path.exists():
        current = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    current["models_dir"] = str(models_dir)
    settings_path.write_text(yaml.safe_dump(current), encoding="utf-8")

    files = ThrottledFiles(args.files_port, int(args.rate * MB)).start()
    os.environ["VVL_HF_BASE_URL"] = files.base_url
    registry_path = root / "models_local.yaml"
    lock_path = root / "engine.lock.yaml"
    detector_lock_path = root / "archival_detector.lock.yaml"

    sys.path.insert(0, str(REPO))
    data = build_models(files, registry_path)
    build_engine_lock(files, lock_path, args.load_seconds, args.reply_seconds)
    build_detector_lock(files, detector_lock_path)
    if args.preinstall_model:
        preinstall_model(models_dir, data)
    if args.preinstall_engine:
        preinstall_engine(models_dir, args.load_seconds, args.reply_seconds)

    # The model list and engine lock overrides are app settings, not environment
    # variables, so the app is built here and handed to the normal serve().
    import app as app_module
    from vvl.server import serve

    app_module.app = app_module.create_app(token=args.token, config={
        "VVL_SETTINGS_PATH": str(settings_path),
        "VVL_PRIVATE_PATH": str(private_path),
        "VVL_LOCAL_MODELS_PATH": str(registry_path),
        "VVL_ENGINE_LOCK_PATH": str(lock_path),
        "VVL_HF_BASE_URL": files.base_url,
        "VVL_DETECTOR_LOCK_PATH": str(detector_lock_path),
        "VVL_DETECTOR_FACTORY": fake_detector_factory(),
    })
    print(f"Fake downloads: {files.base_url} at {args.rate:g} MB/s. Scratch folder: {root}", flush=True)
    return serve(port=args.port)


if __name__ == "__main__":
    sys.exit(main())
