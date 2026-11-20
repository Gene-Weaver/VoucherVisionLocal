"""Install, self-test, and uninstall the pinned llama.cpp engine.

Layout:

    <models_dir>/engine/<tag>/<variant>/              the extracted build
    <models_dir>/engine/<tag>/<variant>/vvl-install.json   install manifest
    <models_dir>/engine/<tag>/downloads/<asset>[.partial]  archives while installing
    <VVL_HOME>/engine/selftest_failures.json          builds that failed here

Install steps (one `engine_install` job):
  1. download every asset of the variant (resumable, sha256-verified);
  2. extract the engine archive into a staging folder (vvl.llama.extract);
     the CUDA runtime archive (cudart) is unpacked beside `llama-server`,
     where the engine's loader looks for it;
  3. clear the macOS quarantine attribute;
  4. self-test: `llama-server --version` and `llama-server --list-devices`.
     A GPU build must list at least one device of its backend; if it does
     not, the job fails with ENGINE_SELF_TEST_FAILED, the failure is recorded
     (so the recommendation moves to the next build), and the message names
     the next build to try;
  5. move the staging folder into place and write the manifest.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import secrets
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable

from vvl import downloads, paths
from vvl.fileutil import atomic_write_text
from vvl.jobs import Job, JobBusy, JobCanceled, JobError, JobManager, utc_now_iso
from vvl.llama import extract as extract_mod
from vvl.llama import variants as variants_mod
from vvl.llama.lock import EngineLock, Variant
from vvl.local_models import StoreError

logger = logging.getLogger(__name__)

JOB_KIND = "engine_install"
MANIFEST_NAME = "vvl-install.json"
SELFTEST_TIMEOUT_SECONDS = 90
DISK_FACTOR = 4                 # archives + unpacked files (CUDA libraries unpack to ~3x)
DISK_MARGIN_BYTES = 1024 ** 3
SERVER_NAMES = ("llama-server", "llama-server.exe")

# `llama-server --list-devices` lines look like
#   "  CUDA0: NVIDIA RTX 6000 Ada Generation (48526 MiB, 47881 MiB free)"
_DEVICE_RE = re.compile(r"^\s*(?P<name>[A-Za-z][A-Za-z0-9_]*?\d+):\s*(?P<desc>.+?)"
                        r"(?:\s*\((?P<total>\d+)\s*MiB,\s*(?P<free>\d+)\s*MiB free\))?\s*$")
BACKEND_DEVICE_PREFIXES = {"cuda": ("CUDA",), "vulkan": ("Vulkan",), "metal": ("Metal", "MTL"), "cpu": ()}
# "version: 1.0.0-dev (build 12359, commit b1b770d90)" (b12359) or "version: 6123 (abc1234)" (older)
VERSION_RE = re.compile(r"version:\s*[^\s(]*\s*\(build\s+(?P<number>\d+)(?:,\s*commit\s+(?P<commit>[0-9a-fA-F]+))?\)")
VERSION_RE_OLD = re.compile(r"version:\s*(?P<number>\d+)\s*\((?P<commit>[0-9a-fA-F]+)\)")
_GLIBC_RE = re.compile(r"version `(GLIBC|GLIBCXX)_([0-9.]+)' not found")

# Engine output that means "this build cannot use this GPU/driver" (used by the
# installer's self-test and by the supervisor when a started engine fails).
INCOMPATIBLE_PATTERNS = (
    (re.compile(r"version `GLIBC_[0-9.]+' not found", re.I),
     "this build needs a newer Linux system library (glibc)"),
    (re.compile(r"version `GLIBCXX_[0-9.]+' not found", re.I),
     "this build needs a newer Linux C++ library (libstdc++)"),
    (re.compile(r"CUDA driver version is insufficient", re.I),
     "the NVIDIA driver is too old for this CUDA build"),
    (re.compile(r"unsupported toolchain|PTX was compiled with an unsupported", re.I),
     "the NVIDIA driver is too old for this CUDA build"),
    (re.compile(r"no kernel image is available", re.I),
     "this CUDA build has no code for this GPU"),
    (re.compile(r"cudaErrorInsufficientDriver|CUDA_ERROR_SYSTEM_DRIVER_MISMATCH", re.I),
     "the NVIDIA driver does not match this CUDA build"),
    (re.compile(r"error while loading shared libraries", re.I),
     "a library the engine needs is missing"),
)


def local_glibc() -> str | None:
    try:
        value = os.confstr("CS_GNU_LIBC_VERSION")  # "glibc 2.35"
    except (AttributeError, ValueError, OSError):
        return None
    return value.split()[-1] if value else None


def classify_incompatible(text: str) -> str | None:
    """A plain-language reason when engine output shows it cannot run here."""
    text = text or ""
    needed = [m for m in _GLIBC_RE.finditer(text) if m.group(1) == "GLIBC"]
    if needed:
        want = max(needed, key=lambda m: tuple(int(x) for x in m.group(2).split(".") if x.isdigit())).group(2)
        have = local_glibc()
        return (f"this build needs glibc {want} or newer (Ubuntu 24.04 or a similarly recent Linux)"
                + (f"; this computer has glibc {have}" if have else ""))
    for pattern, meaning in INCOMPATIBLE_PATTERNS:
        if pattern.search(text):
            return meaning
    return None


def parse_devices(text: str) -> list[dict]:
    devices = []
    for line in (text or "").splitlines():
        match = _DEVICE_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        if name.lower().startswith(("available", "version", "build", "load")):
            continue
        entry = {"name": name, "description": match.group("desc").strip()}
        if match.group("total"):
            entry["total_mib"] = int(match.group("total"))
            entry["free_mib"] = int(match.group("free"))
        devices.append(entry)
    return devices


def parse_version(text: str) -> str | None:
    match = VERSION_RE.search(text or "") or VERSION_RE_OLD.search(text or "")
    if match:
        commit = match.group("commit")
        return f"b{match.group('number')}" + (f" ({commit})" if commit else "")
    first = (text or "").strip().splitlines()
    return first[0][:200] if first else None


def engine_env(bin_dir: Path, base: dict | None = None) -> dict:
    """Environment for running the engine: its own folder on the library path,
    and CUDA devices numbered like nvidia-smi (PCI bus order)."""
    env = dict(os.environ if base is None else base)
    # llama-server reads LLAMA_ARG_* (and LLAMA_API_KEY) from the environment;
    # inherited values must not change how VVL's engine is configured.
    for name in [k for k in env if k.startswith(("LLAMA_ARG_", "LLAMA_API_KEY", "LLAMA_CACHE"))]:
        env.pop(name, None)
    system = platform.system()
    if system == "Linux":
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = str(bin_dir) + (os.pathsep + existing if existing else "")
    elif system == "Windows":
        env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return env


def run_command(argv: list[str], *, env: dict, timeout: float, cwd: str | None = None) -> tuple[int | None, str]:
    """(exit code or None on timeout, combined output)."""
    try:
        proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                              env=env, timeout=timeout, cwd=cwd, check=False)
        return proc.returncode, proc.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else ""
        return None, out
    except OSError as exc:
        return -1, f"{type(exc).__name__}: {exc}"


def _tail(text: str, lines: int = 15) -> list[str]:
    return [line[:400] for line in (text or "").strip().splitlines()[-lines:]]


class EngineInstaller:
    """One instance per app. `models_dir` is a callable (settings are read live)."""

    def __init__(self, jobs: JobManager, lock: Callable[[], EngineLock], models_dir: Callable[[], str],
                 report: Callable[[], dict | None], is_running: Callable[[], bool] = lambda: False,
                 selftest_timeout: float = SELFTEST_TIMEOUT_SECONDS):
        self.jobs = jobs
        self._lock_provider = lock
        self._models_dir = models_dir
        self._report = report
        self._is_running = is_running
        self.selftest_timeout = selftest_timeout
        self._guard = threading.RLock()

    # -- paths ---------------------------------------------------------------

    @property
    def lock(self) -> EngineLock:
        return self._lock_provider()

    def engine_root(self) -> Path:
        return Path(self._models_dir()) / "engine"

    def tag_dir(self) -> Path:
        return self.engine_root() / self.lock.release.tag

    def variant_dir(self, variant_id: str) -> Path:
        return self.tag_dir() / variant_id

    def downloads_dir(self) -> Path:
        return self.tag_dir() / "downloads"

    @staticmethod
    def failures_path() -> Path:
        return paths.home() / "engine" / "selftest_failures.json"

    # -- manifests -------------------------------------------------------------

    def manifest(self, variant_id: str) -> dict | None:
        folder = self.variant_dir(variant_id)
        try:
            data = json.loads((folder / MANIFEST_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("tag") != self.lock.release.tag or data.get("variant") != variant_id:
            return None
        server = folder / str(data.get("server") or "")
        if not server.is_file():
            return None
        data = dict(data)
        version = data.get("version")
        if not (isinstance(version, str) and version.startswith("b")):
            output = "\n".join((data.get("selftest") or {}).get("version_output") or [])
            data["version"] = parse_version(output) or version
        data["server_path"] = str(server)
        data["bin_dir_path"] = str(server.parent)
        return data

    def installed_ids(self) -> set[str]:
        return {v.id for v in self.lock.variants if self.manifest(v.id) is not None}

    # -- self-test failure records --------------------------------------------

    def failures(self) -> dict:
        try:
            data = json.loads(self.failures_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        entries = data.get(self.lock.release.tag) if isinstance(data, dict) else None
        return entries if isinstance(entries, dict) else {}

    def record_failure(self, variant_id: str, message: str) -> None:
        path = self.failures_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            data = {}
        tag = self.lock.release.tag
        entries = data.get(tag) if isinstance(data.get(tag), dict) else {}
        entries[variant_id] = {"message": message, "at": utc_now_iso(),
                               "driver_version": variants_mod.nvidia_driver_version(self._report())}
        data[tag] = entries
        atomic_write_text(path, json.dumps(data, indent=1))

    def clear_failure(self, variant_id: str) -> None:
        path = self.failures_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        entries = data.get(self.lock.release.tag) if isinstance(data, dict) else None
        if isinstance(entries, dict) and entries.pop(variant_id, None) is not None:
            atomic_write_text(path, json.dumps(data, indent=1))

    # -- choosing --------------------------------------------------------------

    def assess(self) -> list[dict]:
        return variants_mod.assess(self.lock, self._report(), installed=self.installed_ids(),
                                   failures=self.failures())

    def recommended(self, assessed: list[dict] | None = None) -> dict | None:
        return variants_mod.recommend(assessed if assessed is not None else self.assess())

    def active_install(self, preferred: str = "auto") -> dict | None:
        """The manifest of the build the supervisor should run.

        A specific `preferred` id is used when it is installed. "auto" takes the
        first installed build in preference order that is compatible here.
        """
        assessed = self.assess()
        if preferred and preferred != "auto":
            manifest = self.manifest(preferred)
            if manifest is not None:
                return manifest
        for entry in assessed:
            if entry["installed"] and entry["compatible"]:
                return self.manifest(entry["id"])
        for entry in assessed:
            if entry["installed"]:
                return self.manifest(entry["id"])
        return None

    # -- install ---------------------------------------------------------------

    def start_install(self, variant_id: str | None = None) -> Job:
        assessed = self.assess()
        if variant_id in (None, "", "auto"):
            rec = self.recommended(assessed)
            if rec is None:
                raise StoreError("CONFLICT", "No llama.cpp build fits this computer.", status=409)
            variant_id = rec["id"]
        variant = self.lock.find(variant_id)
        listed = next((v for v in assessed if v["id"] == variant_id), None)
        if variant is None or listed is None:
            raise StoreError("INVALID_INPUT", f"'{variant_id}' is not a llama.cpp build for this computer.",
                             status=400, details={"variant": variant_id,
                                                  "variants": [v["id"] for v in assessed]})
        with self._guard:
            if self._is_running():
                raise StoreError("CONFLICT", "Stop the engine before installing another build.", status=409)
            active = self.jobs.active_job(JOB_KIND)
            if active is not None:
                raise StoreError("CONFLICT", "An engine install is already running.", status=409,
                                 details={"active_job_id": active.id})
            required = variant.download_size_bytes * DISK_FACTOR + DISK_MARGIN_BYTES
            free = downloads.free_bytes(Path(self._models_dir()))
            if free is not None and free < required:
                raise StoreError("INSUFFICIENT_DISK_SPACE", f"Not enough free disk space for {variant.label}.",
                                 status=507, details={"required_bytes": required, "free_bytes": free})
            files = [{"path": a.name, "size_bytes": a.size_bytes} for a in variant.assets]
            try:
                return self.jobs.submit(JOB_KIND, variant.id, f"llama.cpp {self.lock.release.tag} ({variant.label})",
                                        files, lambda job: self._install(job, variant))
            except JobBusy as busy:
                raise StoreError("CONFLICT", "An engine install is already running.", status=409,
                                 details={"active_job_id": busy.job.id}) from None

    def _install(self, job: Job, variant: Variant) -> dict:
        tag = self.lock.release.tag
        dl_dir = self.downloads_dir()
        dl_dir.mkdir(parents=True, exist_ok=True)
        archives: list[tuple[Path, object]] = []
        for asset in variant.assets:
            job.check_canceled()
            dest = dl_dir / asset.name
            if dest.is_file() and dest.stat().st_size == asset.size_bytes:
                job.set_file_state(asset.name, "verifying")
                if downloads.sha256_of_file(dest, cancel=job.cancel_event) == asset.sha256:
                    job.set_file_state(asset.name, "skipped", bytes_done=asset.size_bytes)
                    archives.append((dest, asset))
                    continue
                dest.unlink()
            job.set_file_state(asset.name, "running")

            def progress(done, net, name=asset.name):
                job.set_file_bytes(name, done)
                if net:
                    job.add_network_bytes(net)

            downloads.download_file(asset.url, dest, size_bytes=asset.size_bytes, sha256=asset.sha256,
                                    cancel=job.cancel_event, progress=progress)
            job.set_file_state(asset.name, "done", bytes_done=asset.size_bytes)
            archives.append((dest, asset))

        job.check_canceled()
        staging = self.tag_dir() / f".staging-{variant.id}-{secrets.token_hex(3)}"
        try:
            server = self._unpack(job, variant, archives, staging)
            job.set_state("verifying")
            job.check_canceled()
            selftest = self._self_test(variant, server)
            final = self.variant_dir(variant.id)
            if final.exists():
                shutil.rmtree(final)
            os.replace(staging, final)
            server_rel = server.relative_to(staging).as_posix()
            manifest = {
                "tag": tag,
                "variant": variant.id,
                "label": variant.label,
                "backend": variant.backend,
                "os": variant.os,
                "arch": variant.arch,
                "server": server_rel,
                "installed_at": utc_now_iso(),
                "version": selftest["version"],
                "devices": selftest["devices"],
                "assets": [{"name": a.name, "sha256": a.sha256, "size_bytes": a.size_bytes} for a in variant.assets],
                "selftest": {"version_output": selftest["version_output"],
                             "devices_output": selftest["devices_output"]},
            }
            atomic_write_text(final / MANIFEST_NAME, json.dumps(manifest, indent=1))
        except JobError as exc:
            shutil.rmtree(staging, ignore_errors=True)
            if exc.code in ("ENGINE_SELF_TEST_FAILED", "EXTRACT_FAILED"):
                # The archives are verified but cannot be used here; free the space.
                for path, _asset in archives:
                    path.unlink(missing_ok=True)
                _remove_if_empty(dl_dir)
            raise
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        self.clear_failure(variant.id)
        for path, _asset in archives:
            path.unlink(missing_ok=True)
        _remove_if_empty(dl_dir)
        logger.info("Installed llama.cpp %s (%s): %s; devices: %s", tag, variant.id, selftest["version"],
                    ", ".join(d["name"] for d in selftest["devices"]) or "none")
        return {"variant": variant.id, "version": selftest["version"], "devices": selftest["devices"]}

    def _unpack(self, job: Job, variant: Variant, archives, staging: Path) -> Path:
        engine_archives = [(p, a) for p, a in archives if a.kind == "engine"]
        runtime_archives = [(p, a) for p, a in archives if a.kind == "runtime"]
        try:
            for path, asset in engine_archives:
                job.check_canceled()
                extract_mod.extract(path, staging)
            server = _find_server(staging)
            if server is None:
                raise JobError("EXTRACT_FAILED", "The engine archive does not contain llama-server.")
            for path, asset in runtime_archives:
                job.check_canceled()
                scratch = staging.parent / f"{staging.name}-runtime"
                try:
                    extract_mod.extract(path, scratch)
                    _flatten_into(scratch, server.parent)
                finally:
                    shutil.rmtree(scratch, ignore_errors=True)
        except extract_mod.UnsafeArchive as exc:
            raise JobError("EXTRACT_FAILED", f"The engine archive was rejected: {exc}") from None
        except (OSError, EOFError, ValueError) as exc:  # tarfile/zipfile errors
            raise JobError("EXTRACT_FAILED", f"The engine archive could not be unpacked ({type(exc).__name__}).") from None
        mode = server.stat().st_mode
        os.chmod(server, mode | 0o755)
        if platform.system() == "Darwin":
            run_command(["xattr", "-dr", "com.apple.quarantine", str(staging)], env=dict(os.environ), timeout=60)
        return server

    def _self_test(self, variant: Variant, server: Path) -> dict:
        env = engine_env(server.parent)
        code, version_out = run_command([str(server), "--version"], env=env, timeout=self.selftest_timeout,
                                        cwd=str(server.parent))
        problem = None
        if code is None:
            problem = "llama-server --version did not finish"
        elif code != 0:
            problem = classify_incompatible(version_out) or f"llama-server --version exited with code {code}"
        devices_out, devices = "", []
        if problem is None:
            code, devices_out = run_command([str(server), "--list-devices"], env=env, timeout=self.selftest_timeout,
                                            cwd=str(server.parent))
            devices = parse_devices(devices_out)
            if code is None:
                problem = "llama-server --list-devices did not finish"
            elif code != 0:
                problem = classify_incompatible(devices_out) or f"llama-server --list-devices exited with code {code}"
            else:
                prefixes = BACKEND_DEVICE_PREFIXES.get(variant.backend, ())
                if prefixes and not any(d["name"].startswith(prefixes) for d in devices):
                    problem = (classify_incompatible(devices_out)
                               or f"it found no {variant.backend.upper()} device on this computer")
        if problem is not None:
            output = (version_out + "\n" + devices_out).strip()
            self.record_failure(variant.id, problem)
            nxt = variants_mod.next_after(self.assess(), variant.id)
            message = f"The {variant.label} build cannot run here: {problem}."
            message += f" Try {nxt['label']} ({nxt['id']}) instead." if nxt else " No other build is available."
            raise JobError("ENGINE_SELF_TEST_FAILED", message,
                           {"variant": variant.id, "next_variant": nxt["id"] if nxt else None,
                            "output_tail": _tail(output)})
        return {"version": parse_version(version_out), "version_output": _tail(version_out, 10),
                "devices": devices, "devices_output": _tail(devices_out, 20)}

    # -- uninstall -------------------------------------------------------------

    def uninstall(self, variant_id: str | None = None) -> list[str]:
        if self._is_running():
            raise StoreError("CONFLICT", "Stop the engine before uninstalling it.", status=409)
        if self.jobs.active_job(JOB_KIND) is not None:
            raise StoreError("CONFLICT", "An engine install is running.", status=409)
        removed = []
        with self._guard:
            if variant_id:
                if self.lock.find(variant_id) is None:
                    raise StoreError("INVALID_INPUT", f"Unknown llama.cpp build '{variant_id}'.", status=400)
                targets = [self.variant_dir(variant_id)]
            else:
                root = self.engine_root()
                targets = [p for p in root.iterdir()] if root.is_dir() else []
            for target in targets:
                if target.exists():
                    shutil.rmtree(target, ignore_errors=False)
                    removed.append(target.name)
            _remove_if_empty(self.tag_dir())
            _remove_if_empty(self.engine_root())
        logger.info("Engine uninstalled (%s).", ", ".join(removed) or "nothing to remove")
        return removed


def _find_server(root: Path) -> Path | None:
    candidates = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name in SERVER_NAMES:
                path = Path(dirpath) / name
                if path.is_file() and not path.is_symlink():
                    candidates.append(path)
    if not candidates:
        return None
    return min(candidates, key=lambda p: len(p.parts))


def _flatten_into(source: Path, target: Path) -> None:
    """Move every file (and symlink) from `source` into `target`, by base name."""
    for dirpath, _dirnames, filenames in os.walk(source):
        for name in filenames:
            src = Path(dirpath) / name
            dest = target / name
            if dest.exists() or dest.is_symlink():
                continue
            if src.is_symlink():
                os.symlink(os.readlink(src), dest)
            else:
                shutil.move(str(src), str(dest))


def _remove_if_empty(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass


__all__ = ["EngineInstaller", "JobCanceled", "classify_incompatible", "engine_env", "parse_devices",
           "parse_version", "run_command"]
