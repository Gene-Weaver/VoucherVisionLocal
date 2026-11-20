"""Local model files on disk: install state, downloads, deletion.

Layout (docs/API.md section 9, "Storage"):

    <models_dir>/models/<hf org>/<hf repo>/<file>           verified model files
    <models_dir>/models/<hf org>/<hf repo>/<file>.partial   a resumable download
    <models_dir>/models/.vvl-verified.json                  which files were verified

Two builds that list the same repo + file (for example one vision projector)
share it on disk. A file counts as installed only when the verified index
records the registry's sha256 for it and its size and modification time are
unchanged since; nothing is used before that. Re-hashing a 35 GB file on every
`GET /v1/models` would be too slow, so the index is the record of the check.

Install state is derived from disk plus the in-memory jobs, so a server
restart shows the right state: verified files stay `installed`, interrupted
downloads show `partial` and resume where they stopped.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Callable

from vvl import downloads
from vvl.fileutil import atomic_write_text
from vvl.jobs import Job, JobCanceled, JobError, JobManager, utc_now_iso

logger = logging.getLogger(__name__)

HF_BASE_URL = "https://huggingface.co"
VERIFIED_INDEX_NAME = ".vvl-verified.json"
DISK_MARGIN_BYTES = 2 * 1024 ** 3
JOB_KIND = "model_download"

GATED_MESSAGE = (
    "Hugging Face refused the download (HTTP {status}). This model may be gated: open its page on "
    "huggingface.co, accept the license, and save a Hugging Face access token on the Keys tab."
)

_index_lock = threading.RLock()


class StoreError(Exception):
    """A request the store refuses. Rendered with the error envelope by the routes."""

    def __init__(self, code: str, message: str, *, status: int, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = details or {}


# ------------------------------------------------------------------ paths --

def models_root(models_dir: str | os.PathLike) -> Path:
    return Path(models_dir) / "models"


def repo_dir(models_dir, build) -> Path:
    org, _, repo = build.hf_repo.partition("/")
    return models_root(models_dir) / org / repo


def file_path(models_dir, build, model_file) -> Path:
    return repo_dir(models_dir, build) / model_file.path


def file_url(build, model_file, base_url: str = HF_BASE_URL) -> str:
    return f"{base_url.rstrip('/')}/{build.hf_repo}/resolve/{build.hf_revision}/{model_file.path}"


def _rel(models_dir, path: Path) -> str:
    return path.relative_to(models_root(models_dir)).as_posix()


# --------------------------------------------------------- verified index --

def _index_path(models_dir) -> Path:
    return models_root(models_dir) / VERIFIED_INDEX_NAME


def load_index(models_dir) -> dict:
    with _index_lock:
        try:
            data = json.loads(_index_path(models_dir).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        files = data.get("files") if isinstance(data, dict) else None
        return files if isinstance(files, dict) else {}


def _save_index(models_dir, files: dict) -> None:
    path = _index_path(models_dir)
    atomic_write_text(path, json.dumps({"version": 1, "files": files}, indent=1, sort_keys=True))


def record_verified(models_dir, path: Path, sha256: str) -> None:
    st = path.stat()
    with _index_lock:
        files = load_index(models_dir)
        files[_rel(models_dir, path)] = {
            "sha256": sha256, "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "verified_at": utc_now_iso(),
        }
        _save_index(models_dir, files)


def forget_verified(models_dir, paths_to_forget: list[Path]) -> None:
    with _index_lock:
        files = load_index(models_dir)
        changed = False
        for path in paths_to_forget:
            if files.pop(_rel(models_dir, path), None) is not None:
                changed = True
        if changed:
            _save_index(models_dir, files)


def file_state(models_dir, build, model_file, index: dict | None = None) -> tuple[str, int]:
    """(state, bytes on disk) for one file.

    state: verified | unverified (complete size, not yet checked) | partial |
           missing | wrong_size
    """
    index = load_index(models_dir) if index is None else index
    path = file_path(models_dir, build, model_file)
    try:
        st = path.stat()
    except OSError:
        st = None
    if st is not None:
        if st.st_size != model_file.size_bytes:
            return "wrong_size", st.st_size
        entry = index.get(_rel(models_dir, path))
        if (isinstance(entry, dict) and entry.get("sha256") == model_file.sha256
                and entry.get("size_bytes") == st.st_size and entry.get("mtime_ns") == st.st_mtime_ns):
            return "verified", st.st_size
        return "unverified", st.st_size
    try:
        return "partial", downloads.partial_path(path).stat().st_size
    except OSError:
        return "missing", 0


def disk_status(models_dir, build) -> dict:
    """Install state from disk alone: installed | partial | not_installed."""
    index = load_index(models_dir)
    states = [file_state(models_dir, build, f, index) for f in build.files]
    installed_bytes = sum(min(n, f.size_bytes) for (_s, n), f in zip(states, build.files))
    if all(s == "verified" for s, _n in states):
        state = "installed"
    elif any(s in ("partial", "unverified", "wrong_size") for s, _n in states):
        state = "partial"
    else:
        # Nothing of this build's own, though a shared file may be present.
        state = "not_installed"
    return {"state": state, "installed_bytes": installed_bytes,
            "files": [{"path": f.path, "state": s, "bytes": n} for (s, n), f in zip(states, build.files)]}


def is_installed(models_dir, build) -> bool:
    return disk_status(models_dir, build)["state"] == "installed"


# ------------------------------------------------------------------ store --

class LocalModelStore:
    """Downloads and deletes local builds. One instance per app.

    `models_dir` and `hf_token` are callables so settings and keys are read at
    the moment they are needed.
    """

    def __init__(self, jobs: JobManager, registry: Callable, models_dir: Callable[[], str],
                 hf_token: Callable[[], str | None] = lambda: None, hf_base_url: str | None = None,
                 loaded_model: Callable[[], str | None] = lambda: None):
        self.jobs = jobs
        self._registry = registry
        self._models_dir = models_dir
        self._hf_token = hf_token
        self.hf_base_url = (hf_base_url or os.environ.get("VVL_HF_BASE_URL") or HF_BASE_URL).rstrip("/")
        self._loaded_model = loaded_model
        self._lock = threading.RLock()

    @property
    def models_dir(self) -> str:
        return self._models_dir()

    def _find(self, build_id: str):
        found = self._registry().find_build(build_id)
        if not found:
            raise StoreError("NOT_FOUND", f"Unknown local model '{build_id}'.", status=404,
                             details={"build_id": build_id})
        return found

    # -- state -------------------------------------------------------------

    def status(self, build) -> dict:
        """install_state / installed_bytes / download_job_id for /v1/models."""
        disk = disk_status(self.models_dir, build)
        job = self.jobs.latest_for(JOB_KIND, build.id)
        state = disk["state"]
        job_id = None
        if job is not None and job.state == "queued":
            state, job_id = "queued", job.id
        elif job is not None and job.active:
            state, job_id = "downloading", job.id
        elif job is not None and job.state == "error" and state != "installed":
            state, job_id = "error", job.id
        return {"install_state": state, "installed_bytes": disk["installed_bytes"], "download_job_id": job_id}

    def installed_builds(self) -> list:
        return [b for _f, b in self._registry().builds() if is_installed(self.models_dir, b)]

    def remaining_bytes(self, build) -> int:
        index = load_index(self.models_dir)
        remaining = 0
        for f in build.files:
            state, have = file_state(self.models_dir, build, f, index)
            if state in ("verified", "unverified"):
                continue
            remaining += f.size_bytes - (have if state == "partial" else 0)
        return remaining

    # -- download ----------------------------------------------------------

    def start_download(self, build_id: str, confirm_size_bytes) -> Job:
        _family, build = self._find(build_id)
        total = build.total_size_bytes
        if isinstance(confirm_size_bytes, bool) or not isinstance(confirm_size_bytes, int):
            raise StoreError("INVALID_INPUT", "Send confirm_size_bytes: the build's total_size_bytes.", status=400,
                             details={"total_size_bytes": total})
        if confirm_size_bytes != total:
            raise StoreError("INVALID_INPUT", "confirm_size_bytes does not match this model's size.", status=400,
                             details={"total_size_bytes": total, "confirm_size_bytes": confirm_size_bytes})
        with self._lock:
            if is_installed(self.models_dir, build):
                raise StoreError("CONFLICT", f"{build.label} is already installed.", status=409,
                                 details={"build_id": build.id})
            pending = self.jobs.active_jobs(JOB_KIND)
            same = next((job for job in pending if job.target == build.id), None)
            if same is not None:
                raise StoreError(
                    "CONFLICT",
                    f"{build.label} is already downloading." if same.state != "queued"
                    else f"{build.label} is already waiting to download.",
                    status=409, details={"active_job_id": same.id})
            # Room for this model and for every download ahead of it in line.
            ahead = 0
            for job in pending:
                found = self._registry().find_build(job.target)
                if found:
                    ahead += self.remaining_bytes(found[1])
            required = self.remaining_bytes(build) + ahead + DISK_MARGIN_BYTES
            free = downloads.free_bytes(Path(self.models_dir))
            if free is not None and free < required:
                raise StoreError(
                    "INSUFFICIENT_DISK_SPACE",
                    f"Not enough free disk space for {build.label}"
                    + (" and the downloads ahead of it." if ahead else "."), status=507,
                    details={"required_bytes": required, "free_bytes": free, "queued_bytes": ahead})
            files = [{"path": f.path, "size_bytes": f.size_bytes} for f in build.files]
            # Downloads run one after another: a second one waits in line.
            return self.jobs.submit(JOB_KIND, build.id, build.label, files, self._runner(build), queue=True)

    def _runner(self, build):
        base_url = self.hf_base_url

        def run(job: Job):
            # Read when the download starts, not when it was queued: settings
            # or the Hugging Face token may have changed while it waited.
            models_dir = self.models_dir
            token = self._hf_token()
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            for f in build.files:
                job.check_canceled()
                dest = file_path(models_dir, build, f)
                state, have = file_state(models_dir, build, f)
                if state == "verified":
                    job.set_file_state(f.path, "skipped", bytes_done=f.size_bytes)
                    continue
                if state == "unverified":
                    # A complete file that was never checked (copied in, or the
                    # server stopped right after the rename): hash it first.
                    job.set_file_state(f.path, "verifying", bytes_done=0)
                    digest = downloads.sha256_of_file(
                        dest, cancel=job.cancel_event, progress=lambda n, p=f.path: job.set_file_bytes(p, n))
                    if digest == f.sha256:
                        record_verified(models_dir, dest, digest)
                        job.set_file_state(f.path, "skipped", bytes_done=f.size_bytes)
                        continue
                    logger.warning("%s on disk does not match its checksum; downloading it again.", f.path)
                    dest.unlink()
                if state == "wrong_size":
                    logger.warning("%s on disk has the wrong size; downloading it again.", f.path)
                    dest.unlink()
                job.set_file_state(f.path, "running", bytes_done=have if state == "partial" else 0)

                def progress(done, net, p=f.path):
                    job.set_file_bytes(p, done)
                    if net:
                        job.add_network_bytes(net)

                digest = downloads.download_file(
                    file_url(build, f, base_url), dest, size_bytes=f.size_bytes, sha256=f.sha256,
                    headers=headers, cancel=job.cancel_event, progress=progress,
                    auth_error_code="GATED_MODEL",
                    auth_error_message=None,
                )
                job.set_file_state(f.path, "verifying")
                record_verified(models_dir, dest, digest)
                job.set_file_state(f.path, "done", bytes_done=f.size_bytes)
            job.set_state("verifying")
            if not is_installed(models_dir, build):
                raise JobError("CHECKSUM_MISMATCH", f"{build.label} could not be verified after download.")
            logger.info("%s downloaded and verified.", build.label)
            return {"build_id": build.id}

        def guarded(job: Job):
            try:
                return run(job)
            except JobError as exc:
                if exc.code == "GATED_MODEL":
                    status = exc.details.get("http_status")
                    raise JobError("GATED_MODEL", GATED_MESSAGE.format(status=status), exc.details) from None
                raise
            except JobCanceled:
                raise

        return guarded

    def cancel_download(self, build_id: str) -> Job:
        _family, build = self._find(build_id)
        job = self.jobs.latest_for(JOB_KIND, build.id)
        if job is None:
            raise StoreError("NOT_FOUND", f"No download of {build.label} was started.", status=404,
                             details={"build_id": build.id})
        return self.jobs.cancel(job)

    # -- delete ------------------------------------------------------------

    def delete(self, build_id: str) -> int:
        """Remove the build's files and partials that no other installed build shares."""
        _family, build = self._find(build_id)
        loaded = self._loaded_model()
        loaded = loaded if isinstance(loaded, (list, tuple, set)) else [loaded]
        if f"local:{build.id}" in loaded:
            raise StoreError("CONFLICT", f"{build.label} is loaded in the engine. Stop the engine first.",
                             status=409, details={"build_id": build.id})
        with self._lock:
            pending = self.jobs.active_jobs(JOB_KIND)
            mine = next((job for job in pending if job.target == build.id), None)
            if mine is not None:
                raise StoreError("CONFLICT", f"{build.label} is downloading or waiting to download. "
                                 "Cancel the download first.", status=409, details={"active_job_id": mine.id})
            busy_ids = {job.target for job in pending}
            models_dir = self.models_dir
            keep: set[Path] = set()
            for _f, other in self._registry().builds():
                if other.id == build.id:
                    continue
                busy = other.id in busy_ids
                if busy or is_installed(models_dir, other):
                    keep.update(file_path(models_dir, other, f) for f in other.files)
            freed = 0
            removed: list[Path] = []
            for f in build.files:
                path = file_path(models_dir, build, f)
                if path in keep:
                    continue
                for candidate in (path, downloads.partial_path(path)):
                    try:
                        size = candidate.stat().st_size
                        candidate.unlink()
                        freed += size
                    except FileNotFoundError:
                        continue
                removed.append(path)
            forget_verified(models_dir, removed)
            for directory in {p.parent for p in removed}:
                _remove_empty_dirs(directory, models_root(models_dir))
        logger.info("Deleted %s (%d bytes freed).", build.label, freed)
        return freed


def _remove_empty_dirs(directory: Path, stop: Path) -> None:
    try:
        directory = directory.resolve()
        stop = stop.resolve()
    except OSError:
        return
    while directory != stop and stop in directory.parents:
        try:
            directory.rmdir()
        except OSError:
            return
        directory = directory.parent
