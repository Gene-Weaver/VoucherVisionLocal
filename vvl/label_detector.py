"""The TextCollage label detector's files and their first-start download.

The detector is pinned in `vvl/registry/archival_detector.lock.yaml`. At first
start `vvl serve` downloads the files for this computer's format (job kind
`detector_download`) into `<VVL_HOME>/detector/<repo>@<revision>/`. Every file
is checked against its sha256, then `verified.json` records the verified set.
Until the detector is installed, the pipeline sends the whole image instead of
a label collage.

Formats: `coreml_fp32` on macOS (Core ML on the Apple GPU), `onnx` elsewhere
(onnxruntime on the CPU). `VVL_DETECTOR_FORMAT=onnx|coreml_fp32` overrides
the choice (testing).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Callable

import yaml

from vvl import downloads, paths
from vvl.jobs import Job, JobError, JobManager

logger = logging.getLogger(__name__)

JOB_KIND = "detector_download"
JOB_LABEL = "Label detector"
FORMATS = ("coreml_fp32", "onnx")
RUNTIME_LABELS = {"coreml_fp32": "Core ML, Apple GPU", "onnx": "ONNX Runtime, CPU"}
MARKER = "verified.json"
DISK_MARGIN_BYTES = 500 * 1024 * 1024
DEFAULT_HF_BASE_URL = "https://huggingface.co"


class DetectorNotInstalled(Exception):
    """The detector's files for this format are not downloaded (or not verified) yet."""


def lock_path() -> Path:
    return paths.registry_dir() / "archival_detector.lock.yaml"


def load_lock(path: str | os.PathLike | None = None) -> dict:
    with open(path or lock_path(), encoding="utf-8") as handle:
        lock = yaml.safe_load(handle)
    for key in ("repo", "revision", "imgsz", "classes", "formats"):
        if key not in lock:
            raise ValueError(f"archival_detector.lock.yaml has no {key!r}")
    return lock


def default_format() -> str:
    override = os.environ.get("VVL_DETECTOR_FORMAT", "").strip()
    if override in FORMATS:
        return override
    return "coreml_fp32" if sys.platform == "darwin" else "onnx"


def install_dir(lock: dict) -> Path:
    name = lock["repo"].rsplit("/", 1)[-1]
    return paths.home() / "detector" / f"{name}@{lock['revision'][:12]}"


def files_for(lock: dict, fmt: str) -> list[dict]:
    try:
        return list(lock["formats"][fmt]["files"])
    except KeyError:
        raise ValueError(f"the detector lock has no {fmt!r} format") from None


def size_bytes(lock: dict, fmt: str) -> int:
    return sum(int(f["size_bytes"]) for f in files_for(lock, fmt))


def file_url(lock: dict, entry: dict, base_url: str | None = None) -> str:
    base = (base_url or DEFAULT_HF_BASE_URL).rstrip("/")
    return f"{base}/{lock['repo']}/resolve/{lock['revision']}/{entry['path']}"


def _read_marker(root: Path) -> dict:
    try:
        data = json.loads((root / MARKER).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def is_installed(lock: dict, fmt: str) -> bool:
    """Every file of `fmt` was verified by sha256 and is still on disk at its size."""
    root = install_dir(lock)
    verified = _read_marker(root).get(fmt) or {}
    for entry in files_for(lock, fmt):
        if verified.get(entry["path"]) != entry["sha256"]:
            return False
        try:
            if (root / entry["path"]).stat().st_size != int(entry["size_bytes"]):
                return False
        except OSError:
            return False
    return True


def _record_verified(lock: dict, fmt: str) -> None:
    root = install_dir(lock)
    marker = _read_marker(root)
    marker[fmt] = {entry["path"]: entry["sha256"] for entry in files_for(lock, fmt)}
    tmp = root / f"{MARKER}.tmp"
    tmp.write_text(json.dumps(marker, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, root / MARKER)


def load_detector(lock: dict | None = None, fmt: str | None = None):
    """The installed detector, loaded. Raises DetectorNotInstalled when it is not downloaded yet."""
    lock = lock or load_lock()
    fmt = fmt or default_format()
    if not is_installed(lock, fmt):
        raise DetectorNotInstalled(f"the label detector ({fmt}) is not downloaded yet")
    from textcollage.detector import ArchivalDetector

    return ArchivalDetector(install_dir(lock), fmt, classes=lock["classes"], imgsz=int(lock["imgsz"]),
                            conf=float(lock.get("conf", 0.25)), max_det=int(lock.get("max_det", 100)),
                            cache_dir=paths.cache_dir() / "coreml")


class DetectorStore:
    """Install state and the download job, for the API, the Setup page and `vvl serve`."""

    def __init__(self, jobs: JobManager, *, lock: Callable[[], dict] = load_lock,
                 hf_base_url: str | None = None, fmt: Callable[[], str] = default_format,
                 on_installed: Callable[[], None] | None = None):
        self.jobs = jobs
        self._lock_loader = lock
        self.hf_base_url = hf_base_url
        self._fmt = fmt
        self.on_installed = on_installed
        self._submit_lock = threading.Lock()

    def lock(self) -> dict:
        return self._lock_loader()

    def format(self) -> str:
        return self._fmt()

    def is_installed(self) -> bool:
        return is_installed(self.lock(), self.format())

    def state(self) -> dict:
        lock, fmt = self.lock(), self.format()
        job = self.jobs.latest_for(JOB_KIND, fmt)
        installed = is_installed(lock, fmt)
        if installed:
            state = "installed"
        elif job is not None and job.active:
            state = "downloading"
        elif job is not None and job.state == "error":
            state = "error"
        else:
            state = "not_installed"
        return {
            "state": state,
            "format": fmt,
            "runtime": RUNTIME_LABELS[fmt],
            "repo": lock["repo"],
            "revision": lock["revision"],
            "size_bytes": size_bytes(lock, fmt),
            "path": str(install_dir(lock)),
            "job": job.to_dict() if job is not None else None,
            "error": (job.error if job is not None and job.state == "error" and not installed else None),
        }

    def install(self) -> Job:
        """Start the download (or return the one already running). Files already verified are skipped."""
        with self._submit_lock:
            lock, fmt = self.lock(), self.format()
            active = self.jobs.active_job(JOB_KIND)
            if active is not None:
                return active
            files = [{"path": f["path"], "size_bytes": int(f["size_bytes"])} for f in files_for(lock, fmt)]
            return self.jobs.submit(JOB_KIND, fmt, JOB_LABEL, files, self._runner(lock, fmt),
                                    on_finish=self._finished)

    def _finished(self, job: Job) -> None:
        if job.state == "done" and self.on_installed is not None:
            try:
                self.on_installed()
            except Exception:  # noqa: BLE001
                logger.exception("Loading the label detector after its download failed.")

    def _runner(self, lock: dict, fmt: str):
        base_url = self.hf_base_url

        def run(job: Job):
            root = install_dir(lock)
            root.mkdir(parents=True, exist_ok=True)
            entries = files_for(lock, fmt)
            remaining = sum(int(e["size_bytes"]) for e in entries if not (root / e["path"]).is_file())
            free = downloads.free_bytes(root)
            if free is not None and free < remaining + DISK_MARGIN_BYTES:
                raise JobError("INSUFFICIENT_DISK_SPACE", "Not enough free disk space for the label detector.",
                               {"required_bytes": remaining + DISK_MARGIN_BYTES, "free_bytes": free})
            for entry in entries:
                job.check_canceled()
                path, size, sha = entry["path"], int(entry["size_bytes"]), entry["sha256"]
                dest = root / path
                if dest.is_file() and dest.stat().st_size == size:
                    job.set_file_state(path, "verifying", bytes_done=0)
                    digest = downloads.sha256_of_file(dest, cancel=job.cancel_event,
                                                      progress=lambda n, p=path: job.set_file_bytes(p, n))
                    if digest == sha:
                        job.set_file_state(path, "skipped", bytes_done=size)
                        continue
                    logger.warning("%s does not match its checksum; downloading it again.", path)
                    dest.unlink()
                elif dest.is_file():
                    dest.unlink()
                job.set_file_state(path, "running", bytes_done=0)

                def progress(done, net, p=path):
                    job.set_file_bytes(p, done)
                    if net:
                        job.add_network_bytes(net)

                downloads.download_file(file_url(lock, entry, base_url), dest, size_bytes=size, sha256=sha,
                                        cancel=job.cancel_event, progress=progress)
                job.set_file_state(path, "done", bytes_done=size)
            job.set_state("verifying")
            _record_verified(lock, fmt)
            if not is_installed(lock, fmt):
                raise JobError("CHECKSUM_MISMATCH", "The label detector could not be verified after its download.")
            logger.info("Label detector (%s) downloaded and verified.", fmt)
            return {"format": fmt, "path": str(root)}

        return run
