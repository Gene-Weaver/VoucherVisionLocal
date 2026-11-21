"""Background jobs for long operations (model downloads, engine installs, the label detector).

docs/API.md section 9, "Jobs". A job runs in its own thread and reports
progress through a `Job` object; routes return `Job.to_dict()` snapshots.
Jobs live in memory only: they survive UI reloads, and after a server restart
the state that matters (verified files, resumable `.partial` files) is read
back from disk by the model store and the engine installer.

Rules:
  * one job per `kind` runs at a time. A second job of that kind either waits
    in a queue (`submit(..., queue=True)`, used for model downloads: they run
    one after another, since one download already fills the connection) or is
    refused with `JobBusy` (engine installs);
  * a job can be canceled; the runner sees `job.cancel_event` and raises
    `JobCanceled` at the next chunk boundary;
  * a runner reports failures by raising `JobError(code, message)`.
"""

from __future__ import annotations

import datetime
import logging
import secrets
import threading
import time
from collections import deque
from typing import Callable

logger = logging.getLogger(__name__)

JOB_KINDS = ("model_download", "engine_install", "detector_download")
JOB_STATES = ("queued", "running", "verifying", "done", "error", "canceled")
ACTIVE_STATES = ("queued", "running", "verifying")
FILE_STATES = ("pending", "running", "verifying", "done", "skipped", "error")

KEEP_FINISHED_JOBS = 20
SPEED_WINDOW_SECONDS = 8.0


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class JobError(Exception):
    """A runner failure with a code from the contract (NETWORK_ERROR, CHECKSUM_MISMATCH, ...)."""

    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class JobCanceled(Exception):
    """Raised inside a runner when its job was canceled."""


class JobBusy(Exception):
    """Another job of the same kind is active."""

    def __init__(self, job: "Job"):
        super().__init__(f"Job {job.id} ({job.kind}) is already running.")
        self.job = job


class Job:
    def __init__(self, kind: str, target: str, label: str, files: list[dict]):
        if kind not in JOB_KINDS:
            raise ValueError(f"unknown job kind {kind!r}")
        self.id = "j_" + secrets.token_hex(4)
        self.kind = kind
        self.target = target
        self.label = label
        self.state = "queued"
        self.files = [
            {"path": f["path"], "size_bytes": int(f.get("size_bytes") or 0), "bytes_done": 0, "state": "pending"}
            for f in files
        ]
        self.current_file: str | None = None
        self.error: dict | None = None
        self.started_at = utc_now_iso()
        self.finished_at: str | None = None
        self.result: dict | None = None   # runner's return value (not part of the contract)
        self.cancel_event = threading.Event()
        self._lock = threading.RLock()
        self._samples: deque[tuple[float, int]] = deque()
        self._network_bytes = 0
        self.queue_position: int | None = None   # 1 = next to run; None when not waiting

    # -- progress (called from the runner thread) --------------------------

    def _file(self, path: str) -> dict:
        for entry in self.files:
            if entry["path"] == path:
                return entry
        raise KeyError(path)

    def set_state(self, state: str) -> None:
        with self._lock:
            self.state = state

    def set_file_state(self, path: str, state: str, *, bytes_done: int | None = None) -> None:
        with self._lock:
            entry = self._file(path)
            entry["state"] = state
            if bytes_done is not None:
                entry["bytes_done"] = int(bytes_done)
            if state in ("running", "verifying"):
                self.current_file = path

    def set_file_bytes(self, path: str, bytes_done: int) -> None:
        with self._lock:
            self._file(path)["bytes_done"] = int(bytes_done)

    def add_network_bytes(self, count: int) -> None:
        """Bytes received from the network (for speed); file progress is set separately."""
        now = time.monotonic()
        with self._lock:
            self._network_bytes += count
            self._samples.append((now, self._network_bytes))
            while self._samples and now - self._samples[0][0] > SPEED_WINDOW_SECONDS:
                self._samples.popleft()

    def check_canceled(self) -> None:
        if self.cancel_event.is_set():
            raise JobCanceled()

    # -- reading ----------------------------------------------------------

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_STATES

    def _speed(self) -> int | None:
        if self.state != "running" or len(self._samples) < 2:
            return None
        (t0, b0), (t1, b1) = self._samples[0], self._samples[-1]
        # Measure up to now, so a stalled download decays toward zero.
        elapsed = max(time.monotonic(), t1) - t0
        if elapsed <= 0.2:
            return None
        return int((b1 - b0) / elapsed)

    def to_dict(self) -> dict:
        with self._lock:
            bytes_total = sum(f["size_bytes"] for f in self.files)
            bytes_done = sum(min(f["bytes_done"], f["size_bytes"]) if f["size_bytes"] else f["bytes_done"]
                             for f in self.files)
            speed = self._speed()
            eta = None
            if speed and speed > 0 and bytes_total > bytes_done:
                eta = int((bytes_total - bytes_done) / speed)
            return {
                "id": self.id,
                "kind": self.kind,
                "target": self.target,
                "label": self.label,
                "state": self.state,
                "bytes_done": bytes_done,
                "bytes_total": bytes_total,
                "speed_bps": speed,
                "eta_seconds": eta,
                "current_file": self.current_file if self.active else None,
                "files": [dict(f) for f in self.files],
                "queue_position": self.queue_position if self.state == "queued" else None,
                "error": dict(self.error) if self.error else None,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }


Runner = Callable[[Job], "dict | None"]


class JobManager:
    """Owns the job threads. One instance per app."""

    def __init__(self, keep_finished: int = KEEP_FINISHED_JOBS):
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._keep_finished = keep_finished
        self._threads: dict[str, threading.Thread] = {}
        self._running: dict[str, Job] = {}                   # kind -> the job whose thread runs
        self._queues: dict[str, deque] = {}                  # kind -> waiting (job, runner, on_finish)

    def active_job(self, kind: str) -> Job | None:
        """The running job of `kind`, else the next queued one, else None."""
        with self._lock:
            running = self._running.get(kind)
            if running is not None and running.active:
                return running
            for job_id in self._order:
                job = self._jobs[job_id]
                if job.kind == kind and job.active:
                    return job
        return None

    def active_jobs(self, kind: str) -> list[Job]:
        """Every running or queued job of `kind`, running first, then in queue order."""
        with self._lock:
            out = []
            running = self._running.get(kind)
            if running is not None and running.active:
                out.append(running)
            out.extend(job for job, _r, _f in self._queues.get(kind, ()) if job.active)
            return out

    def latest_for(self, kind: str, target: str) -> Job | None:
        with self._lock:
            for job_id in reversed(self._order):
                job = self._jobs[job_id]
                if job.kind == kind and job.target == target:
                    return job
        return None

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return [self._jobs[i] for i in self._order]

    def submit(self, kind: str, target: str, label: str, files: list[dict], runner: Runner,
               *, on_finish: Callable[[Job], None] | None = None, queue: bool = False) -> Job:
        """Run `runner(job)` in a thread.

        When a job of this kind is already running, the new job waits in line
        (`queue=True`, state "queued") or JobBusy is raised (`queue=False`).
        """
        with self._lock:
            busy = self.active_job(kind)
            if busy is not None and not queue:
                raise JobBusy(busy)
            job = Job(kind, target, label, files)
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._prune()
            if busy is not None:
                self._queues.setdefault(kind, deque()).append((job, runner, on_finish))
                self._renumber(kind)
                return job
            thread = self._start_locked(job, runner, on_finish)
        thread.start()
        return job

    def _start_locked(self, job: Job, runner: Runner, on_finish) -> threading.Thread:
        self._running[job.kind] = job
        job.queue_position = None
        thread = threading.Thread(target=self._run, args=(job, runner, on_finish),
                                  name=f"vvl-job-{job.id}", daemon=True)
        self._threads[job.id] = thread
        return thread

    def _renumber(self, kind: str) -> None:
        for position, (job, _r, _f) in enumerate(self._queues.get(kind, ()), start=1):
            job.queue_position = position

    def _start_next(self, kind: str) -> None:
        """After a job of `kind` ends, start the next one waiting in line."""
        with self._lock:
            if self._running.get(kind) is not None and self._running[kind].active:
                return
            self._running.pop(kind, None)
            waiting = self._queues.get(kind)
            thread = None
            while waiting:
                job, runner, on_finish = waiting.popleft()
                if job.state == "queued" and not job.cancel_event.is_set():
                    thread = self._start_locked(job, runner, on_finish)
                    break
            self._renumber(kind)
        if thread is not None:
            thread.start()

    def _run(self, job: Job, runner: Runner, on_finish) -> None:
        job.set_state("running")
        try:
            job.result = runner(job) or None
            job.check_canceled()
            with job._lock:
                job.state = "done"
        except JobCanceled:
            with job._lock:
                job.state = "canceled"
                for entry in job.files:
                    if entry["state"] in ("running", "verifying"):
                        entry["state"] = "pending"
            logger.info("Job %s (%s %s) canceled.", job.id, job.kind, job.target)
        except JobError as exc:
            with job._lock:
                job.state = "error"
                job.error = {"code": exc.code, "message": exc.message}
                if exc.details:
                    job.error["details"] = exc.details
                for entry in job.files:
                    if entry["state"] in ("running", "verifying"):
                        entry["state"] = "error"
            logger.warning("Job %s (%s %s) failed: %s %s", job.id, job.kind, job.target, exc.code, exc.message)
        except Exception as exc:  # noqa: BLE001 - a job must always finish
            with job._lock:
                job.state = "error"
                job.error = {"code": "INTERNAL_ERROR", "message": f"Unexpected error ({type(exc).__name__})."}
                for entry in job.files:
                    if entry["state"] in ("running", "verifying"):
                        entry["state"] = "error"
            logger.exception("Job %s (%s %s) crashed.", job.id, job.kind, job.target)
        finally:
            with job._lock:
                job.finished_at = utc_now_iso()
            if on_finish is not None:
                try:
                    on_finish(job)
                except Exception:  # noqa: BLE001
                    logger.exception("Job %s finish hook failed.", job.id)
            self._start_next(job.kind)

    def cancel(self, job: Job) -> Job:
        """Stop a running job, or take a queued one out of the line."""
        with self._lock:
            waiting = self._queues.get(job.kind)
            if job.state == "queued" and waiting and any(item[0] is job for item in waiting):
                self._queues[job.kind] = deque(item for item in waiting if item[0] is not job)
                self._renumber(job.kind)
                job.cancel_event.set()
                with job._lock:
                    job.state = "canceled"
                    job.queue_position = None
                    job.finished_at = utc_now_iso()
                logger.info("Job %s (%s %s) removed from the queue.", job.id, job.kind, job.target)
                return job
        if job.active:
            job.cancel_event.set()
        return job

    def wait(self, job: Job, timeout: float | None = None) -> bool:
        thread = self._threads.get(job.id)
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def cancel_all(self, timeout: float = 5.0) -> None:
        with self._lock:
            queued = [item[0] for waiting in self._queues.values() for item in waiting]
        for job in queued:
            self.cancel(job)
        for job in self.list():
            if job.active:
                job.cancel_event.set()
        deadline = time.monotonic() + timeout
        for job in self.list():
            self.wait(job, max(0.0, deadline - time.monotonic()))

    def _prune(self) -> None:
        finished = [i for i in self._order if not self._jobs[i].active]
        excess = len(finished) - self._keep_finished
        for job_id in finished[:max(0, excess)]:
            self._order.remove(job_id)
            self._jobs.pop(job_id, None)
            self._threads.pop(job_id, None)
