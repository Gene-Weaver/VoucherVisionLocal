"""Resumable, verified file downloads (model files and engine archives).

`download_file()` writes to `<dest>.partial`, resumes with an HTTP Range
request when a partial exists (re-hashing the bytes already on disk first, so
the final sha256 covers the whole file), streams the sha256 while writing,
checks the size, and renames the partial to `<dest>` only after the digest
matches. A canceled or interrupted download keeps its partial.

Credentials: `headers` may carry an Authorization header. It is sent only to
the URL's own host; `requests` drops it when a redirect leads to another host
(the Hugging Face CDN). Header values never appear in log lines or error
messages, and request exceptions are reduced to their type name.
"""

from __future__ import annotations

import errno
import hashlib
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Callable

import requests

from vvl.jobs import JobCanceled, JobError

logger = logging.getLogger(__name__)

CHUNK_BYTES = 1024 * 1024        # at most this much is re-fetched after a dropped connection
HASH_READ_BYTES = 16 * 1024 * 1024
CONNECT_TIMEOUT_SECONDS = 20.0
READ_TIMEOUT_SECONDS = 90.0
MAX_ATTEMPTS_WITHOUT_PROGRESS = 6
RETRY_BACKOFF_SECONDS = (1.0, 2.0, 5.0, 10.0, 20.0, 30.0)
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}
PARTIAL_SUFFIX = ".partial"
USER_AGENT = "VoucherVisionLocal-downloader"

Progress = Callable[[int, int], None]   # (bytes on disk for this file, new network bytes)


def partial_path(dest: Path) -> Path:
    return dest.with_name(dest.name + PARTIAL_SUFFIX)


def free_bytes(path: Path) -> int | None:
    """Free space on the drive that holds `path` (or its nearest existing parent)."""
    probe = Path(path)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return None


def sha256_of_file(path: Path, *, cancel=None, progress: Callable[[int], None] | None = None) -> str:
    """Hex sha256 of a file, checking `cancel` (an Event) between reads."""
    digest = hashlib.sha256()
    done = 0
    with open(path, "rb") as handle:
        while True:
            if cancel is not None and cancel.is_set():
                raise JobCanceled()
            block = handle.read(HASH_READ_BYTES)
            if not block:
                break
            digest.update(block)
            done += len(block)
            if progress is not None:
                progress(done)
    return digest.hexdigest()


def _hash_prefix(path: Path, length: int, cancel, progress) -> "hashlib._Hash":
    digest = hashlib.sha256()
    done = 0
    with open(path, "rb") as handle:
        while done < length:
            if cancel is not None and cancel.is_set():
                raise JobCanceled()
            block = handle.read(min(HASH_READ_BYTES, length - done))
            if not block:
                break
            digest.update(block)
            done += len(block)
            if progress is not None:
                progress(done, 0)
    if done != length:
        raise OSError("partial file shrank while it was being read")
    return digest


def _content_range(value: str | None) -> tuple[int | None, int | None]:
    """(start, total) from 'bytes 100-199/1000'."""
    if not value or not value.startswith("bytes "):
        return None, None
    try:
        span, _, total = value[6:].partition("/")
        start = int(span.split("-", 1)[0])
        return start, (int(total) if total.strip() not in ("", "*") else None)
    except ValueError:
        return None, None


def _display_name(url: str) -> str:
    return url.rsplit("/", 1)[-1] or "file"


class _FileLock:
    """An exclusive lock next to a download (`<file>.partial.lock`), held while it runs.

    Two VoucherVisionLocal processes that share a models folder must never
    write the same file at once; the second one is refused with a clear
    message. The lock belongs to the process, so it disappears if it crashes.
    """

    def __init__(self, path: Path, name: str):
        self.path = Path(str(path) + ".lock")
        self.name = name
        self.handle = None

    def __enter__(self):
        self.handle = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            self.handle = None
            raise JobError("DOWNLOAD_FAILED",
                           f"{self.name} is already being downloaded by another VoucherVisionLocal window. "
                           "Let that download finish, or cancel it there first.") from None
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            try:
                if os.name == "nt":
                    import msvcrt

                    self.handle.seek(0)
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            self.handle.close()
            self.handle = None
            try:
                self.path.unlink()
            except OSError:
                pass
        return False


def download_file(
    url: str,
    dest: Path,
    *,
    size_bytes: int,
    sha256: str,
    headers: dict | None = None,
    cancel=None,
    progress: Progress | None = None,
    auth_error_code: str = "HTTP_ERROR",
    auth_error_message: str | None = None,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Download `url` to `dest` and verify it. Returns the hex sha256.

    Raises JobError (NETWORK_ERROR, HTTP_ERROR, CHECKSUM_MISMATCH, SIZE_MISMATCH,
    INSUFFICIENT_DISK_SPACE, or `auth_error_code` for 401/403) or JobCanceled.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    name = _display_name(url)
    with _FileLock(partial_path(dest), name):
        return _download_file_locked(url, dest, size_bytes=size_bytes, sha256=sha256, headers=headers,
                                     cancel=cancel, progress=progress, auth_error_code=auth_error_code,
                                     auth_error_message=auth_error_message, session=session, sleep=sleep)


def _download_file_locked(url, dest, *, size_bytes, sha256, headers, cancel, progress, auth_error_code,
                          auth_error_message, session, sleep) -> str:
    part = partial_path(dest)
    name = _display_name(url)
    expected = sha256.lower()
    http = session or requests.Session()
    base_headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    base_headers.update(headers or {})

    offset = part.stat().st_size if part.exists() else 0
    if offset > size_bytes:
        logger.warning("Partial download of %s is larger than expected; starting over.", name)
        part.unlink()
        offset = 0
    digest = _hash_prefix(part, offset, cancel, progress) if offset else hashlib.sha256()
    if progress is not None:
        progress(offset, 0)

    attempts = 0
    try:
        while offset < size_bytes:
            if cancel is not None and cancel.is_set():
                raise JobCanceled()
            request_headers = dict(base_headers)
            if offset:
                request_headers["Range"] = f"bytes={offset}-"
            progressed = False
            try:
                with http.get(url, headers=request_headers, stream=True, allow_redirects=True,
                              timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS)) as response:
                    status = response.status_code
                    if status in (401, 403):
                        raise JobError(auth_error_code, auth_error_message or
                                       f"The server refused the download of {name} (HTTP {status}).",
                                       {"http_status": status})
                    if status == 404:
                        raise JobError("HTTP_ERROR", f"{name} was not found on the server (HTTP 404).",
                                       {"http_status": status})
                    if status == 416:
                        if offset == size_bytes:
                            break
                        logger.warning("Server refused to resume %s at byte %d; starting over.", name, offset)
                        offset, digest = 0, hashlib.sha256()
                        part.unlink(missing_ok=True)
                        raise _Retry()
                    if status in RETRY_STATUS:
                        raise _Retry(f"HTTP {status}")
                    if status not in (200, 206):
                        raise JobError("HTTP_ERROR", f"The server answered HTTP {status} for {name}.",
                                       {"http_status": status})
                    mode = "ab"
                    if status == 206:
                        start, total = _content_range(response.headers.get("Content-Range"))
                        if start is not None and start != offset:
                            raise _Retry("unexpected Content-Range")
                        if total is not None and total != size_bytes:
                            raise JobError("SIZE_MISMATCH",
                                           f"{name} on the server is {total} bytes, expected {size_bytes}.")
                    else:  # 200: whole file
                        if offset:
                            logger.info("Server ignored the resume request for %s; starting over.", name)
                        offset, digest, mode = 0, hashlib.sha256(), "wb"
                        length = response.headers.get("Content-Length")
                        if length and length.isdigit() and int(length) != size_bytes:
                            raise JobError("SIZE_MISMATCH",
                                           f"{name} on the server is {length} bytes, expected {size_bytes}.")
                    with open(part, mode) as handle:
                        for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                            if cancel is not None and cancel.is_set():
                                handle.flush()
                                raise JobCanceled()
                            if not chunk:
                                continue
                            if offset + len(chunk) > size_bytes:
                                raise JobError("SIZE_MISMATCH",
                                               f"{name} is larger on the server than the expected {size_bytes} bytes.")
                            handle.write(chunk)
                            digest.update(chunk)
                            offset += len(chunk)
                            progressed = True
                            if progress is not None:
                                progress(offset, len(chunk))
                        handle.flush()
                        try:
                            os.fsync(handle.fileno())
                        except OSError:
                            pass
                if offset < size_bytes:
                    raise _Retry("connection closed early")
            except _Retry as retry:
                attempts = 0 if progressed else attempts + 1
                if attempts >= MAX_ATTEMPTS_WITHOUT_PROGRESS:
                    raise JobError("NETWORK_ERROR",
                                   f"The download of {name} kept failing ({retry.reason}). "
                                   "Check the internet connection and try again; it will resume.") from None
                _wait(sleep, attempts, cancel)
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as exc:
                attempts = 0 if progressed else attempts + 1
                logger.info("Download of %s interrupted (%s) at byte %d; retrying.", name, type(exc).__name__, offset)
                if attempts >= MAX_ATTEMPTS_WITHOUT_PROGRESS:
                    raise JobError("NETWORK_ERROR",
                                   f"Could not download {name} ({type(exc).__name__}). Check the internet "
                                   "connection and try again; the download will resume.") from None
                _wait(sleep, attempts, cancel)
            except requests.RequestException as exc:
                raise JobError("NETWORK_ERROR", f"Could not download {name} ({type(exc).__name__}).") from None
    except OSError as exc:
        if exc.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", -1)):
            raise JobError("INSUFFICIENT_DISK_SPACE", f"The disk filled up while downloading {name}.") from None
        raise JobError("DOWNLOAD_FAILED", f"Could not write {name} ({type(exc).__name__}: {exc.strerror}).") from None

    actual = digest.hexdigest()
    if actual != expected:
        part.unlink(missing_ok=True)
        raise JobError("CHECKSUM_MISMATCH",
                       f"{name} did not match its published checksum and was deleted. Try the download again.",
                       {"expected_sha256": expected, "actual_sha256": actual})
    os.replace(part, dest)
    return actual


class _Retry(Exception):
    def __init__(self, reason: str = "retry"):
        super().__init__(reason)
        self.reason = reason


def _wait(sleep, attempt: int, cancel) -> None:
    delay = RETRY_BACKOFF_SECONDS[min(max(attempt - 1, 0), len(RETRY_BACKOFF_SECONDS) - 1)]
    end = time.monotonic() + delay
    while time.monotonic() < end:
        if cancel is not None and cancel.is_set():
            raise JobCanceled()
        sleep(min(0.25, max(0.0, end - time.monotonic())))
