"""The one place probes run external programs.

Rules enforced here for every command:
- explicit argument list, never a shell;
- a hard timeout (default 5 s) after which the child (and, on POSIX, its whole
  process group, which we created) is killed;
- stdin closed, stdout/stderr captured, tolerant decoding;
- no console window on Windows (CREATE_NO_WINDOW + hidden STARTUPINFO);
- a stable "C" locale on POSIX so tool output parses the same everywhere.
"""

from __future__ import annotations

import locale
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field

DEFAULT_TIMEOUT_S = 5.0
MAX_TIMEOUT_S = 5.0
_MAX_OUTPUT_CHARS = 4_000_000


@dataclass
class CmdResult:
    argv: list
    found: bool = True
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    error: str | None = None
    duration_ms: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.found and not self.timed_out and self.error is None and self.returncode == 0

    @property
    def output(self) -> str:
        """stdout and stderr together (tools disagree on where errors go)."""
        if self.stdout and self.stderr:
            return self.stdout + "\n" + self.stderr
        return self.stdout or self.stderr or ""

    def describe_failure(self) -> str:
        name = os.path.basename(str(self.argv[0])) if self.argv else "command"
        if not self.found:
            return f"{name} not found"
        if self.timed_out:
            return f"{name} did not finish within the time limit"
        if self.error:
            return f"{name} could not run: {self.error}"
        return f"{name} exited with code {self.returncode}"


def _fallback_encodings():
    encs = []
    if os.name == "nt":
        encs += ["oem", "mbcs"]
    try:
        pref = locale.getpreferredencoding(False)
        if pref:
            encs.append(pref)
    except Exception:
        pass
    encs.append("latin-1")
    return encs


def decode_output(data) -> str:
    """Decode tool output without ever raising."""
    if not data:
        return ""
    if isinstance(data, str):
        return data
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            pass
    if data[:3] == b"\xef\xbb\xbf":
        data = data[3:]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = None
        for enc in _fallback_encodings():
            try:
                text = data.decode(enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        if text is None:
            text = data.decode("utf-8", errors="replace")
    if len(text) > _MAX_OUTPUT_CHARS:
        text = text[:_MAX_OUTPUT_CHARS]
    return text.replace("\r\n", "\n")


def find_tool(name, candidates=()):
    """Return an executable path for ``name``: PATH first, then known locations."""
    try:
        found = shutil.which(name)
    except Exception:
        found = None
    if found:
        return found
    for cand in candidates:
        try:
            if cand and os.path.isfile(cand):
                return cand
        except (OSError, ValueError):
            continue
    return None


def _posix_env():
    env = dict(os.environ)
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    env.pop("LANGUAGE", None)
    return env


def _windows_kwargs():
    kwargs = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    try:
        si = subprocess.STARTUPINFO()
        si.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 1)
        si.wShowWindow = 0  # SW_HIDE
        kwargs["startupinfo"] = si
    except Exception:
        pass
    return kwargs


def _kill(proc):
    try:
        if os.name != "nt":
            try:
                # The child was started in its own session, so its process group
                # contains only processes this probe started.
                os.killpg(proc.pid, signal.SIGKILL)
                return
            except Exception:
                pass
        proc.kill()
    except Exception:
        pass


def run(argv, timeout=DEFAULT_TIMEOUT_S) -> CmdResult:
    """Run ``argv`` (a list) with a hard timeout. Never raises."""
    start = time.perf_counter()
    argv = [str(a) for a in argv]
    timeout = max(0.1, min(float(timeout or DEFAULT_TIMEOUT_S), MAX_TIMEOUT_S))

    def done(**kw):
        kw.setdefault("argv", argv)
        res = CmdResult(**kw)
        res.duration_ms = int((time.perf_counter() - start) * 1000)
        return res

    if not argv:
        return done(found=False, error="empty command")
    exe = argv[0]
    if not os.path.isabs(exe):
        resolved = find_tool(exe)
        if resolved is None:
            return done(found=False)
        argv = [resolved] + argv[1:]

    kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "shell": False,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs.update(_windows_kwargs())
    else:
        kwargs["env"] = _posix_env()
        kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(argv, **kwargs)
    except FileNotFoundError:
        return done(argv=argv, found=False)
    except (OSError, ValueError) as exc:
        return done(argv=argv, error=f"{type(exc).__name__}: {exc}")

    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill(proc)
        try:
            out, err = proc.communicate(timeout=1.0)
        except Exception:
            # A grandchild may still hold the pipes; give up on the output.
            out, err = b"", b""
    except Exception as exc:  # pragma: no cover - defensive
        _kill(proc)
        return done(argv=argv, error=f"{type(exc).__name__}: {exc}")

    return done(
        argv=argv,
        returncode=None if timed_out else proc.returncode,
        stdout=decode_output(out),
        stderr=decode_output(err),
        timed_out=timed_out,
    )
