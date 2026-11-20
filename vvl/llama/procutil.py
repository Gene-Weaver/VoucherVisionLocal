"""Child-process plumbing for the engine: launch so it can never be orphaned,
and stop it (and anything it started) for sure.

Linux    the engine runs in its own session/process group and gets
         PR_SET_PDEATHSIG = SIGKILL, so it dies with VVL even when VVL is
         SIGKILLed. PDEATHSIG fires when the *thread* that forked the child
         exits, so engines are spawned from one long-lived launcher thread
         (`Launcher`) and never from a short-lived worker thread.
macOS    no PDEATHSIG: the engine is started through `vvl.llama.lifeline`, a
         tiny guardian that runs the engine as its child and kills it as soon
         as VVL (its parent) is gone.
Windows  the engine is assigned to a Job object with
         JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; the job handle lives in VVL, so the
         engine dies when VVL's process ends for any reason. Stopping sends
         CTRL_BREAK to its process group, then `taskkill /T /F`.

Everywhere: a pidfile under VVL_HOME (pid + process start time + command)
lets the next VVL start reap an engine a crashed VVL left behind, without
ever killing an unrelated process that reused the pid.
"""

from __future__ import annotations

import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time

import psutil

logger = logging.getLogger(__name__)

IS_WINDOWS = os.name == "nt"
IS_LINUX = sys.platform.startswith("linux")
IS_MAC = sys.platform == "darwin"

PR_SET_PDEATHSIG = 1

_prctl = None
if IS_LINUX:
    try:
        import ctypes

        _libc = ctypes.CDLL(None, use_errno=True)
        _prctl = _libc.prctl
        _prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        _prctl.restype = ctypes.c_int
    except Exception:  # noqa: BLE001
        _prctl = None


def _linux_preexec(parent_pid: int):
    """Runs in the child between fork and exec: die with the parent."""
    def _preexec():
        if _prctl is not None:
            _prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
        # The parent may have died between fork and prctl.
        if os.getppid() != parent_pid:
            os._exit(1)
    return _preexec


# ----------------------------------------------------------------- Windows --

class _WindowsJob:
    """A kill-on-close Job object (ctypes; no pywin32)."""

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32 = kernel32

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.handle = kernel32.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        JobObjectExtendedLimitInformation = 9
        kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        if not kernel32.SetInformationJobObject(self.handle, JobObjectExtendedLimitInformation,
                                                ctypes.byref(info), ctypes.sizeof(info)):
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")

    def assign(self, proc: subprocess.Popen) -> None:
        from ctypes import wintypes

        self._kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        if not self._kernel32.AssignProcessToJobObject(self.handle, int(proc._handle)):  # noqa: SLF001
            raise OSError(self._ctypes.get_last_error(), "AssignProcessToJobObject failed")

    def terminate(self) -> None:
        try:
            self._kernel32.TerminateJobObject(self.handle, 1)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------- launcher --

class Launcher:
    """One long-lived thread that does every engine spawn (see module docstring)."""

    def __init__(self):
        self._queue: "queue.Queue" = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="vvl-engine-launcher", daemon=True)
        self._thread.start()
        self._windows_jobs: dict[int, _WindowsJob] = {}

    def _loop(self):
        while True:
            func, result = self._queue.get()
            try:
                result.put((True, func()))
            except BaseException as exc:  # noqa: BLE001
                result.put((False, exc))

    def run(self, func):
        result: "queue.Queue" = queue.Queue(maxsize=1)
        self._queue.put((func, result))
        ok, value = result.get()
        if not ok:
            raise value
        return value

    def spawn(self, argv: list[str], *, env: dict, cwd: str | None, use_lifeline: bool | None = None) -> subprocess.Popen:
        """Start the engine; stdout+stderr are one pipe of bytes."""
        lifeline = IS_MAC if use_lifeline is None else use_lifeline

        def _spawn():
            command = list(argv)
            if lifeline:
                # Run the guardian by path in isolated mode: it needs only the
                # standard library, so it works the same in a packaged app.
                guardian = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lifeline.py")
                command = [sys.executable, "-I", guardian, str(os.getpid()), "--"] + command
            kwargs = dict(stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          env=env, cwd=cwd, close_fds=True)
            if IS_WINDOWS:
                kwargs["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                           | getattr(subprocess, "CREATE_NO_WINDOW", 0))
            else:
                kwargs["start_new_session"] = True
                if IS_LINUX and not lifeline:
                    kwargs["preexec_fn"] = _linux_preexec(os.getpid())
            proc = subprocess.Popen(command, **kwargs)
            if IS_WINDOWS:
                try:
                    job = _WindowsJob()
                    job.assign(proc)
                    self._windows_jobs[proc.pid] = job
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not attach the engine to a Windows job object (%s).", exc)
            return proc

        return self.run(_spawn)

    def release(self, pid: int) -> None:
        self._windows_jobs.pop(pid, None)

    def windows_job(self, pid: int):
        return self._windows_jobs.get(pid)


# ------------------------------------------------------------------ stopping --

def terminate_tree(proc: subprocess.Popen, *, term_timeout: float = 10.0, kill_timeout: float = 5.0,
                   windows_job=None) -> str:
    """Stop the engine: polite signal, then force. Returns how it ended."""
    if proc.poll() is not None:
        return "already exited"
    if IS_WINDOWS:
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.wait(term_timeout)
            return "stopped"
        except subprocess.TimeoutExpired:
            pass
        if windows_job is not None:
            windows_job.terminate()
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, check=False)
        try:
            proc.wait(kill_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
        return "killed"
    pgid = None
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pass
    _signal_group(proc, pgid, signal.SIGTERM)
    try:
        proc.wait(term_timeout)
        _signal_group(None, pgid, signal.SIGKILL)  # anything the engine left in its group
        return "stopped"
    except subprocess.TimeoutExpired:
        pass
    _signal_group(proc, pgid, signal.SIGKILL)
    try:
        proc.wait(kill_timeout)
    except subprocess.TimeoutExpired:
        logger.error("Engine process %s did not exit after SIGKILL.", proc.pid)
    return "killed"


def _signal_group(proc, pgid, sig) -> None:
    if pgid is not None and pgid != os.getpgid(0):
        try:
            os.killpg(pgid, sig)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass
    if proc is not None:
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass


# ------------------------------------------------------------------ identity --

def process_identity(pid: int) -> dict | None:
    """{pid, create_time, cmdline, exe} or None when the process is gone."""
    try:
        proc = psutil.Process(pid)
        if proc.status() == psutil.STATUS_ZOMBIE:
            return None
        with proc.oneshot():
            info = {"pid": pid, "create_time": proc.create_time()}
            try:
                info["cmdline"] = proc.cmdline()
            except (psutil.AccessDenied, psutil.ZombieProcess):
                info["cmdline"] = None
            try:
                info["exe"] = proc.exe()
            except (psutil.AccessDenied, psutil.ZombieProcess, OSError):
                info["exe"] = None
        return info
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return None
    except psutil.AccessDenied:
        return {"pid": pid, "create_time": None, "cmdline": None, "exe": None}


def same_process(record: dict, live: dict | None) -> bool:
    """True only when `live` is provably the process the record describes."""
    if not live or live.get("create_time") is None or record.get("create_time") is None:
        return False
    if abs(float(live["create_time"]) - float(record["create_time"])) > 1.0:
        return False
    want = record.get("cmdline") or []
    have = live.get("cmdline") or []
    if not want or not have:
        return False
    return list(have) == list(want)


def kill_pid_tree(pid: int, *, term_timeout: float = 5.0) -> None:
    """Stop a process we do not hold a Popen for (the stale-engine reaper)."""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    procs = [parent]
    try:
        procs += parent.children(recursive=True)
    except psutil.Error:
        pass
    for p in procs:
        try:
            p.terminate()
        except psutil.Error:
            pass
    _gone, alive = psutil.wait_procs(procs, timeout=term_timeout)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=term_timeout)


def wait_gone(pid: int, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if process_identity(pid) is None:
            return True
        time.sleep(0.05)
    return process_identity(pid) is None
