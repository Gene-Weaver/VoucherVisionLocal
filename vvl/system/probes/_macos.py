"""macOS helpers: sysctl values through libc (no subprocess), with a
``/usr/sbin/sysctl -n`` fallback. Only called on Darwin."""

from __future__ import annotations

import sys
import threading

from .. import _proc
from .._util import to_int

SYSCTL = "/usr/sbin/sysctl"

_libc = None
_libc_lock = threading.Lock()


def _get_libc():
    global _libc
    with _libc_lock:
        if _libc is None:
            import ctypes

            lib = ctypes.CDLL("/usr/lib/libc.dylib", use_errno=True)
            fn = lib.sysctlbyname
            fn.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            fn.restype = ctypes.c_int
            _libc = lib
        return _libc


def sysctl_raw(name):
    """Raw bytes for a sysctl name, or None if it does not exist."""
    try:
        import ctypes

        lib = _get_libc()
        size = ctypes.c_size_t(0)
        key = name.encode("ascii")
        if lib.sysctlbyname(key, None, ctypes.byref(size), None, 0) != 0 or size.value == 0:
            return None
        buf = ctypes.create_string_buffer(size.value)
        if lib.sysctlbyname(key, buf, ctypes.byref(size), None, 0) != 0:
            return None
        return buf.raw[: size.value]
    except Exception:
        return None


def _sysctl_cmd(name, run=_proc.run):
    res = run([_proc.find_tool("sysctl", [SYSCTL]) or SYSCTL, "-n", name], timeout=3.0)
    if not res.ok:
        return None
    text = res.stdout.strip()
    return text or None


def sysctl_str(name, run=_proc.run):
    raw = sysctl_raw(name)
    if raw is not None:
        text = raw.split(b"\0", 1)[0].decode("utf-8", "replace").strip()
        return text or None
    return _sysctl_cmd(name, run)


def sysctl_int(name, run=_proc.run):
    raw = sysctl_raw(name)
    if raw is not None and len(raw) in (1, 2, 4, 8):
        return int.from_bytes(raw, sys.byteorder, signed=False)
    return to_int(_sysctl_cmd(name, run))


def arch_info(run=_proc.run):
    """(native_is_arm64, python_is_translated) on macOS; None when unknown."""
    arm64 = sysctl_int("hw.optional.arm64", run)
    if arm64 is None and sysctl_int("hw.ncpu", run) is not None:
        arm64 = 0  # the key only exists on Apple Silicon (and under Rosetta there)
    translated = sysctl_int("sysctl.proc_translated", run)
    return (
        None if arm64 is None else bool(arm64),
        # The key is missing on Intel Macs: not translated.
        False if translated is None else bool(translated),
    )
