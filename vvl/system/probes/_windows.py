"""Windows helpers: registry reads through ``winreg`` and a few documented
Win32 calls through ``ctypes``. Every function returns None/empty on failure
and is only called on Windows."""

from __future__ import annotations

import os

DISPLAY_CLASS_GUID = "{4d36e968-e325-11ce-bfc1-08002be10318}"

# IMAGE_FILE_MACHINE_* values returned by IsWow64Process2.
_MACHINES = {0x8664: "x86_64", 0xAA64: "arm64", 0x014C: "x86", 0x01C4: "arm"}


def reg_values(root_name, path):
    """All values of one registry key as a dict, or None if it cannot be opened.

    Types map as winreg returns them: REG_SZ -> str, REG_DWORD/QWORD -> int,
    REG_BINARY -> bytes, REG_MULTI_SZ -> list[str].
    """
    try:
        import winreg

        root = getattr(winreg, root_name)
        out = {}
        with winreg.OpenKey(root, path, 0, winreg.KEY_READ) as key:
            i = 0
            while True:
                try:
                    name, value, _typ = winreg.EnumValue(key, i)
                except OSError:
                    break
                out[name] = value
                i += 1
                if i > 4096:
                    break
        return out
    except Exception:
        return None


def reg_subkeys(root_name, path):
    try:
        import winreg

        root = getattr(winreg, root_name)
        names = []
        with winreg.OpenKey(root, path, 0, winreg.KEY_READ) as key:
            i = 0
            while True:
                try:
                    names.append(winreg.EnumKey(key, i))
                except OSError:
                    break
                i += 1
                if i > 4096:
                    break
        return names
    except Exception:
        return None


def native_machine():
    """Native CPU architecture via IsWow64Process2 (Windows 10 1709+), else env."""
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        fn = getattr(k32, "IsWow64Process2", None)
        if fn is not None:
            fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_ushort), ctypes.POINTER(ctypes.c_ushort)]
            fn.restype = wintypes.BOOL
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            proc_m = ctypes.c_ushort(0)
            native_m = ctypes.c_ushort(0)
            if fn(k32.GetCurrentProcess(), ctypes.byref(proc_m), ctypes.byref(native_m)):
                arch = _MACHINES.get(native_m.value)
                if arch:
                    return arch
    except Exception:
        pass
    env = os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get("PROCESSOR_ARCHITECTURE")
    return env or None


# IsProcessorFeaturePresent constants (winnt.h).
PF = {
    "ssse3": 36,
    "sse4_1": 37,
    "sse4_2": 38,
    "avx": 39,
    "avx2": 40,
    "avx512f": 41,
    "arm_neon": 19,
    "arm_v8": 29,
    "arm_v82_dp": 43,
    "arm_sve": 46,
}


def processor_features():
    """{name: bool} from IsProcessorFeaturePresent, or None if the call fails."""
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32")
        fn = k32.IsProcessorFeaturePresent
        fn.argtypes = [wintypes.DWORD]
        fn.restype = wintypes.BOOL
        return {name: bool(fn(code)) for name, code in PF.items()}
    except Exception:
        return None


def present_display_devices():
    """Device instance IDs of display adapters that are present right now.

    Uses cfgmgr32 CM_Get_Device_ID_ListW with the display class filter. Returns
    None if the call is unavailable, so callers can fall back to reading every
    driver key (which may include adapters that were removed).
    """
    try:
        import ctypes

        cfg = ctypes.WinDLL("cfgmgr32")
        flags = 0x00000200 | 0x00000100  # CM_GETIDLIST_FILTER_CLASS | _PRESENT
        get_size = cfg.CM_Get_Device_ID_List_SizeW
        get_size.argtypes = [ctypes.POINTER(ctypes.c_ulong), ctypes.c_wchar_p, ctypes.c_ulong]
        get_size.restype = ctypes.c_ulong
        get_list = cfg.CM_Get_Device_ID_ListW
        get_list.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_ulong]
        get_list.restype = ctypes.c_ulong
        for _attempt in range(3):
            size = ctypes.c_ulong(0)
            if get_size(ctypes.byref(size), DISPLAY_CLASS_GUID, flags) != 0:
                return None
            if size.value <= 1:
                return []
            buf = ctypes.create_unicode_buffer(size.value + 16)
            cr = get_list(DISPLAY_CLASS_GUID, buf, size.value + 16, flags)
            if cr == 0x1A:  # CR_BUFFER_SMALL: list changed between calls
                continue
            if cr != 0:
                return None
            raw = ctypes.wstring_at(ctypes.addressof(buf), size.value + 16)
            return [s for s in raw.split("\x00") if s]
        return None
    except Exception:
        return None


def volume_info(path):
    """(filesystem_name, is_network) for the volume holding ``path``."""
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        buf = ctypes.create_unicode_buffer(1024)
        k32.GetVolumePathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        k32.GetVolumePathNameW.restype = wintypes.BOOL
        if not k32.GetVolumePathNameW(str(path), buf, 1024):
            return None, None
        root = buf.value
        k32.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
        k32.GetDriveTypeW.restype = wintypes.UINT
        is_network = k32.GetDriveTypeW(root) == 4  # DRIVE_REMOTE
        fs = ctypes.create_unicode_buffer(64)
        k32.GetVolumeInformationW.argtypes = [
            wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD), wintypes.LPWSTR, wintypes.DWORD,
        ]
        k32.GetVolumeInformationW.restype = wintypes.BOOL
        ok = k32.GetVolumeInformationW(root, None, 0, None, None, None, fs, 64)
        return (fs.value or None) if ok else None, is_network
    except Exception:
        return None, None
