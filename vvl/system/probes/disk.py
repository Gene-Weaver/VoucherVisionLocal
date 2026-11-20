"""Disk space at the models folder, and whether that folder is synced to the cloud.

If the folder does not exist yet, the nearest existing parent is measured.
Cloud-synced folders (Dropbox, OneDrive, iCloud Drive, Google Drive, Box) are
detected from the client's own records where cheap (Dropbox info.json, the
OneDrive environment variables) and from well-known path names.
"""

from __future__ import annotations

import json
import os
import re
import shutil

from .. import models_dir as models_dir_mod

NETWORK_FS = {"nfs", "nfs4", "cifs", "smb3", "smbfs", "afpfs", "sshfs", "fuse.sshfs", "davfs", "fuse.rclone",
              "webdav", "fuse.gvfsd-fuse", "ncpfs", "afs"}
FAT_FS = {"vfat", "msdos", "fat", "fat16", "fat32"}
WSL_WINDOWS_FS = {"9p", "drvfs", "v9fs"}


def find_existing_parent(path):
    p = os.path.abspath(path)
    for _ in range(256):
        if os.path.exists(p):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            return p if os.path.exists(p) else None
        p = parent
    return None


def _is_under(path, root):
    try:
        a = os.path.normcase(os.path.normpath(path))
        b = os.path.normcase(os.path.normpath(root))
        return os.path.commonpath([a, b]) == b
    except ValueError:  # different drives on Windows
        return False


def dropbox_roots(env, home):
    """Dropbox folders recorded in the client's info.json (paths only)."""
    files = [os.path.join(home, ".dropbox", "info.json")]
    for var in ("APPDATA", "LOCALAPPDATA"):
        if env.get(var):
            files.append(os.path.join(env[var], "Dropbox", "info.json"))
    roots = []
    for f in files:
        try:
            if not os.path.isfile(f) or os.path.getsize(f) > 100_000:
                continue
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                data = json.load(fh)
            for v in (data.values() if isinstance(data, dict) else []):
                if isinstance(v, dict) and isinstance(v.get("path"), str) and v["path"]:
                    roots.append(v["path"])
        except Exception:
            continue
    return roots


def known_sync_roots(env, home):
    roots = [("Dropbox", r) for r in dropbox_roots(env, home)]
    for var in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        if env.get(var):
            roots.append(("OneDrive", env[var]))
    roots.append(("iCloud Drive", os.path.join(home, "Library", "Mobile Documents")))
    roots.append(("iCloud Drive", os.path.join(home, "iCloudDrive")))
    return roots


def detect_synced_folder(path, roots=()):
    """Provider name ("Dropbox", "OneDrive", ...) if ``path`` is in a synced folder."""
    for provider, root in roots:
        if root and _is_under(path, root):
            return provider
    parts = [p for p in re.split(r"[\\/]+", path or "") if p]
    prev = ""
    for comp in parts:
        c = comp.lower()
        if prev == "cloudstorage":  # macOS File Provider: ~/Library/CloudStorage/<Provider>-<account>
            for prefix, provider in (("dropbox", "Dropbox"), ("onedrive", "OneDrive"),
                                     ("googledrive", "Google Drive"), ("box", "Box"), ("icloud", "iCloud Drive")):
                if c.startswith(prefix):
                    return provider
            return f"cloud storage ({comp.split('-')[0]})"
        if c == "dropbox" or c.startswith("dropbox (") or c.startswith("dropbox-"):
            return "Dropbox"
        if c == "onedrive" or c.startswith("onedrive - ") or c.startswith("onedrive-"):
            return "OneDrive"
        if c in ("mobile documents", "iclouddrive", "icloud drive", "com~apple~clouddocs"):
            return "iCloud Drive"
        if c in ("google drive", "my drive", "googledrive", "shared drives") or c.startswith("googledrive-"):
            return "Google Drive"
        if c in ("box sync",):
            return "Box"
        prev = c
    return None


def filesystem_info(path, system):
    """(fstype, is_network) for the filesystem holding ``path``; Nones if unknown."""
    if system == "Windows":
        from . import _windows

        fs, net = _windows.volume_info(path)
        if path.startswith("\\\\"):
            net = True
        return fs, net
    try:
        import psutil

        best, best_len = None, -1
        real = os.path.realpath(path)
        for part in psutil.disk_partitions(all=True):
            mp = part.mountpoint
            if mp and _is_under(real, mp) and len(mp) > best_len:
                best, best_len = part, len(mp)
        if best is None:
            return None, None
        fstype = (best.fstype or "").lower() or None
        return fstype, (fstype in NETWORK_FS or (fstype or "").startswith("nfs")) if fstype else None
    except Exception:
        return None, None


def probe(ctx, log):
    path, source = ctx.models_dir, ctx.models_dir_source
    if not path:
        path, source = models_dir_mod.resolve(None, ctx.env)
    out = {
        "models_dir": path,
        "models_dir_source": source,
        "exists": None,
        "checked_path": None,
        "total_bytes": None,
        "free_bytes": None,
        "is_synced_folder": None,
        "synced_folder_hint": None,
        "writable": None,
        "filesystem": None,
        "is_network": None,
    }
    if not path:
        log.fail("the models folder could not be determined")
        return out
    out["exists"] = os.path.isdir(path)
    checked = path if out["exists"] else find_existing_parent(path)
    out["checked_path"] = checked
    if checked:
        try:
            usage = shutil.disk_usage(checked)
            out["total_bytes"], out["free_bytes"] = int(usage.total), int(usage.free)
        except OSError as exc:
            log.fail(f"disk usage: {exc}")
        try:
            out["writable"] = os.access(checked, os.W_OK)
        except OSError:
            pass
        out["filesystem"], out["is_network"] = filesystem_info(checked, ctx.system)
    try:
        home = os.path.expanduser("~")
        real = os.path.realpath(path)
        provider = detect_synced_folder(path, known_sync_roots(ctx.env, home)) or \
            detect_synced_folder(real, known_sync_roots(ctx.env, home))
        out["is_synced_folder"] = provider is not None
        out["synced_folder_hint"] = provider
    except Exception as exc:
        log.fail(f"synced-folder check: {type(exc).__name__}")
    return out
