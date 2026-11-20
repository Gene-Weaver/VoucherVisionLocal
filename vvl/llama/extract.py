"""Safe archive extraction for engine downloads (.tar.gz and .zip).

Every member is checked before anything is written:
  * no absolute paths, drive letters, or `..` components;
  * symbolic and hard links must point inside the target folder;
  * no device files, FIFOs, or other special members;
  * a cap on member count and total unpacked size (zip-bomb guard).
Regular files keep their executable bits (zip entries carry Unix modes in
`external_attr`). Setuid/setgid/sticky bits and group/other write bits are
dropped.
"""

from __future__ import annotations

import os
import posixpath
import stat
import tarfile
import zipfile
from pathlib import Path

MAX_MEMBERS = 20000
MAX_TOTAL_BYTES = 16 * 1024 ** 3


class UnsafeArchive(Exception):
    """The archive tries to write outside the target or contains a forbidden member."""


def _clean_name(name: str) -> str:
    """Normalized relative POSIX path, or UnsafeArchive."""
    if not name or "\x00" in name:
        raise UnsafeArchive("an entry has an empty or invalid name")
    unified = name.replace("\\", "/")
    if unified.startswith("/") or (len(unified) > 1 and unified[1] == ":"):
        raise UnsafeArchive(f"absolute path in archive: {name!r}")
    parts = [p for p in unified.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise UnsafeArchive(f"path traversal in archive: {name!r}")
    if not parts:
        return ""
    return "/".join(parts)


def _link_target_ok(member_name: str, target: str) -> bool:
    """A relative link target that stays inside the extraction root."""
    if not target or target.startswith("/") or "\\" in target or (len(target) > 1 and target[1] == ":"):
        return False
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(member_name), target))
    return not (resolved == ".." or resolved.startswith("../") or resolved.startswith("/"))


def _safe_mode(mode: int, is_dir: bool) -> int:
    mode &= 0o777
    mode &= ~0o022          # no group/other write
    if is_dir:
        mode |= 0o700
    else:
        mode |= 0o600
    return mode


def _ensure_inside(root: Path, path: Path) -> None:
    root_real = os.path.realpath(root)
    path_real = os.path.realpath(path)
    if path_real != root_real and not path_real.startswith(root_real + os.sep):
        raise UnsafeArchive(f"entry would be written outside the target: {path}")


def extract_tar(archive: Path, target: Path) -> list[str]:
    target.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        if len(members) > MAX_MEMBERS:
            raise UnsafeArchive("the archive has too many entries")
        total = 0
        plan = []
        for member in members:
            name = _clean_name(member.name)
            if not name:
                continue
            if member.isdir():
                plan.append(("dir", name, member))
            elif member.isreg():
                total += member.size
                plan.append(("file", name, member))
            elif member.issym():
                if not _link_target_ok(name, member.linkname):
                    raise UnsafeArchive(f"link escapes the target: {member.name!r} -> {member.linkname!r}")
                plan.append(("symlink", name, member))
            elif member.islnk():
                link = _clean_name(member.linkname)
                if not link:
                    raise UnsafeArchive(f"hard link with an empty target: {member.name!r}")
                plan.append(("hardlink", name, member))
            else:
                raise UnsafeArchive(f"special file in archive: {member.name!r}")
        if total > MAX_TOTAL_BYTES:
            raise UnsafeArchive("the archive unpacks to more than the allowed size")

        for kind, name, member in plan:
            dest = target / name
            _ensure_inside(target, dest.parent)
            if kind == "dir":
                dest.mkdir(parents=True, exist_ok=True)
                os.chmod(dest, _safe_mode(member.mode or 0o755, True))
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            _ensure_inside(target, dest.parent)
            if dest.is_symlink() or dest.exists():
                if dest.is_dir() and not dest.is_symlink():
                    raise UnsafeArchive(f"entry replaces a folder: {member.name!r}")
                dest.unlink()
            if kind == "file":
                source = tar.extractfile(member)
                if source is None:
                    raise UnsafeArchive(f"unreadable entry: {member.name!r}")
                with source, open(dest, "wb") as out:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        out.write(block)
                os.chmod(dest, _safe_mode(member.mode or 0o644, False))
            elif kind == "symlink":
                os.symlink(member.linkname, dest)
            elif kind == "hardlink":
                source_path = target / _clean_name(member.linkname)
                _ensure_inside(target, source_path)
                if not source_path.is_file():
                    raise UnsafeArchive(f"hard link to a missing file: {member.name!r}")
                try:
                    os.link(source_path, dest)
                except OSError:
                    import shutil

                    shutil.copy2(source_path, dest)
            written.append(name)
    _check_links(target)
    return written


def extract_zip(archive: Path, target: Path) -> list[str]:
    target.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_MEMBERS:
            raise UnsafeArchive("the archive has too many entries")
        if sum(i.file_size for i in infos) > MAX_TOTAL_BYTES:
            raise UnsafeArchive("the archive unpacks to more than the allowed size")
        plan = []
        for info in infos:
            name = _clean_name(info.filename)
            if not name:
                continue
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            file_type = stat.S_IFMT(unix_mode) if unix_mode else 0
            if info.is_dir() or file_type == stat.S_IFDIR:
                plan.append(("dir", name, info, unix_mode))
            elif file_type == stat.S_IFLNK:
                link = zf.read(info).decode("utf-8", "replace")
                if not _link_target_ok(name, link):
                    raise UnsafeArchive(f"link escapes the target: {info.filename!r} -> {link!r}")
                plan.append(("symlink", name, info, link))
            elif file_type in (0, stat.S_IFREG):
                plan.append(("file", name, info, unix_mode))
            else:
                raise UnsafeArchive(f"special file in archive: {info.filename!r}")
        for kind, name, info, extra in plan:
            dest = target / name
            _ensure_inside(target, dest.parent)
            if kind == "dir":
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            _ensure_inside(target, dest.parent)
            if dest.is_symlink() or dest.exists():
                if dest.is_dir() and not dest.is_symlink():
                    raise UnsafeArchive(f"entry replaces a folder: {info.filename!r}")
                dest.unlink()
            if kind == "symlink":
                if os.name == "nt":
                    raise UnsafeArchive(f"symbolic link in a Windows archive: {info.filename!r}")
                os.symlink(extra, dest)
            else:
                with zf.open(info) as source, open(dest, "wb") as out:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        out.write(block)
                mode = extra & 0o777 if extra else 0o644
                if os.name != "nt":
                    os.chmod(dest, _safe_mode(mode, False))
            written.append(name)
    _check_links(target)
    return written


def _check_links(target: Path) -> None:
    """After extraction, every symlink must still resolve inside the target."""
    root = os.path.realpath(target)
    for dirpath, dirnames, filenames in os.walk(target):
        for entry in dirnames + filenames:
            path = os.path.join(dirpath, entry)
            if os.path.islink(path):
                real = os.path.realpath(path)
                if real != root and not real.startswith(root + os.sep):
                    raise UnsafeArchive(f"link escapes the target: {os.path.relpath(path, target)}")


def extract(archive: Path, target: Path) -> list[str]:
    lower = archive.name.lower()
    if lower.endswith((".tar.gz", ".tgz", ".tar")):
        return extract_tar(archive, target)
    if lower.endswith(".zip"):
        return extract_zip(archive, target)
    raise UnsafeArchive(f"unknown archive type: {archive.name}")
