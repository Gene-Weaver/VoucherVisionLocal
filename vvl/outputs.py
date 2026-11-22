"""Saved results: one folder per press of Run.

    <output_dir>/<folder>/
        json/<image>.json            the full result, as the API returned it
        xlsx/results.xlsx            one row per parsed record (see write_xlsx)
        label_collage/<image>.jpg    the label collage that was sent to OCR
        input_image/<image>.<ext>    the image that was processed
        ocr_txt/<image>.txt          the final OCR text

`output_dir` is the `output_dir` setting (default `<VVL_HOME>/runs`). `<folder>`
is a timestamp (`YYYYMMDD-HH-MM-SS`), or `<name>_<timestamp>` when the user named
it, so a new run never writes into an older one. With `append`, `<folder>` is
exactly `<name>`: files are added to it and a file with the same name is replaced.

The spreadsheet follows VoucherVisionGO's `save_results_to_xlsx` (client.py): a
sheet named "results", `filename` (without extension) first, then the record's
fields in the order they first appear, every cell stored as text. Each cell goes
through VoucherVisionGO's own write-time guards (vendor_vv/OCR_sanitize.py:
flattening, XML-illegal characters, formula injection, the 32,767 character cell
limit, text cell type and format). VoucherVisionGO's CSV export is marked "use at
your own risk" there (Excel converts cells), so only the spreadsheet is written.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import secrets
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path

from vvl import paths
from vvl.fileutil import atomic_write_bytes, atomic_write_text

logger = logging.getLogger(__name__)

SUBDIRS = ("json", "xlsx", "label_collage", "input_image", "ocr_txt")
OPTIONAL_SUBDIRS = ("xlsx", "label_collage", "input_image", "ocr_txt")
STAMP_FORMAT = "%Y%m%d-%H-%M-%S"
XLSX_NAME = "results.xlsx"
SHEET_TITLE = "results"
MAX_NAME_LENGTH = 100
# The spreadsheet is rewritten after every record while it is small, then at most this often.
XLSX_SMALL_ROWS = 100
XLSX_MIN_SECONDS = 3.0
MAX_BATCHES = 64

_FORBIDDEN = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
_ID = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


class OutputError(Exception):
    """A results folder could not be prepared. `code` is the API error code."""

    def __init__(self, code: str, message: str, *, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


def default_dir() -> Path:
    return paths.home() / "runs"


def validate_folder_name(name) -> str:
    """The folder name a user typed, or "" for none. Raises OutputError when it cannot be a folder name."""
    if name is None:
        return ""
    if not isinstance(name, str):
        raise OutputError("INVALID_INPUT", "The folder name must be text.")
    name = name.strip()
    if not name:
        return ""
    if len(name) > MAX_NAME_LENGTH:
        raise OutputError("INVALID_INPUT", f"The folder name is too long (at most {MAX_NAME_LENGTH} characters).")
    if _FORBIDDEN.search(name):
        raise OutputError("INVALID_INPUT", 'The folder name cannot contain any of < > : " / \\ | ? *')
    if name in (".", "..") or name.endswith((".", " ")) or name.startswith("."):
        raise OutputError("INVALID_INPUT", "The folder name cannot start with a dot or end with a dot or space.")
    if name.split(".")[0].upper() in _WINDOWS_RESERVED:
        raise OutputError("INVALID_INPUT", f'"{name}" is a reserved name on Windows. Choose another folder name.')
    return name


def safe_stem(filename: str) -> str:
    """A file name without its extension that is safe on Windows, macOS and Linux."""
    base = os.path.basename(str(filename or "").replace("\\", "/"))
    stem = os.path.splitext(base)[0]
    stem = _FORBIDDEN.sub("_", stem).strip(" .")
    if not stem or stem.upper() in _WINDOWS_RESERVED:
        stem = f"image_{stem}" if stem else "image"
    return stem[:150]


def _extension(filename: str, default: str = ".jpg") -> str:
    ext = os.path.splitext(str(filename or ""))[1].lower()
    return ext if re.fullmatch(r"\.[a-z0-9]{1,8}", ext) else default


def _cell_text(value) -> str:
    """One cell, as text, through VoucherVisionGO's write-time guards."""
    from vvl.vendor_vv.OCR_sanitize import to_excel_literal

    guarded = to_excel_literal(value)
    if guarded is None:
        return ""
    if not isinstance(guarded, str):
        # VoucherVisionGO stores every column as text so Excel does not convert it.
        guarded = to_excel_literal(str(guarded))
    return guarded


def write_xlsx(path: Path, rows: "OrderedDict[str, dict]") -> None:
    """Write `rows` ({file name without extension: formatted_json}) to `path`, replacing it atomically."""
    from openpyxl import Workbook

    from vvl.vendor_vv.OCR_sanitize import write_excel_safe

    columns: list[str] = []
    seen = set()
    for record in rows.values():
        for key in record:
            if key not in seen:
                seen.add(key)
                columns.append(key)

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = SHEET_TITLE
    for col, header in enumerate(["filename", *columns], start=1):
        write_excel_safe(sheet, 1, col, _cell_text(header))
    for row, (stem, record) in enumerate(rows.items(), start=2):
        write_excel_safe(sheet, row, 1, _cell_text(stem))
        for col, key in enumerate(columns, start=2):
            write_excel_safe(sheet, row, col, _cell_text(record.get(key)))

    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        workbook.save(tmp)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _record_of(result: dict):
    """The parsed record of a result, or None (OCR-only runs have none; VoucherVisionGO skips those too)."""
    record = result.get("formatted_json") if isinstance(result, dict) else None
    if isinstance(record, str) and record.strip():
        try:
            record = json.loads(record, object_pairs_hook=OrderedDict)
        except ValueError:
            return None
    return record if isinstance(record, dict) and record else None


def normalize_enabled_outputs(include=None) -> set[str]:
    """Return the set of enabled output subfolders (`json` is always required)."""
    if include is None:
        return set(SUBDIRS)
    enabled = {"json"}
    if isinstance(include, dict):
        unknown = sorted(set(include) - set(SUBDIRS))
        if unknown:
            raise OutputError("INVALID_INPUT", f"Unknown output type(s): {', '.join(unknown)}.")
        for key in OPTIONAL_SUBDIRS:
            val = include.get(key, True)
            if not isinstance(val, bool):
                raise OutputError("INVALID_INPUT", f"save.{key} must be true or false.")
            if val:
                enabled.add(key)
        return enabled
    if isinstance(include, (list, tuple, set)):
        for item in include:
            if not isinstance(item, str) or item not in SUBDIRS:
                raise OutputError("INVALID_INPUT", f"Unknown output type '{item}'.")
            enabled.add(item)
        return enabled
    raise OutputError("INVALID_INPUT", "save must be an object of output types.")


def list_existing_folders(base_dir: str | os.PathLike, *, limit: int = 200) -> list[dict]:
    """Existing run folders inside `base_dir`, newest first."""
    base = Path(base_dir)
    if not base.is_dir():
        return []
    entries = []
    try:
        for child in base.iterdir():
            if child.name.startswith(".") or not child.is_dir():
                continue
            try:
                st = child.stat()
            except OSError:
                continue
            json_dir = child / "json" if (child / "json").is_dir() else child
            try:
                json_count = sum(1 for p in json_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json")
            except OSError:
                json_count = 0
            entries.append({
                "name": child.name,
                "path": str(child),
                "json_dir": str(json_dir),
                "json_count": json_count,
                "modified_at": datetime.datetime.fromtimestamp(st.st_mtime, tz=datetime.timezone.utc).isoformat(),
                "_mtime": st.st_mtime,
            })
    except OSError:
        return []
    entries.sort(key=lambda e: (e["_mtime"], e["name"].lower()), reverse=True)
    return [{k: v for k, v in e.items() if k != "_mtime"} for e in entries[:limit]]


class OutputBatch:
    """The results folder of one run. Safe to use from several request threads."""

    def __init__(
        self,
        batch_id: str,
        path: Path,
        *,
        name: str,
        append: bool,
        existed: bool,
        enabled: set[str] | None = None,
        replace_existing: bool = False,
    ):
        self.id = batch_id
        self.path = path
        self.name = name
        self.append = append
        self.existed = existed
        self.enabled = set(enabled) | {"json"} if enabled is not None else set(SUBDIRS)
        self.replace_existing = bool(replace_existing)
        self.created_at = time.time()
        self._lock = threading.Lock()
        self._xlsx_lock = threading.Lock()
        self._used: dict[str, int] = {}
        self._rows: "OrderedDict[str, dict]" = OrderedDict()
        self._dirty = False
        self._last_xlsx = 0.0
        self.saved = {sub: 0 for sub in SUBDIRS}
        self.errors: list[str] = []
        self.xlsx_path = self.path / "xlsx" / XLSX_NAME
        if append and existed:
            if "xlsx" in self.enabled:
                self._load_existing_rows()
            if not self.replace_existing:
                self._seed_existing_stems()

    # ------------------------------------------------------------- names --

    def _seed_existing_stems(self) -> None:
        """Safe append: record stems already saved in this folder so nothing is overwritten."""
        for sub in ("json", "input_image", "label_collage", "ocr_txt"):
            folder = self.path / sub
            if not folder.is_dir():
                continue
            try:
                for p in folder.iterdir():
                    if p.is_file() and not p.name.startswith("."):
                        self._used.setdefault(p.stem.lower(), 1)
            except OSError:
                continue

    def reserve(self, filename: str) -> str:
        """The file name (without extension) this image's files get; unique within this run
        (and unique across existing files when safely adding to an existing folder)."""
        base = safe_stem(filename)
        with self._lock:
            low = base.lower()
            if low not in self._used:
                self._used[low] = 1
                return base
            n = self._used[low] + 1
            while f"{base}_{n}".lower() in self._used:
                n += 1
            self._used[low] = n
            candidate = f"{base}_{n}"
            self._used[candidate.lower()] = 1
            return candidate

    # ------------------------------------------------------------- files --

    def _fail(self, what: str, exc: Exception) -> None:
        message = f"{what}: {type(exc).__name__}: {exc}"
        logger.error("Saving results to %s failed (%s)", self.path, message)
        with self._lock:
            if len(self.errors) < 50:
                self.errors.append(message)

    def _count(self, sub: str) -> None:
        with self._lock:
            self.saved[sub] += 1

    def save_input(self, stem: str, source_path: str, filename: str) -> None:
        """Copy the image that is being processed (when enabled)."""
        if "input_image" not in self.enabled:
            return
        try:
            shutil.copyfile(source_path, self.path / "input_image" / f"{stem}{_extension(filename)}")
            self._count("input_image")
        except Exception as exc:
            self._fail(f"input image {stem}", exc)

    def save_collage(self, stem: str, jpeg_bytes: bytes) -> None:
        if "label_collage" not in self.enabled:
            return
        try:
            atomic_write_bytes(self.path / "label_collage" / f"{stem}.jpg", jpeg_bytes)
            self._count("label_collage")
        except Exception as exc:
            self._fail(f"label collage {stem}", exc)

    def save_result(self, stem: str, result: dict) -> None:
        """Write the required JSON, plus optional OCR text and spreadsheet row."""
        try:
            atomic_write_text(self.path / "json" / f"{stem}.json", json.dumps(result, indent=2, ensure_ascii=False))
            self._count("json")
        except Exception as exc:
            self._fail(f"JSON {stem}", exc)
        if "ocr_txt" in self.enabled:
            try:
                ocr = result.get("ocr")
                atomic_write_text(self.path / "ocr_txt" / f"{stem}.txt", ocr if isinstance(ocr, str) else "")
                self._count("ocr_txt")
            except Exception as exc:
                self._fail(f"OCR text {stem}", exc)
        if "xlsx" in self.enabled:
            record = _record_of(result)
            if record is not None:
                with self._lock:
                    self._rows[stem] = record
                    self._dirty = True
                    rows = len(self._rows)
                    due = rows <= XLSX_SMALL_ROWS or (time.monotonic() - self._last_xlsx) >= XLSX_MIN_SECONDS
                if due:
                    self.flush()

    # -------------------------------------------------------- spreadsheet --

    def _load_existing_rows(self) -> None:
        """Adding to a folder: start the spreadsheet from the records already saved in it."""
        folder = self.path / "json"
        try:
            files = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".json" and p.is_file())
        except OSError:
            return
        for file in files:
            try:
                record = _record_of(json.loads(file.read_text(encoding="utf-8"), object_pairs_hook=OrderedDict))
            except (OSError, ValueError):
                continue
            if record is not None:
                self._rows[file.stem] = record

    def flush(self) -> None:
        """Write the spreadsheet if records were added since it was last written."""
        if "xlsx" not in self.enabled:
            return
        with self._xlsx_lock:
            with self._lock:
                if not self._dirty:
                    return
                rows = OrderedDict(self._rows)
                self._dirty = False
            try:
                write_xlsx(self.xlsx_path, rows)
                with self._lock:
                    self.saved["xlsx"] = len(rows)
                    self._last_xlsx = time.monotonic()
            except Exception as exc:
                with self._lock:
                    self._dirty = True
                    self._last_xlsx = time.monotonic()
                logger.warning("Could not write %s (%s: %s); it is tried again.", self.xlsx_path, type(exc).__name__, exc)
                self._xlsx_error = exc

    def finish(self) -> dict:
        """Last write of the spreadsheet. If results.xlsx cannot be replaced (open in Excel
        on Windows), the records go to a second file instead of being lost."""
        if "xlsx" not in self.enabled:
            return self.summary()
        self._xlsx_error = None
        self.flush()
        with self._lock:
            still_dirty = self._dirty
            rows = OrderedDict(self._rows)
        if still_dirty:
            stamp = datetime.datetime.now().strftime(STAMP_FORMAT)
            fallback = self.xlsx_path.with_name(f"results_{stamp}.xlsx")
            try:
                write_xlsx(fallback, rows)
                with self._lock:
                    self._dirty = False
                    self.saved["xlsx"] = len(rows)
                self._fail(f"{XLSX_NAME} could not be replaced (is it open in another program?); "
                           f"the records are in {fallback.name}", self._xlsx_error or OSError("write failed"))
            except Exception as exc:
                self._fail("spreadsheet", exc)
        return self.summary()

    def summary(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "path": str(self.path),
                "folder": self.path.name,
                "name": self.name,
                "append": self.append,
                "existed": self.existed,
                "replace_existing": self.replace_existing,
                "enabled": [sub for sub in SUBDIRS if sub in self.enabled],
                "saved": dict(self.saved),
                "xlsx": str(self.xlsx_path) if self.saved["xlsx"] else None,
                "errors": list(self.errors),
            }

    _xlsx_error: Exception | None = None


class OutputStore:
    """The results folders of this run of the app, by id."""

    def __init__(self):
        self._lock = threading.Lock()
        self._batches: "OrderedDict[str, OutputBatch]" = OrderedDict()

    def create(
        self,
        base_dir: str | os.PathLike,
        *,
        name=None,
        append: bool = False,
        include=None,
        replace_existing: bool = False,
    ) -> OutputBatch:
        name = validate_folder_name(name)
        enabled = normalize_enabled_outputs(include)
        base = Path(base_dir)
        if append and not name:
            raise OutputError("INVALID_INPUT", "Name the folder to add to.")
        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OutputError("OUTPUT_DIR_UNAVAILABLE",
                              f"The results folder {base} could not be created ({exc.strerror or exc}).") from None

        if append:
            path = base / name
            existed = path.is_dir()
            if path.exists() and not existed:
                raise OutputError("OUTPUT_DIR_UNAVAILABLE", f"{path} is a file, not a folder.")
        else:
            stamp = datetime.datetime.now().strftime(STAMP_FORMAT)
            folder = f"{name}_{stamp}" if name else stamp
            path, n = base / folder, 1
            while path.exists():        # two runs in the same second
                n += 1
                path = base / f"{folder}-{n}"
            existed = False
        try:
            for sub in SUBDIRS:
                if sub in enabled:
                    (path / sub).mkdir(parents=True, exist_ok=True)
            probe = path / f".vvl-write-test-{secrets.token_hex(4)}"
            probe.write_bytes(b"")
            probe.unlink()
        except OSError as exc:
            raise OutputError("OUTPUT_DIR_UNAVAILABLE",
                              f"Results cannot be written to {path} ({exc.strerror or exc}).") from None

        batch = OutputBatch(
            secrets.token_urlsafe(18),
            path,
            name=name,
            append=append,
            existed=existed,
            enabled=enabled,
            replace_existing=replace_existing,
        )
        with self._lock:
            # A new run: make sure earlier runs' spreadsheets are complete, and forget the oldest.
            earlier = list(self._batches.values())
            self._batches[batch.id] = batch
            while len(self._batches) > MAX_BATCHES:
                self._batches.popitem(last=False)
        for other in earlier:
            other.flush()
        logger.info("Results folder for this run: %s%s", path, " (adding to it)" if append else "")
        return batch

    def get(self, batch_id) -> OutputBatch:
        if not isinstance(batch_id, str) or not _ID.match(batch_id):
            raise OutputError("OUTPUT_BATCH_NOT_FOUND", "The results folder id is not valid.", status=404)
        with self._lock:
            batch = self._batches.get(batch_id)
        if batch is None:
            raise OutputError("OUTPUT_BATCH_NOT_FOUND",
                              "This run's results folder is no longer known. Start the run again.", status=404)
        return batch

    def flush_all(self) -> None:
        with self._lock:
            batches = list(self._batches.values())
        for batch in batches:
            batch.flush()
