"""JSON result viewer: scan projects, aggregate stats, and decode embedded base64 images.

Images are decoded strictly from inside each `.json` file (never assuming
external `input_image/` or `label_collage/` folders exist) using an independent
worker process pool (`ProcessPoolExecutor`) for speed and parallelism.
"""

from __future__ import annotations

import atexit
import base64
import binascii
import concurrent.futures
import json
import logging
import os
import re
import secrets
import threading
from collections import OrderedDict
from pathlib import Path

from vvl import paths

logger = logging.getLogger(__name__)

MAX_OPEN_PROJECTS = 64
MAX_IMAGE_CACHE = 32
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
IMAGE_KEYS = {
    "input": "base64image_input_resized",
    "collage": "base64image_text_collage",
}


class ViewerError(Exception):
    def __init__(self, code: str, message: str, *, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


def _detect_image_mime(raw: bytes) -> str:
    if raw.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    return "image/jpeg"


def _clean_base64(value: str) -> str:
    text = value.strip()
    if text.startswith("data:") and "," in text:
        text = text.split(",", 1)[1].strip()
    return text


def _decode_image_from_json(json_path_str: str, kind: str) -> tuple[bytes | None, str]:
    """Top-level worker function (runs in an independent process) to decode one embedded image."""
    key = IMAGE_KEYS.get(kind)
    if not key:
        return None, "image/jpeg"
    try:
        with open(json_path_str, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None, "image/jpeg"
    if not isinstance(data, dict):
        return None, "image/jpeg"
    collage = data.get("collage_info")
    raw_b64 = collage.get(key) if isinstance(collage, dict) else None
    if not isinstance(raw_b64, str) or not raw_b64.strip():
        return None, "image/jpeg"
    try:
        raw_bytes = base64.b64decode(_clean_base64(raw_b64), validate=False)
    except (ValueError, binascii.Error):
        return None, "image/jpeg"
    if not raw_bytes:
        return None, "image/jpeg"
    return raw_bytes, _detect_image_mime(raw_bytes)


def _num(val) -> float:
    if val in (None, "", True, False):
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _int_num(val) -> int:
    return int(round(_num(val)))


def _truthy_flag(val) -> bool:
    if isinstance(val, bool):
        return val
    return str(val or "").strip().lower() in ("true", "1", "yes")


def _extract_metrics(data: dict) -> dict:
    """Compute per-record token and cost breakdown from a VoucherVision JSON dict."""
    ocr_info = data.get("ocr_info") if isinstance(data.get("ocr_info"), dict) else {}
    ocr_in = ocr_out = ocr_think = 0
    ocr_cost_in = ocr_cost_out = ocr_cost_think = ocr_cost_total = 0.0
    ocr_models: list[str] = []
    ocr_engines: list[dict] = []

    for model_id, info in ocr_info.items():
        if not isinstance(info, dict):
            continue
        ocr_models.append(str(model_id))
        t_in = _int_num(info.get("tokens_in", info.get("input", 0)))
        t_out = _int_num(info.get("tokens_out", info.get("output", 0)))
        t_thk = _int_num(info.get("thinking_tokens", 0))
        c_in = _num(info.get("cost_in", 0.0))
        c_out = _num(info.get("cost_out", 0.0))
        c_thk = _num(info.get("thinking_cost", 0.0))
        c_tot = _num(info.get("total_cost", c_in + c_out + c_thk))
        ocr_in += t_in
        ocr_out += t_out
        ocr_think += t_thk
        ocr_cost_in += c_in
        ocr_cost_out += c_out
        ocr_cost_think += c_thk
        ocr_cost_total += c_tot
        ocr_engines.append({
            "model": str(model_id),
            "tokens_in": t_in,
            "tokens_out": t_out,
            "thinking_tokens": t_thk,
            "cost_in": c_in,
            "cost_out": c_out,
            "thinking_cost": c_thk,
            "total_cost": c_tot,
        })

    parsing = data.get("parsing_info") if isinstance(data.get("parsing_info"), dict) else {}
    parse_model = str(parsing.get("model") or "").strip()
    parse_in = _int_num(parsing.get("input", parsing.get("tokens_in", 0)))
    parse_out = _int_num(parsing.get("output", parsing.get("tokens_out", 0)))
    parse_think = _int_num(parsing.get("thinking_tokens", 0))
    parse_cost_in = _num(parsing.get("cost_in", 0.0))
    parse_cost_out = _num(parsing.get("cost_out", 0.0))
    parse_cost_think = _num(parsing.get("thinking_cost", 0.0))
    parse_cost_total = _num(parsing.get("total_cost", parse_cost_in + parse_cost_out + parse_cost_think))

    computed_total = ocr_cost_total + parse_cost_total
    reported_total = data.get("total_request_cost_usd")
    total_cost = _num(reported_total) if reported_total not in (None, "") else computed_total

    total_tokens = ocr_in + ocr_out + ocr_think + parse_in + parse_out + parse_think
    return {
        "total_cost_usd": total_cost,
        "ocr_cost_usd": ocr_cost_total,
        "parse_cost_usd": parse_cost_total,
        "total_tokens": total_tokens,
        "ocr_tokens_in": ocr_in,
        "ocr_tokens_out": ocr_out,
        "ocr_thinking_tokens": ocr_think,
        "parse_tokens_in": parse_in,
        "parse_tokens_out": parse_out,
        "parse_thinking_tokens": parse_think,
        "ocr_models": ocr_models,
        "parse_model": parse_model,
        "ocr_engines": ocr_engines,
        "parsing": {
            "model": parse_model,
            "thinking_budget_requested": parsing.get("thinking_budget_requested", ""),
            "tokens_in": parse_in,
            "tokens_out": parse_out,
            "thinking_tokens": parse_think,
            "cost_in": parse_cost_in,
            "cost_out": parse_cost_out,
            "thinking_cost": parse_cost_think,
            "total_cost": parse_cost_total,
        },
    }


def _strip_base64_for_view(data: dict) -> tuple[dict, dict]:
    """Return `(lightweight_record, image_meta)` with giant base64 strings replaced by metadata."""
    out = dict(data)
    collage = data.get("collage_info")
    image_meta = {
        "has_input": False,
        "has_collage": False,
        "input_b64_chars": 0,
        "collage_b64_chars": 0,
    }
    if isinstance(collage, dict):
        clean_collage = {}
        for k, v in collage.items():
            if k == IMAGE_KEYS["input"]:
                if isinstance(v, str) and v.strip():
                    image_meta["has_input"] = True
                    image_meta["input_b64_chars"] = len(v)
                    clean_collage[k] = f"<base64 image ({len(v) * 3 // 4096} KB)>"
                else:
                    clean_collage[k] = None
            elif k == IMAGE_KEYS["collage"]:
                if isinstance(v, str) and v.strip():
                    image_meta["has_collage"] = True
                    image_meta["collage_b64_chars"] = len(v)
                    clean_collage[k] = f"<base64 image ({len(v) * 3 // 4096} KB)>"
                else:
                    clean_collage[k] = None
            else:
                clean_collage[k] = v
        out["collage_info"] = clean_collage
    return out, image_meta


def _summarize_file(path: Path) -> dict | None:
    """Read one JSON file and produce its lightweight index row + stats contribution."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    metrics = _extract_metrics(data)
    fj = data.get("formatted_json")
    if isinstance(fj, str) and fj.strip():
        try:
            fj = json.loads(fj)
        except ValueError:
            fj = {}
    if not isinstance(fj, dict):
        fj = {}

    collage = data.get("collage_info") if isinstance(data.get("collage_info"), dict) else {}
    has_input = isinstance(collage.get(IMAGE_KEYS["input"]), str) and bool(collage.get(IMAGE_KEYS["input"]).strip())
    has_collage = isinstance(collage.get(IMAGE_KEYS["collage"]), str) and bool(collage.get(IMAGE_KEYS["collage"]).strip())
    success = data.get("success") if isinstance(data.get("success"), dict) else {}

    lat = fj.get("decimalLatitude")
    lon = fj.get("decimalLongitude")
    has_coords = False
    if lat not in (None, "") and lon not in (None, ""):
        try:
            la, lo = float(lat), float(lon)
            has_coords = -90.0 <= la <= 90.0 and -180.0 <= lo <= 180.0
        except (TypeError, ValueError):
            pass

    elev = data.get("COP90_elevation_m")
    has_elev = elev not in (None, "")
    wfo = data.get("WFO_info")
    has_wfo = isinstance(wfo, dict) and bool(wfo)

    return {
        "file": path.name,
        "stem": path.stem,
        "filename": str(data.get("filename") or path.stem),
        "scientific_name": str(fj.get("scientificName") or "").strip(),
        "catalog_number": str(fj.get("catalogNumber") or "").strip(),
        "prompt": str(data.get("prompt") or "").strip(),
        "parsed": bool(fj),
        "ocr_ok": _truthy_flag(success.get("ocr")) or bool(str(data.get("ocr") or "").strip()),
        "llm_ok": _truthy_flag(success.get("llm")) or bool(fj),
        "collage_ok": _truthy_flag(success.get("text_collage")),
        "has_coords": has_coords,
        "has_elevation": has_elev,
        "has_wfo": has_wfo,
        "has_input_image": has_input,
        "has_collage_image": has_collage,
        "cost_usd": metrics["total_cost_usd"],
        "total_tokens": metrics["total_tokens"],
        "_metrics": metrics,
    }


def _resolve_json_dir(folder: Path) -> tuple[Path, Path]:
    """Return `(root_dir, json_dir)`. Accepts either a run folder with `json/` or a folder of `.json` files."""
    if not folder.is_dir():
        raise ViewerError("NOT_FOUND", f"Folder not found: {folder}", status=404)
    sub = folder / "json"
    if sub.is_dir():
        return folder, sub
    return folder, folder


class ViewerStore:
    """Open viewer projects + independent worker process pool for base64 image decoding."""

    def __init__(self, *, max_workers: int | None = None):
        self._lock = threading.Lock()
        self._projects: "OrderedDict[str, dict]" = OrderedDict()
        self._image_cache: "OrderedDict[tuple[str, int, str], tuple[bytes, str]]" = OrderedDict()
        self._max_workers = max_workers or min(4, max(2, os.cpu_count() or 2))
        self._pool: concurrent.futures.Executor | None = None
        self._pool_kind: str = "uninitialized"
        atexit.register(self.shutdown)

    def _get_pool(self) -> concurrent.futures.Executor:
        with self._lock:
            if self._pool is not None:
                return self._pool
            try:
                self._pool = concurrent.futures.ProcessPoolExecutor(max_workers=self._max_workers)
                self._pool_kind = "process"
            except Exception as exc:
                logger.warning("ProcessPoolExecutor unavailable (%s); falling back to threads.", exc)
                self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers)
                self._pool_kind = "thread"
            return self._pool

    def shutdown(self) -> None:
        with self._lock:
            pool = self._pool
            self._pool = None
        if pool is not None:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                pool.shutdown(wait=False)
            except Exception:
                pass

    def open_project(self, folder_path: str | os.PathLike) -> dict:
        if not folder_path or not isinstance(folder_path, (str, os.PathLike)):
            raise ViewerError("INVALID_INPUT", "Provide a folder path to open.", status=400)
        expanded = Path(os.path.expanduser(str(folder_path).strip())).resolve()
        root_dir, json_dir = _resolve_json_dir(expanded)

        try:
            json_files = sorted(
                (p for p in json_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json" and not p.name.startswith(".")),
                key=lambda p: p.name.lower(),
            )
        except OSError as exc:
            raise ViewerError("INVALID_INPUT", f"Could not read folder: {exc.strerror or exc}", status=400) from None

        summaries: list[dict] = []
        if len(json_files) > 8:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(json_files))) as pool:
                for item in pool.map(_summarize_file, json_files):
                    if item is not None:
                        summaries.append(item)
        else:
            for p in json_files:
                item = _summarize_file(p)
                if item is not None:
                    summaries.append(item)

        stats = {
            "total_records": len(summaries),
            "parsed_records": 0,
            "ocr_ok": 0,
            "llm_ok": 0,
            "collage_ok": 0,
            "has_coords": 0,
            "has_elevation": 0,
            "has_wfo": 0,
            "total_cost_usd": 0.0,
            "ocr_cost_usd": 0.0,
            "parse_cost_usd": 0.0,
            "total_tokens": 0,
            "ocr_tokens_in": 0,
            "ocr_tokens_out": 0,
            "ocr_thinking_tokens": 0,
            "parse_tokens_in": 0,
            "parse_tokens_out": 0,
            "parse_thinking_tokens": 0,
        }
        ocr_models: set[str] = set()
        parse_models: set[str] = set()
        prompts_used: set[str] = set()
        records: list[dict] = []

        for s in summaries:
            m = s.pop("_metrics")
            stats["parsed_records"] += int(s["parsed"])
            stats["ocr_ok"] += int(s["ocr_ok"])
            stats["llm_ok"] += int(s["llm_ok"])
            stats["collage_ok"] += int(s["collage_ok"])
            stats["has_coords"] += int(s["has_coords"])
            stats["has_elevation"] += int(s["has_elevation"])
            stats["has_wfo"] += int(s["has_wfo"])
            stats["total_cost_usd"] += m["total_cost_usd"]
            stats["ocr_cost_usd"] += m["ocr_cost_usd"]
            stats["parse_cost_usd"] += m["parse_cost_usd"]
            stats["total_tokens"] += m["total_tokens"]
            stats["ocr_tokens_in"] += m["ocr_tokens_in"]
            stats["ocr_tokens_out"] += m["ocr_tokens_out"]
            stats["ocr_thinking_tokens"] += m["ocr_thinking_tokens"]
            stats["parse_tokens_in"] += m["parse_tokens_in"]
            stats["parse_tokens_out"] += m["parse_tokens_out"]
            stats["parse_thinking_tokens"] += m["parse_thinking_tokens"]
            ocr_models.update(m["ocr_models"])
            if m["parse_model"]:
                parse_models.add(m["parse_model"])
            if s["prompt"]:
                prompts_used.add(s["prompt"])
            records.append(s)

        stats["ocr_models"] = sorted(ocr_models)
        stats["parse_models"] = sorted(parse_models)
        stats["prompts"] = sorted(prompts_used)

        project_id = secrets.token_urlsafe(18)
        meta = {
            "id": project_id,
            "name": root_dir.name,
            "path": str(root_dir),
            "json_dir": str(json_dir),
        }
        with self._lock:
            self._projects[project_id] = meta
            while len(self._projects) > MAX_OPEN_PROJECTS:
                self._projects.popitem(last=False)

        return {
            "project": meta,
            "stats": stats,
            "records": records,
        }

    def _record_path(self, project_id: str, filename: str) -> Path:
        if not isinstance(project_id, str) or not _ID_RE.match(project_id):
            raise ViewerError("NOT_FOUND", "Viewer project not found.", status=404)
        with self._lock:
            proj = self._projects.get(project_id)
        if proj is None:
            raise ViewerError("NOT_FOUND", "Viewer project expired. Reopen the folder.", status=404)
        if (
            not isinstance(filename, str)
            or not filename
            or "/" in filename
            or "\\" in filename
            or filename.startswith(".")
            or not filename.lower().endswith(".json")
        ):
            raise ViewerError("INVALID_INPUT", "Invalid record file name.", status=400)
        json_dir = Path(proj["json_dir"])
        candidate = json_dir / filename
        if not paths.is_within(candidate, json_dir) or not candidate.is_file():
            raise ViewerError("NOT_FOUND", f"Record '{filename}' was not found.", status=404)
        return candidate

    def get_record(self, project_id: str, filename: str) -> dict:
        target = self._record_path(project_id, filename)
        try:
            with open(target, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            raise ViewerError("INVALID_INPUT", f"Could not read JSON: {exc}", status=400) from None
        if not isinstance(data, dict):
            raise ViewerError("INVALID_INPUT", "JSON file does not contain an object.", status=400)
        metrics = _extract_metrics(data)
        clean_record, image_meta = _strip_base64_for_view(data)
        return {
            "file": target.name,
            "stem": target.stem,
            "record": clean_record,
            "images": image_meta,
            "metrics": metrics,
        }

    def decode_image(self, project_id: str, filename: str, kind: str) -> tuple[bytes, str]:
        if kind not in IMAGE_KEYS:
            raise ViewerError("INVALID_INPUT", "Image kind must be 'input' or 'collage'.", status=400)
        target = self._record_path(project_id, filename)
        try:
            mtime_ns = target.stat().st_mtime_ns
        except OSError as exc:
            raise ViewerError("NOT_FOUND", f"Record '{filename}' was not found ({exc}).", status=404) from None

        cache_key = (str(target), mtime_ns, kind)
        with self._lock:
            cached = self._image_cache.get(cache_key)
            if cached is not None:
                self._image_cache.move_to_end(cache_key)
                return cached

        pool = self._get_pool()
        future = pool.submit(_decode_image_from_json, str(target), kind)
        raw_bytes, mime = future.result(timeout=30)
        if not raw_bytes:
            raise ViewerError("NOT_FOUND", f"No {kind} image embedded in '{filename}'.", status=404)

        with self._lock:
            self._image_cache[cache_key] = (raw_bytes, mime)
            while len(self._image_cache) > MAX_IMAGE_CACHE:
                self._image_cache.popitem(last=False)
        return raw_bytes, mime
