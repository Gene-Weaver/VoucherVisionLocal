"""The processing pipeline, adapted from VoucherVisionGO's VoucherVisionProcessor.

Same shape and same response JSON as VoucherVisionGO:
  upload -> (resize) -> TextCollage label collage -> OCR (one or more engines)
  -> parse into the prompt's fields -> enrichment -> response assembly.

What changed: the model calls go through the engine seam (vvl/engines), the
prompt text comes from vvl.prompting.PromptBuilder instead of the VoucherVision
class, and every Gemini/Vertex/quota/billing/analytics step is gone. Response
fields `payment_inference` (billing) is removed and `impact` (a Google Cloud
energy estimate) is null; `engine_notice` is added at the end.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import tempfile
import threading
import traceback
import types
from collections import OrderedDict

from werkzeug.utils import secure_filename

from vvl import label_detector, paths, stages
from vvl.engines.base import (
    DEFAULT_THINKING_LEVEL,
    EngineError,
    OCRRequest,
    ParseRequest,
)
from vvl.enrichment import vvgo_client
from vvl.errors import RESPONSE_HOST_NAME, _error_payload, _pdf_page_error, current_request_id
from vvl.images import MAX_PDF_PAGES, PdfTooLong, convert_pdf_to_page_images, process_uploaded_file_with_resize
from vvl.prompting import PromptBuilder, ocr_prompt_text
from vvl.vendor_vv.OCR_sanitize import markdown_to_simple_text, sanitize_excel_record, sanitize_for_storage
from vvl.vendor_vv.utils_LLM_JSON_validation import validate_and_align_JSON_keys_with_template

logger = logging.getLogger(__name__)

COLLAGE_CLASSES = ["barcode", "label", "map"]  # classes rendered into the collage
COLLAGE_RESIZE_METHOD = "gemini"                # no resizing; original resolution


class RequestThrottler:
    """Bounds concurrent processing; a full server answers 429 instead of queuing."""

    def __init__(self, max_concurrent=32):
        self.semaphore = threading.Semaphore(max_concurrent)
        self.active_count = 0
        self.lock = threading.Lock()
        self.max_concurrent = max_concurrent

    def acquire(self):
        acquired = self.semaphore.acquire(blocking=False)
        if acquired:
            with self.lock:
                self.active_count += 1
        return acquired

    def release(self):
        self.semaphore.release()
        with self.lock:
            self.active_count -= 1

    def get_active_count(self):
        with self.lock:
            return self.active_count


def _add_tokens(*values) -> float:
    """Sum token counts; None and "" count as 0."""
    total = 0.0
    for x in values:
        if x in (None, ""):
            continue
        try:
            total += float(x)
        except (ValueError, TypeError):
            pass
    return total


def _empty_parsing_info():
    return OrderedDict([
        ("model", ""),
        ("thinking_budget_requested", ""),
        ("input", 0),
        ("output", 0),
        ("thinking_tokens", 0),
        ("cost_in", 0),
        ("cost_out", 0),
        ("thinking_cost", 0),
        ("total_cost", 0),
    ])


def combine_notices(engines) -> str | None:
    seen = []
    for engine in engines:
        notice = getattr(engine, "notice", None)
        if notice and notice not in seen:
            seen.append(notice)
    return " ".join(seen) if seen else None


class VoucherVisionProcessor:
    """Image -> record. One instance per app; safe to call from several threads."""

    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "tif", "tiff", "heic", "heif", "pdf"}
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB, room for multi-page PDFs

    def __init__(self, max_concurrent=32, detector_factory=None, elevation_lookup=None):
        self.throttler = RequestThrottler(max_concurrent)
        self.collage_engine_lock = threading.Lock()
        self._collage_init_lock = threading.Lock()
        self._collage_engine = None
        self._collage_error = None
        self._missing_logged = False
        # detector_factory() -> textcollage.detector.ArchivalDetector, or raises
        # label_detector.DetectorNotInstalled. Injectable for tests.
        self.detector_factory = detector_factory or label_detector.load_detector
        # elevation_lookup(record) -> int | None. Injectable for tests.
        self.elevation_lookup = elevation_lookup or vvgo_client.elevation_for_record

    # ---------------------------------------------------------- helpers --

    def allowed_file(self, filename):
        return "." in filename and filename.rsplit(".", 1)[1].lower() in self.ALLOWED_EXTENSIONS

    @property
    def collage_engine(self):
        """TextCollage engine, created on first use; None until the label detector is
        downloaded (asked again on every call, so it appears once the download ends)
        or when it failed to load (kept until reset_collage())."""
        if self._collage_engine is None and self._collage_error is None:
            with self._collage_init_lock:
                if self._collage_engine is None and self._collage_error is None:
                    try:
                        detector = self.detector_factory()
                    except label_detector.DetectorNotInstalled as e:
                        if not self._missing_logged:
                            self._missing_logged = True
                            logger.info("No label collage yet: %s.", e)
                        return None
                    except Exception as e:
                        self._collage_error = f"{type(e).__name__}: {e}"
                        logger.error("Failed to load the label detector: %s", self._collage_error)
                        return None
                    from textcollage.CollageEngine import CollageEngine

                    self._collage_engine = CollageEngine(
                        detector=detector,
                        collage_classes=COLLAGE_CLASSES,
                        engine=COLLAGE_RESIZE_METHOD,
                        output_path=None,          # return the collage in memory
                        hide_long_objects=False,
                        draw_overlay=False,
                    )
                    logger.info("CollageEngine ready (label detector: %s)", detector.label)
        return self._collage_engine

    def reset_collage(self):
        """Forget the engine and any load error (after the detector's download), then load it again."""
        with self._collage_init_lock:
            self._collage_engine = None
            self._collage_error = None
            self._missing_logged = False
        threading.Thread(target=self.warm_up, name="vvl-detector-load", daemon=True).start()

    def collage_status(self):
        """What the Setup page and `vvl doctor` show about the loaded detector."""
        engine = self._collage_engine
        detector = engine.detector if engine is not None else None
        return {
            "loaded": engine is not None,
            "loaded_runtime": detector.label if detector is not None else None,
            "gpu_error": getattr(detector, "gpu_error", None),
            "load_error": self._collage_error,
        }

    def warm_up(self):
        """Create the collage engine now (called from a background thread at startup)."""
        return self.collage_engine is not None

    @staticmethod
    def _sanitize_formatted_json(vv_results):
        """Deep-sanitize the parsed record for storage/export (VoucherVisionGO)."""
        if isinstance(vv_results, (dict, list, tuple)):
            return sanitize_excel_record(vv_results)
        if isinstance(vv_results, str):
            try:
                parsed = json.loads(vv_results)
            except Exception:
                return sanitize_for_storage(vv_results)
            return sanitize_excel_record(parsed)
        return sanitize_excel_record(vv_results)

    # -------------------------------------------------------- model calls --

    def perform_ocr(self, file_path, ocr_engines, ocr_prompt_option, ocr_thinking_level=DEFAULT_THINKING_LEVEL):
        """Run every OCR engine on the collage. Returns (ocr_packet, ocr_all, ocr_tokens_total)."""
        prompt_key, prompt_text = ocr_prompt_text(ocr_prompt_option)
        ocr_packet = OrderedDict()
        ocr_all = ""
        ocr_tokens_total = 0.0
        for engine in ocr_engines:
            logger.info("OCR with %s (prompt %s)", engine.model_id, prompt_key)
            result = engine.ocr(OCRRequest(
                image_path=file_path,
                prompt_text=prompt_text,
                prompt_key=prompt_key,
                thinking_level=ocr_thinking_level,
            ))
            ocr_packet[engine.model_id] = result.to_packet()
            ocr_all += f"{result.text} "
            ocr_tokens_total += _add_tokens(result.tokens_in, result.tokens_out, result.thinking_tokens)
        return ocr_packet, ocr_all, ocr_tokens_total

    def process_voucher_vision(self, ocr_text, prompt_path, parse_engine, llm_thinking_level=DEFAULT_THINKING_LEVEL):
        """Parse OCR text into the prompt's fields. Returns (record, ParseResult)."""
        builder = PromptBuilder(prompt_path)
        result = parse_engine.parse(ParseRequest(
            prompt_text=builder.parse_prompt_text(ocr_text),
            template=dict(builder.JSON_dict_structure),
            json_schema=builder.json_schema(),
            thinking_level=llm_thinking_level,
        ))
        record = None
        if result.record is not None:
            record = validate_and_align_JSON_keys_with_template(result.record, builder.JSON_dict_structure)
        return record, result

    # ----------------------------------------------------------- collage --

    def _run_collage(self, image_path, *, use_original):
        """(collage_json_data, collage_ran). Falls back to the whole image when the
        label detector is not downloaded yet, finds no labels, or fails, instead of
        failing the request."""
        from textcollage.CollageEngine import whole_image

        engine = None if use_original else self.collage_engine
        if engine is None:
            return whole_image(image_path), False
        with self.collage_engine_lock:
            try:
                return engine.run(image_path), True
            except Exception as e:
                logger.warning("TextCollage could not build a collage (%s); using the whole image.", e)
                return whole_image(image_path), False

    # ------------------------------------------------------------- images --

    def process_image_request(self, file,
                              ocr_engines,
                              parse_engine=None,
                              ocr_prompt_option=None,
                              prompt=None,
                              prompt_path=None,
                              ocr_only=False,
                              include_wfo=False,
                              include_cop90=False,
                              url_source="",
                              notebook_mode=False,
                              skip_label_collage=False,
                              ocr_thinking_level=DEFAULT_THINKING_LEVEL,
                              llm_thinking_level=DEFAULT_THINKING_LEVEL,
                              elevation_enabled=True,
                              stage=None,
                              stage_token=None,
                              output=None):
        """Process one image. Returns (result_dict, status_code).

        `output` (a vvl.outputs.OutputBatch) saves the input image, the label
        collage, the result JSON, the OCR text and the spreadsheet row into that
        run's results folder. A failure to save never fails the request.

        `stage` splits the work for two-pass batches (see vvl/stages.py):
        "ocr" stops after OCR, saves the result and returns a `stage_token`;
        "parse" picks a saved result up by `stage_token` (no file needed) and
        finishes it, returning the same result a one-pass request returns.
        """
        if not self.throttler.acquire():
            return _error_payload("SERVER_BUSY", "Server is at maximum capacity. Please try again later."), 429

        temp_dir = None
        try:
            if notebook_mode:
                ocr_only = True

            resumed = None
            output_stem = None
            if stage == "parse":
                if parse_engine is None or prompt_path is None:
                    return _error_payload("INVALID_INPUT", "No parsing model or prompt was selected."), 400
                try:
                    resumed = stages.load(stage_token)
                except stages.StageNotFound as exc:
                    return _error_payload("STAGE_NOT_FOUND", str(exc)), 404
                ocr_only = False
                original_filename = resumed["filename"]
                url_source = resumed.get("url_source") or ""
                ocr_info, ocr = resumed["ocr_info"], resumed["ocr"]
                collage_json_data = resumed["collage_info"]
                text_collage_flag = resumed["text_collage_flag"]
                engines_used = [types.SimpleNamespace(notice=resumed.get("ocr_notice")), parse_engine]
                output_stem = resumed.get("output_stem")

            if resumed is None:
                if not file.filename:
                    return _error_payload("NO_FILE", "No file selected"), 400
                if not self.allowed_file(file.filename):
                    return _error_payload(
                        "BAD_FILE_TYPE",
                        f'File type not allowed. Supported types: {", ".join(sorted(self.ALLOWED_EXTENSIONS))}',
                        details={"allowed_extensions": sorted(self.ALLOWED_EXTENSIONS)},
                    ), 415
                if not ocr_engines:
                    return _error_payload("INVALID_INPUT", "No OCR model was selected."), 400
                if not ocr_only and (parse_engine is None or prompt_path is None):
                    return _error_payload("INVALID_INPUT", "No parsing model or prompt was selected."), 400

                # --- STAGE 1: COLLAGE ENGINE PRE-PROCESSING ---
                temp_dir = tempfile.mkdtemp(prefix="vvl-", dir=str(paths.ensure_dir(paths.temp_dir())))
                safe_name = secure_filename(file.filename) or "upload"
                original_temp_path = os.path.join(temp_dir, f"original_{safe_name}")
                file.save(original_temp_path)
                if output is not None:
                    output_stem = output.reserve(file.filename)
                    output.save_input(output_stem, original_temp_path, file.filename)

                with open(original_temp_path, "rb") as f_orig:
                    base64image_input_resized = base64.b64encode(f_orig.read()).decode("utf-8")

                if notebook_mode:
                    logger.info("[notebook_mode] Not running CollageEngine; using the original image.")
                elif skip_label_collage:
                    logger.info("[skip_label_collage] Not running CollageEngine; using the original image.")
                collage_json_data, collage_ran = self._run_collage(
                    original_temp_path, use_original=bool(notebook_mode or skip_label_collage))
                collage_json_data["base64image_input_resized"] = base64image_input_resized

                if collage_json_data.get("base64image_text_collage") is None:
                    raise RuntimeError("CollageEngine failed to produce an image.")

                collage_temp_path = os.path.join(temp_dir, f"collage_{safe_name}")
                collage_bytes = base64.b64decode(collage_json_data["base64image_text_collage"])
                with open(collage_temp_path, "wb") as f:
                    f.write(collage_bytes)
                if output is not None:
                    output.save_collage(output_stem, collage_bytes)

                engines_used = list(ocr_engines) + ([parse_engine] if (parse_engine and not ocr_only) else [])
                text_collage_flag = "True" if collage_ran else "False"
                original_filename = os.path.basename(file.filename)

                if ocr_prompt_option is None:
                    if notebook_mode:
                        ocr_prompt_option = "verbatim_notebook"
                    elif ocr_only:
                        ocr_prompt_option = "verbatim_with_annotations"

                ocr_info, ocr, _ocr_tokens_total = self.perform_ocr(
                    collage_temp_path, ocr_engines, ocr_prompt_option, ocr_thinking_level=ocr_thinking_level)

                if stage == "ocr" and not ocr_only:
                    # First pass of a two-pass batch: keep the OCR result for the parsing pass.
                    token = stages.save({
                        "filename": original_filename, "url_source": url_source, "ocr_info": ocr_info, "ocr": ocr,
                        "collage_info": collage_json_data, "text_collage_flag": text_collage_flag,
                        "ocr_notice": combine_notices(ocr_engines),
                        "output_stem": output_stem,
                    })
                    return OrderedDict([
                        ("staged", True),
                        ("stage", "ocr"),
                        ("stage_token", token),
                        ("filename", original_filename),
                        ("url_source", url_source),
                        ("host", RESPONSE_HOST_NAME),
                        ("ocr_info", sanitize_excel_record(ocr_info)),
                        ("ocr", sanitize_for_storage(ocr)),
                        ("engine_notice", combine_notices(ocr_engines)),
                    ]), 200


            if notebook_mode:
                # "ocr" is Markdown here; the plain field gets a flattened copy.
                ocr_plain = markdown_to_simple_text(ocr, remove_headers=True, guard_excel=True)
                results = OrderedDict([
                    ("filename", original_filename),
                    ("url_source", url_source),
                    ("host", RESPONSE_HOST_NAME),
                    ("prompt", ocr_prompt_option),
                    ("ocr_info", ocr_info),
                    ("WFO_info", ""),
                    ("COP90_elevation_m", ""),
                    ("ocr", ocr_plain),
                    ("formatted_json", ""),
                    ("formatted_md", ocr),
                    ("parsing_info", _empty_parsing_info()),
                    ("impact", None),
                    ("collage_info", collage_json_data),
                    ("collage_image_format", "jpeg"),
                    ("success", {
                        "image_available": "True",
                        "text_collage": "False",
                        "text_collage_resize": COLLAGE_RESIZE_METHOD,
                        "ocr": "True",
                        "llm": "False",
                    }),
                ])
            elif ocr_only:
                results = OrderedDict([
                    ("filename", original_filename),
                    ("url_source", url_source),
                    ("host", RESPONSE_HOST_NAME),
                    ("prompt", ocr_prompt_option),
                    ("ocr_info", sanitize_excel_record(ocr_info)),
                    ("WFO_info", ""),
                    ("COP90_elevation_m", ""),
                    ("ocr", sanitize_for_storage(ocr)),
                    ("formatted_json", ""),
                    ("formatted_md", ""),
                    ("parsing_info", _empty_parsing_info()),
                    ("impact", None),
                    ("collage_info", collage_json_data),
                    ("collage_image_format", "jpeg"),
                    ("success", {
                        "image_available": "True",
                        "text_collage": text_collage_flag,
                        "text_collage_resize": COLLAGE_RESIZE_METHOD,
                        "ocr": "True",
                        "llm": "False",
                    }),
                ])
            else:
                vv_results, parsed = self.process_voucher_vision(
                    ocr, prompt_path, parse_engine, llm_thinking_level=llm_thinking_level)

                # World Flora Online arrives in Phase 3 (VoucherVisionGO also
                # returns "" when no WFO lookup is available).
                WFO = ""

                results = OrderedDict([
                    ("filename", original_filename),
                    ("url_source", url_source),
                    ("host", RESPONSE_HOST_NAME),
                    ("prompt", prompt),
                    ("ocr_info", sanitize_excel_record(ocr_info)),
                    ("WFO_info", WFO),
                    ("COP90_elevation_m", ""),
                    ("ocr", sanitize_for_storage(ocr)),
                    ("formatted_json", self._sanitize_formatted_json(vv_results)),
                    ("formatted_md", ""),
                    ("parsing_info", OrderedDict([
                        ("model", parse_engine.model_id),
                        ("thinking_budget_requested", parsed.thinking_budget_requested),
                        ("input", parsed.tokens_in),
                        ("output", parsed.tokens_out),
                        ("thinking_tokens", parsed.thinking_tokens),
                        ("cost_in", parsed.cost_in),
                        ("cost_out", parsed.cost_out),
                        ("thinking_cost", parsed.thinking_cost),
                        ("total_cost", parsed.cost_in + parsed.cost_out + parsed.thinking_cost),
                    ])),
                    ("impact", None),
                    ("collage_info", collage_json_data),
                    ("collage_image_format", "jpeg"),
                    ("success", {
                        "image_available": "True",
                        "text_collage": text_collage_flag,
                        "text_collage_resize": COLLAGE_RESIZE_METHOD,
                        "ocr": "True",
                        "llm": "True" if vv_results is not None else "False",
                    }),
                ])

            if include_cop90 and elevation_enabled:
                fj = results.get("formatted_json")
                if isinstance(fj, dict):
                    try:
                        elev = self.elevation_lookup(fj)
                        results["COP90_elevation_m"] = elev if elev is not None else ""
                    except Exception as e:
                        logger.info("Elevation lookup skipped (%s).", type(e).__name__)

            # Token-derived cost estimate across OCR engines and parsing.
            try:
                ocr_cost_sum = 0.0
                for v in (results.get("ocr_info") or {}).values():
                    if isinstance(v, dict):
                        ocr_cost_sum += float(v.get("total_cost", 0.0) or 0.0)
                parsing = results.get("parsing_info") or {}
                parsing_cost = (
                    float(parsing.get("cost_in", 0.0) or 0.0)
                    + float(parsing.get("cost_out", 0.0) or 0.0)
                    + float(parsing.get("thinking_cost", 0.0) or 0.0)
                )
                results["total_request_cost_usd"] = ocr_cost_sum + parsing_cost
            except Exception as cost_err:
                logger.warning("Could not compute total_request_cost_usd: %s", cost_err)
                results["total_request_cost_usd"] = 0.0

            results["engine_notice"] = combine_notices(engines_used)
            if resumed is not None:
                stages.discard(stage_token)
            if output is not None:
                output.save_result(output_stem or output.reserve(original_filename), results)
            logger.info("Processing completed successfully")
            return results, 200

        except EngineError as e:
            logger.error("Engine failed (request_id=%s, code=%s): %s", current_request_id(), e.code, e)
            return _error_payload(e.code, str(e)), e.status
        except Exception as e:
            logger.error("Processing failed (request_id=%s): %s\n%s", current_request_id(), e, traceback.format_exc())
            return _error_payload(
                "INTERNAL_ERROR",
                "The request could not be completed. Quote the request id when reporting this.",
            ), 500
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
            self.throttler.release()

    # --------------------------------------------------------------- PDFs --

    def process_pdf_request(self, file, **kwargs):
        """Convert each PDF page to a JPEG and run the image pipeline on it.

        Returns (result_dict, status_code); result_dict has one `pages` entry per page.
        """
        pdf_bytes = file.read()
        pdf_filename = file.filename

        try:
            page_files = convert_pdf_to_page_images(pdf_bytes, pdf_filename, max_pages=MAX_PDF_PAGES)
        except PdfTooLong as e:
            return _error_payload(
                "INVALID_INPUT", str(e),
                details={"page_count": e.page_count, "max_pages": e.max_pages},
            ), 400
        except Exception as e:
            logger.error("PDF conversion failed for '%s' (request_id=%s): %s", pdf_filename, current_request_id(), e)
            return _error_payload(
                "INVALID_INPUT", "Failed to convert PDF. The file may be corrupt or password-protected.",
            ), 400

        if not page_files:
            return _error_payload("INVALID_INPUT", "PDF contains no pages"), 400

        logger.info("PDF '%s' has %d pages, processing each as a separate image", pdf_filename, len(page_files))

        page_results = []
        for page_file in page_files:
            try:
                page_file = process_uploaded_file_with_resize(page_file)
            except Exception as e:
                logger.error("Resize failed for %s: %s", page_file.filename, e)
                page_results.append(_pdf_page_error(
                    page_file.filename, "INTERNAL_ERROR", "Failed to prepare this page for processing.", 500,
                ))
                continue

            result, status_code = self.process_image_request(file=page_file, **kwargs)
            if status_code != 200:
                entry = OrderedDict()
                entry["filename"] = page_file.filename
                entry.update(result if isinstance(result, dict) else {})
                entry.setdefault("error", "Unknown error")
                entry["status_code"] = status_code
                page_results.append(entry)
            else:
                result["filename"] = page_file.filename
                page_results.append(result)

        pdf_cost_total_usd = 0.0
        for r in page_results:
            try:
                pdf_cost_total_usd += float(r.get("total_request_cost_usd", 0.0) or 0.0)
            except (TypeError, ValueError):
                pass

        notices = [r.get("engine_notice") for r in page_results if r.get("engine_notice")]
        return OrderedDict([
            ("source_pdf", pdf_filename),
            ("page_count", len(page_files)),
            ("host", RESPONSE_HOST_NAME),
            ("pages", page_results),
            ("total_request_cost_usd", pdf_cost_total_usd),
            ("engine_notice", notices[0] if notices else None),
        ]), 200
