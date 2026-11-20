"""VoucherVisionLocal server: the Flask app object `app`.

Started from VoucherVisionGO's app.py (commit 2658891, see UPSTREAM.md) and cut
down to a local, single-user server. Kept, following VoucherVisionGO closely:
the /process and /process-url request handling, image resizing and HEIF, the
guarded URL fetchers, TextCollage pre-processing, the synchronous PDF path, the
response JSON (same field names and order), the error envelope, /prompts,
/health, and /api-costs. Everything that existed only for the hosted service
(Firebase/Firestore/GCS, accounts, quotas, billing, analytics, email, async
PDF jobs, CORS, Gemini credential passing) is gone.

The pipeline lives in vvl/pipeline.py and the model calls in vvl/engines; the
/v1 API for the desktop UI lives in vvl/api; the managed llama.cpp engine, model
downloads, and background jobs live in vvl/llama, vvl/local_models.py, and
vvl/jobs.py, held per app by vvl.runtime.Runtime. Run with `uv run vvl serve`.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import uuid
from pathlib import Path

from flask import Blueprint, Flask, current_app, g, jsonify, make_response, request, send_from_directory
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

import vvl
from vvl import paths, prompts, security, settings
from vvl.api import register_blueprints
from vvl.engines import (
    ALLOWED_THINKING_LEVELS,
    DEFAULT_THINKING_LEVEL,
    ModelIdError,
    ModelNotAvailable,
    ResolveContext,
    resolve_engine,
)
from vvl.errors import ApiError, _error_payload, _error_response, current_request_id  # noqa: F401  (_error_payload: contract name)
from vvl.fetch.policy import TERMINAL_FETCH_FAILURES, check_url_destination, classify_fetch_failure, fetch_image
from vvl.images import process_uploaded_file_with_resize
from vvl.llama.supervisor import SupervisorError
from vvl.local_models import StoreError
from vvl.pipeline import VoucherVisionProcessor
from vvl.providers import ProviderState
from vvl.runtime import Runtime
from vvl.registry.loader import RegistryError, load_api_costs, load_local_models, load_providers
from vvl.secrets.store import SecretsFileError, SecretStore, SecretValueError
from vvl.settings import MAX_OCR_ENGINES, SettingsError

logger = logging.getLogger("vvl.app")


class OrderedJsonEncoder(json.JSONEncoder):
    """Keeps the result envelope's key order (VoucherVisionGO)."""

    def __init__(self, **kwargs):
        kwargs["sort_keys"] = False
        super().__init__(**kwargs)


def _json_response(payload, status):
    response = make_response(json.dumps(payload, cls=OrderedJsonEncoder), status)
    response.headers["Content-Type"] = "application/json"
    return response


# ------------------------------------------------------- request parsing --

def resolve_thinking_levels(ocr_thinking_level=None, llm_thinking_level=None):
    """Return validated, normalized OCR and parsing thinking levels."""
    resolved = []
    for field_name, raw_value in (
        ("ocr_thinking_level", ocr_thinking_level),
        ("llm_thinking_level", llm_thinking_level),
    ):
        level = DEFAULT_THINKING_LEVEL if raw_value in (None, "") else str(raw_value).strip().lower()
        if level not in ALLOWED_THINKING_LEVELS:
            allowed = ", ".join(ALLOWED_THINKING_LEVELS)
            raise ValueError(f"Invalid {field_name} '{raw_value}'. Allowed values: {allowed}.")
        resolved.append(level)
    return tuple(resolved)


def _as_bool(value, default):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def _model_list(values):
    """`engines` may repeat or be comma-separated; order kept, duplicates dropped."""
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]
    out = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part and part not in out:
                out.append(part)
    return out


def app_config(key):
    return current_app.config.get(key)


def _settings_file():
    override = app_config("VVL_SETTINGS_PATH")
    return Path(override) if override else paths.settings_path()


def resolve_models(engine_options, llm_model_name, *, ocr_only=False, notebook_mode=False):
    """Apply the saved selection as defaults and resolve engines at the request edge.

    Raises ModelIdError (400) or ModelNotAvailable (409) before any work is done.
    Returns (ocr_engines, parse_engine_or_None, current_settings).
    """
    current = settings.load(_settings_file())
    ocr_ids = engine_options or list(current["selection"]["ocr"])
    if len(ocr_ids) > MAX_OCR_ENGINES:
        raise ApiError("INVALID_INPUT", f"At most {MAX_OCR_ENGINES} OCR models per request.", status=400)
    parse_id = (llm_model_name or "").strip() or current["selection"]["parse"]
    context = ResolveContext(
        providers=load_providers(app_config("VVL_PROVIDERS_PATH")),
        local_models=load_local_models(app_config("VVL_LOCAL_MODELS_PATH")),
        secrets=SecretStore(app_config("VVL_PRIVATE_PATH")),
        settings=current,
        runtime=current_app.extensions["vvl"]["runtime"],
    )
    ocr_engines = [resolve_engine(model_id, context) for model_id in ocr_ids]
    parse_engine = None
    if not (ocr_only or notebook_mode):
        parse_engine = resolve_engine(parse_id, context)
    return ocr_engines, parse_engine, current


def _processor() -> VoucherVisionProcessor:
    return current_app.extensions["vvl"]["processor"]


def _prepare(params):
    """Shared /process and /process-url preflight: validate, resolve models and the
    prompt. Returns the keyword arguments for the processor (url_source empty)."""
    try:
        ocr_thinking_level, llm_thinking_level = resolve_thinking_levels(
            params["ocr_thinking_level"], params["llm_thinking_level"])
    except ValueError as exc:
        raise ApiError("INVALID_INPUT", str(exc), status=400,
                       details={"allowed_thinking_levels": list(ALLOWED_THINKING_LEVELS)}) from None

    ocr_only, notebook_mode = params["ocr_only"], params["notebook_mode"]
    ocr_engines, parse_engine, current = resolve_models(
        params["engines"], params["llm_model"], ocr_only=ocr_only, notebook_mode=notebook_mode)

    prompt_name, prompt_path = params["prompt"], None
    if prompt_name or not (ocr_only or notebook_mode):
        prompt_path, _source = prompts.resolve(prompt_name)
        prompt_name = prompt_path.name

    return dict(
        ocr_engines=ocr_engines,
        parse_engine=parse_engine,
        prompt=prompt_name,
        prompt_path=prompt_path,
        ocr_only=ocr_only,
        include_wfo=params["include_wfo"],
        include_cop90=params["include_cop90"],
        url_source="",
        notebook_mode=notebook_mode,
        skip_label_collage=params["skip_label_collage"],
        ocr_thinking_level=ocr_thinking_level,
        llm_thinking_level=llm_thinking_level,
        elevation_enabled=bool(current["enrichment"]["elevation"]),
        stage=_stage(params.get("stage")),
    )


def _stage(value):
    """`stage` of a two-pass batch request: None (one pass) or "ocr" (first pass)."""
    if value in (None, ""):
        return None
    if value == "ocr":
        return "ocr"
    raise ApiError("INVALID_INPUT", "stage must be \"ocr\" or left out.", status=400)


def _params_from_form(form):
    return {
        "stage": form.get("stage") or None,
        "engines": _model_list(form.getlist("engines")) if "engines" in form else None,
        "prompt": form.get("prompt") or None,
        "ocr_only": _as_bool(form.get("ocr_only"), False),
        "notebook_mode": _as_bool(form.get("notebook_mode"), False),
        "skip_label_collage": _as_bool(form.get("skip_label_collage"), False),
        "include_wfo": _as_bool(form.get("include_wfo"), True),
        "include_cop90": _as_bool(form.get("include_cop90"), True),
        "llm_model": form.get("llm_model") or None,
        "ocr_thinking_level": form.get("ocr_thinking_level"),
        "llm_thinking_level": form.get("llm_thinking_level"),
    }


def _params_from_json(data):
    return {
        "stage": data.get("stage") or None,
        "engines": _model_list(data.get("engines")),
        "prompt": data.get("prompt") or None,
        "ocr_only": _as_bool(data.get("ocr_only"), False),
        "notebook_mode": _as_bool(data.get("notebook_mode"), False),
        "skip_label_collage": _as_bool(data.get("skip_label_collage"), False),
        "include_wfo": _as_bool(data.get("include_wfo"), True),
        "include_cop90": _as_bool(data.get("include_cop90"), True),
        "llm_model": data.get("llm_model") or None,
        "ocr_thinking_level": data.get("ocr_thinking_level"),
        "llm_thinking_level": data.get("llm_thinking_level"),
    }


# ----------------------------------------------------------------- routes --

core = Blueprint("core", __name__)


@core.get("/health")
def health_check():
    """Public. Server status for the desktop shell's startup wait."""
    throttler = _processor().throttler
    active_requests = throttler.get_active_count()
    max_requests = throttler.max_concurrent
    return jsonify({
        "status": "ok",
        "app": vvl.APP_NAME,
        "version": vvl.__version__,
        "pid": os.getpid(),
        "active_requests": active_requests,
        "max_concurrent_requests": max_requests,
        "server_load": f"{(active_requests / max_requests) * 100:.1f}%",
        "api_status": "available",
    })


@core.post("/process")
def process_image():
    """Process an uploaded image (or PDF) with automatic resizing."""
    if "file" not in request.files:
        return _error_response("NO_FILE", "No file provided", status=400)
    file = request.files["file"]
    if not file.filename:
        return _error_response("NO_FILE", "No file selected", status=400)
    processor = _processor()
    if not processor.allowed_file(file.filename):
        return _error_response(
            "BAD_FILE_TYPE",
            f'File type not allowed. Supported types: {", ".join(sorted(processor.ALLOWED_EXTENSIONS))}',
            status=415, details={"allowed_extensions": sorted(processor.ALLOWED_EXTENSIONS)},
        )

    process_kwargs = _prepare(_params_from_form(request.form))

    if file.filename.lower().endswith(".pdf"):
        logger.info("PDF upload detected: %s", file.filename)
        process_kwargs["stage"] = None      # PDFs are always processed in one pass
        results, status_code = processor.process_pdf_request(file=file, **process_kwargs)
    else:
        try:
            file = process_uploaded_file_with_resize(file)
        except Exception:
            logger.exception("Image preparation failed (request_id=%s)", current_request_id())
            return _error_response(
                "INVALID_INPUT",
                "The uploaded image could not be read. It may be corrupt or an unsupported format.",
                status=400)
        results, status_code = processor.process_image_request(file=file, **process_kwargs)

    return _json_response(results, status_code)


@core.post("/v1/process/finish")
def process_finish():
    """Second pass of a two-pass batch: parse an image whose OCR result was saved
    by a `stage=ocr` request. Body (JSON or form): `stage_token`, plus the usual
    optional `llm_model`, `prompt`, `include_wfo`, `include_cop90`,
    `llm_thinking_level`. Returns what a one-pass /process request returns."""
    if request.is_json:
        data = request.get_json(silent=True)
        data = data if isinstance(data, dict) else {}
        params = _params_from_json(data)
    else:
        data = request.form
        params = _params_from_form(request.form)
    token = data.get("stage_token")
    if not isinstance(token, str) or not token.strip():
        return _error_response("INVALID_INPUT", "Send the stage_token returned by the OCR pass.", status=400)
    params["stage"] = None
    params["ocr_only"] = params["notebook_mode"] = False
    process_kwargs = _prepare(params)
    process_kwargs["stage"] = "parse"
    results, status_code = _processor().process_image_request(file=None, stage_token=token.strip(),
                                                              **process_kwargs)
    return _json_response(results, status_code)


@core.post("/process-url")
def process_image_by_url():
    """Process an image fetched from a URL (JSON or form body)."""
    if request.is_json:
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not data.get("image_url"):
            return _error_response("INVALID_INPUT", "No image URL provided", status=400)
        image_url = str(data.get("image_url")).strip()
        params = _params_from_json(data)
    else:
        if not request.form.get("image_url"):
            return _error_response("INVALID_INPUT", "No image URL provided", status=400)
        image_url = request.form.get("image_url").strip()
        params = _params_from_form(request.form)

    process_kwargs = _prepare(params)
    process_kwargs["url_source"] = image_url

    # Terminal before any fetch: a malformed or refused URL never downloads.
    url_error = check_url_destination(image_url)
    if url_error is not None:
        logger.warning("Rejected image URL (request_id=%s): %s", current_request_id(), url_error.error_code)
        return _error_response(url_error.error_code, url_error.error, status=url_error.status,
                               details={"url_source": image_url})

    file_obj = None
    try:
        try:
            file_obj, _filename = fetch_image(image_url)
        except TERMINAL_FETCH_FAILURES as e:
            code, status, message = classify_fetch_failure(e)
            logger.warning("Terminal fetch failure (request_id=%s): %s -> %s",
                           current_request_id(), type(e).__name__, code)
            return _error_response(code, message, status=status, details={"url_source": image_url})
        except Exception as e:
            code, status, message = classify_fetch_failure(e)
            logger.error("URL fetch failed (request_id=%s) -> %s %s (%s)",
                         current_request_id(), code, status, type(e).__name__)
            return _error_response(code, message, status=status, details={"url_source": image_url})

        results, status_code = _processor().process_image_request(file=file_obj, **process_kwargs)
        return _json_response(results, status_code)
    finally:
        # The downloader may hand back a FileStorage wrapping a spooled temp file.
        try:
            close = getattr(file_obj, "close", None)
            if callable(close):
                close()
        except Exception:
            logger.debug("Could not close the fetched image stream", exc_info=True)


@core.get("/prompts")
def list_prompts_api():
    """List prompt templates (built-in + user) in VoucherVisionGO's shape."""
    format_type = request.args.get("format", "json")
    view_details = request.args.get("view", "false").lower() == "true"
    specific_prompt = request.args.get("prompt")

    entries = [(paths.prompts_builtin_dir() / n, "builtin") for n in prompts.builtin_names()]
    entries += [(paths.prompts_user_dir() / n, "user") for n in prompts.user_names()]

    if specific_prompt:
        match = next(((p, s) for p, s in entries if p.name == specific_prompt), None)
        if match is None:
            available = [p.name for p, _ in entries]
            if format_type == "text":
                return (f"Prompt '{specific_prompt}' not found.\nAvailable prompts: {', '.join(available)}",
                        404, {"Content-Type": "text/plain"})
            return _error_response("PROMPT_NOT_FOUND", f"Prompt '{specific_prompt}' not found.", status=404,
                                   details={"available_prompts": available})
        path, source = match
        prompt_details = prompts.extract_prompt_details(path)
        if format_type == "text":
            return prompts.format_prompt_as_text(path.name, prompt_details), 200, {"Content-Type": "text/plain"}
        return jsonify({
            "status": "success",
            "prompt": {
                "filename": path.name,
                "prompt_ref": path.name,
                "source": source,
                "is_user_generated": source == "user",
                "details": prompt_details,
            },
        })

    prompt_info_list = []
    for path, source in entries:
        info = prompts.extract_prompt_info(path)
        info["source"] = source
        info["is_user_generated"] = source == "user"
        info["prompt_ref"] = info.get("filename")
        if view_details:
            info["details"] = prompts.extract_prompt_details(path)
        prompt_info_list.append(info)

    if not prompt_info_list:
        if format_type == "text":
            return "No prompt files found.", 404, {"Content-Type": "text/plain"}
        return _error_response("PROMPT_NOT_FOUND", "No prompt files found.", status=404)

    if format_type == "text":
        return prompts.format_prompts_as_text_table(prompt_info_list), 200, {"Content-Type": "text/plain"}
    return jsonify({"status": "success", "count": len(prompt_info_list), "prompts": prompt_info_list})


@core.get("/api-costs")
def api_costs():
    """The api_cost.yaml contents, for the cost calculator."""
    return jsonify(load_api_costs())


@core.get("/")
def ui_index():
    ui = paths.ui_dir()
    if not (ui / "index.html").is_file():
        return _error_response("NOT_FOUND", "The UI files are missing (ui/index.html).", status=404)
    response = send_from_directory(ui, "index.html", max_age=0)
    response.headers["Cache-Control"] = "no-store"
    return response


@core.get("/ui/<path:filename>")
def ui_file(filename):
    response = send_from_directory(paths.ui_dir(), filename, max_age=0)
    response.headers["Cache-Control"] = "no-store"
    return response


# ------------------------------------------------------------ app factory --

def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(ApiError)
    def _api_error(e):
        return e.to_response()

    @app.errorhandler(ModelIdError)
    def _model_id_error(e):
        return _error_response("INVALID_INPUT", str(e), status=400, details={"model_id": e.model_id})

    @app.errorhandler(ModelNotAvailable)
    def _model_not_available(e):
        return _error_response(
            "MODEL_NOT_AVAILABLE", f"Model '{e.model_id}' is not available.", message=e.reason,
            status=409, details={"model_id": e.model_id, "unavailable_reason": e.reason},
        )

    @app.errorhandler(StoreError)
    def _store_error(e):
        return _error_response(e.code, e.message, status=e.status, details=e.details or None)

    @app.errorhandler(SupervisorError)
    def _supervisor_error(e):
        return _error_response(e.code, e.message, status=e.status, details=e.details or None)

    @app.errorhandler(SettingsError)
    def _settings_error(e):
        return _error_response("INVALID_INPUT", str(e), status=400, details=e.details or None)

    @app.errorhandler(SecretValueError)
    def _secret_value_error(e):
        return _error_response("INVALID_INPUT", str(e), status=400)

    @app.errorhandler(SecretsFileError)
    def _secrets_file_error(e):
        return _error_response("INTERNAL_ERROR", str(e), status=500)

    @app.errorhandler(RegistryError)
    def _registry_error(e):
        logger.error("Registry error: %s", e)
        return _error_response("INTERNAL_ERROR", "A registry file is invalid. See the server log.", status=500)

    @app.errorhandler(RequestEntityTooLarge)
    def _too_large(e):
        limit = app.config.get("MAX_CONTENT_LENGTH")
        return _error_response(
            "PAYLOAD_TOO_LARGE", f"The upload is larger than the {limit // (1024 * 1024)} MB limit.", status=413,
        )

    @app.errorhandler(HTTPException)
    def _http_error(e):
        code = {400: "INVALID_INPUT", 403: "FORBIDDEN", 404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED",
                409: "CONFLICT", 415: "BAD_FILE_TYPE", 501: "NOT_IMPLEMENTED"}.get(e.code, "INTERNAL_ERROR")
        return _error_response(code, e.name, message=e.description, status=e.code or 500)

    @app.errorhandler(Exception)
    def _unhandled(e):
        logger.exception("Unhandled error (request_id=%s)", current_request_id())
        return _error_response(
            "INTERNAL_ERROR", "The request could not be completed. Quote the request id when reporting this.",
            status=500,
        )


def create_app(*, token: str | None = None, config: dict | None = None) -> Flask:
    """Build the app. `token` defaults to $VVL_TOKEN, else a fresh random token."""
    flask_app = Flask(__name__, static_folder=None)
    flask_app.json.sort_keys = False
    flask_app.config["MAX_CONTENT_LENGTH"] = VoucherVisionProcessor.MAX_CONTENT_LENGTH
    flask_app.config["VVL_TOKEN"] = token or os.environ.get("VVL_TOKEN") or secrets.token_urlsafe(32)
    flask_app.config.update(config or {})

    processor = VoucherVisionProcessor()
    flask_app.extensions["vvl"] = {
        "processor": processor,
        "provider_state": ProviderState(),
        # Phase 2: jobs, local model store, engine installer and supervisor.
        "runtime": Runtime(flask_app.config),
    }

    @flask_app.before_request
    def begin_request():
        # First hook: every response, including 401/403, carries a request id.
        g.request_id = str(uuid.uuid4())

    security.install(flask_app)

    @flask_app.after_request
    def _add_correlation_headers(response):
        request_id = getattr(g, "request_id", None)
        if request_id:
            response.headers["X-Request-ID"] = request_id
        return response

    flask_app.register_blueprint(core)
    register_blueprints(flask_app)
    _register_error_handlers(flask_app)
    return flask_app


app = create_app()
