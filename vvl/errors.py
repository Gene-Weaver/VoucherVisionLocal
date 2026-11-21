"""The error envelope, borrowed from VoucherVisionGO's app.py.

`_error_payload()` keeps VoucherVisionGO's shape and key order:
    error, error_code, [message], host, [request_id], [details], [legacy fields]

Differences from VoucherVisionGO: `host` is "VoucherVisionLocal", there is no
`payment_inference` (nothing here is billed by the server), and the code
vocabulary drops the account/billing/quota codes and adds the codes the VVL
contract needs (MODEL_NOT_AVAILABLE, NOT_IMPLEMENTED, NOT_FOUND, ...).
"""

from __future__ import annotations

import logging
import uuid
from collections import OrderedDict

from flask import g, jsonify, make_response

logger = logging.getLogger(__name__)

RESPONSE_HOST_NAME = "VoucherVisionLocal"

# Closed vocabulary. `error_code` is the machine-readable discriminator; the
# `error` string is for display.
ERROR_CODES = frozenset({
    # request shape
    "NO_FILE", "BAD_FILE_TYPE", "INVALID_INPUT", "PAYLOAD_TOO_LARGE",
    "NOT_FOUND", "METHOD_NOT_ALLOWED", "CONFLICT",
    # identity and access
    "UNAUTHENTICATED", "FORBIDDEN", "PROMPT_NOT_FOUND",
    # capacity
    "SERVER_BUSY",
    # source image retrieval
    "URL_INVALID", "URL_BLOCKED", "FETCH_FAILED",
    # models and engines
    "MODEL_NOT_AVAILABLE", "MODEL_UNAVAILABLE", "NOT_IMPLEMENTED",
    "UPSTREAM_AUTH_FAILED", "UPSTREAM_QUOTA_EXHAUSTED",
    "UPSTREAM_UNAVAILABLE", "UPSTREAM_TIMEOUT",
    # Phase 2: the managed engine, downloads, and installs
    "ENGINE_ERROR", "ENGINE_NOT_READY", "ENGINE_NOT_INSTALLED", "ENGINE_SELF_TEST_FAILED",
    "INSUFFICIENT_DISK_SPACE", "GATED_MODEL", "CHECKSUM_MISMATCH", "SIZE_MISMATCH",
    "NETWORK_ERROR", "HTTP_ERROR", "EXTRACT_FAILED", "DOWNLOAD_FAILED",
    # two-pass batches
    "STAGE_NOT_FOUND",
    # Saved results (vvl/outputs.py).
    "OUTPUT_DIR_UNAVAILABLE",
    "OUTPUT_BATCH_NOT_FOUND",
    # anything else
    "INTERNAL_ERROR",
})

# Keys the envelope owns; a `legacy` mapping may never overwrite them.
_ERROR_RESERVED_KEYS = frozenset({
    "error", "error_code", "message", "host", "request_id", "details",
})


def current_request_id() -> str:
    """The correlation ID assigned at the start of the request."""
    try:
        existing = getattr(g, "request_id", None)
        if existing:
            return existing
        g.request_id = str(uuid.uuid4())
        return g.request_id
    except Exception:
        return str(uuid.uuid4())


def _error_payload(error_code, error, *, message=None, details=None, legacy=None):
    """Build an error body as a plain dict (not a Flask response).

    The processing helpers return `(dict, status)` to their callers; routes
    want `_error_response()` instead.
    """
    if error_code not in ERROR_CODES:
        # Never fail a request over a bad label: log it and degrade.
        logger.error("Unknown error_code %r; falling back to INTERNAL_ERROR", error_code)
        error_code = "INTERNAL_ERROR"

    payload = OrderedDict()
    payload["error"] = error
    payload["error_code"] = error_code
    if message:
        payload["message"] = message
    payload["host"] = RESPONSE_HOST_NAME

    try:
        request_id = getattr(g, "request_id", None)
    except Exception:
        request_id = None
    if request_id:
        payload["request_id"] = request_id

    if details:
        payload["details"] = details

    for key, value in (legacy or {}).items():
        if key in _ERROR_RESERVED_KEYS:
            logger.warning("Ignoring legacy error field %r: reserved by the envelope", key)
            continue
        payload[key] = value

    return payload


def _error_response(error_code, error, *, message=None, status=500, details=None, legacy=None):
    """Flask response form of `_error_payload`, for use directly in routes."""
    payload = _error_payload(error_code, error, message=message, details=details, legacy=legacy)
    return make_response(jsonify(payload), status)


def _pdf_page_error(filename, error_code, error, status_code, **kwargs):
    """A failed PDF page entry, carrying the same envelope as a top-level error."""
    entry = OrderedDict()
    entry["filename"] = filename
    entry.update(_error_payload(error_code, error, **kwargs))
    entry["status_code"] = status_code
    return entry


class ApiError(Exception):
    """Raised inside /v1 handlers; rendered with the envelope by app.py."""

    def __init__(self, error_code, error, *, status=400, message=None, details=None):
        super().__init__(error)
        self.error_code = error_code
        self.error = error
        self.status = status
        self.message = message
        self.details = details

    def to_response(self):
        return _error_response(
            self.error_code, self.error, message=self.message,
            status=self.status, details=self.details,
        )
