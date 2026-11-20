"""Source-image fetching for /process-url, borrowed from VoucherVisionGO's app.py.

The URL safety rules live in vvl.fetch.guarded_fetch; these adapters map them
onto the HTTP error contract. Both downloaders (the browser-like primary in
anti_bot_fetch and the plain `requests` fallback here) validate every redirect
hop with the same `validate_redirect_target`, so a destination refused by one
can never be fetched by the other.
"""

from __future__ import annotations

import logging
import tempfile
from urllib.parse import urljoin, urlparse

import requests
from PIL import Image
from werkzeug.datastructures import FileStorage

import vvl
from vvl.fetch.anti_bot_fetch import smart_fetch_image_as_filestorage
from vvl.fetch.guarded_fetch import (
    BlockedDestination,
    FetchDeadlineExceeded,
    InvalidFetchUrl,
    resolve_public_addresses,
    validate_url_shape,
)
from vvl.fetch.url_name_parser import resolve_filename_from_response
from vvl.images import DEFAULT_MAX_PIXELS, encode_resized

logger = logging.getLogger(__name__)

# Generic, identifies the software only (no user or institution details).
APP_USER_AGENT = f"VoucherVisionLocal/{vvl.__version__} python-requests/{requests.__version__}"

# Exception type names that mean "the caller's URL is malformed".
URL_INVALID_EXCEPTIONS = frozenset({
    "InvalidURL", "UnsupportedProtocol", "LocalProtocolError",
    "MissingSchema", "InvalidSchema", "URLRequired", "LocationParseError",
})

URL_FETCH_MAX_REDIRECTS = 5
# Spool threshold: images under this stay in memory, larger ones spill to disk.
SPOOL_MAX_IN_MEMORY_BYTES = 8 * 1024 * 1024
URL_FETCH_READ_TIMEOUT = 60.0


class TerminalFetchError(Exception):
    """A fetch failure that must not be retried or fall back to another method."""

    def __init__(self, error_code, error, status, cause=None):
        super().__init__(error)
        self.error_code = error_code
        self.error = error
        self.status = status
        self.cause = cause


# Fetch failures that are settled: retrying, or trying the other downloader,
# cannot change the outcome.
TERMINAL_FETCH_FAILURES = (
    BlockedDestination, InvalidFetchUrl, FetchDeadlineExceeded, TerminalFetchError,
)


def _exception_chain(exc):
    """The exception and everything it was raised from, most specific first."""
    seen, chain = set(), []
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        chain.append(exc)
        exc = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    return chain


def check_url_destination(url):
    """Safety check for one URL. Returns a TerminalFetchError or None.

    Rejects malformed URLs, non-web ports, embedded credentials, and any host
    that resolves to a non-public address (loopback, private, link-local, or a
    cloud metadata endpoint).
    """
    try:
        validate_url_shape(url)
        resolve_public_addresses(urlparse(url).hostname)
    except InvalidFetchUrl as e:
        return TerminalFetchError("URL_INVALID", e.message, 400, cause=e)
    except BlockedDestination as e:
        return TerminalFetchError("URL_BLOCKED", e.message, 403, cause=e)
    except FetchDeadlineExceeded as e:
        return TerminalFetchError(
            "UPSTREAM_TIMEOUT", "Timed out resolving the host of the supplied image URL.", 504, cause=e)
    return None


def validate_redirect_target(url):
    """Validate one redirect destination, raising its own verdict."""
    rejected = check_url_destination(url)
    if rejected is not None:
        raise rejected
    return None


def classify_fetch_failure(exc):
    """Map a source-image fetch failure to (error_code, http_status, message).

    Never includes library exception text in the message; the raw exception is
    logged against the request ID instead.
    """
    if isinstance(exc, TerminalFetchError):
        return exc.error_code, exc.status, exc.error

    for e in _exception_chain(exc):
        if isinstance(e, BlockedDestination):
            return ("URL_BLOCKED", 403, e.message)
        if isinstance(e, InvalidFetchUrl):
            return ("URL_INVALID", 400, e.message)
        if isinstance(e, FetchDeadlineExceeded):
            return ("UPSTREAM_TIMEOUT", 504, "Timed out downloading the image from the supplied URL.")

    names = [type(e).__name__ for e in _exception_chain(exc)]
    # Order matters: smart_fetch wraps everything in a RuntimeError, so
    # RuntimeError is the least specific signal and is tested last.
    if any("Timeout" in n for n in names):
        return ("UPSTREAM_TIMEOUT", 504, "Timed out downloading the image from the supplied URL.")
    if any(n in URL_INVALID_EXCEPTIONS for n in names):
        return ("URL_INVALID", 400, "The supplied image URL is not a valid URL.")
    if any(n in ("UnidentifiedImageError", "DecompressionBombError", "DecompressionBombWarning") for n in names):
        return ("INVALID_INPUT", 415, "The supplied URL did not return a readable image.")
    if any(n in ("ConnectError", "ReadError", "HTTPStatusError", "ConnectionError",
                 "RemoteProtocolError", "TooManyRedirects", "ProtocolError",
                 "RequestException", "HTTPError", "SSLError") for n in names):
        return ("FETCH_FAILED", 502, "The image could not be retrieved from the supplied URL.")
    if "RuntimeError" in names:
        return ("FETCH_FAILED", 502, "The image could not be retrieved from the supplied URL.")
    return ("INTERNAL_ERROR", 500, "An unexpected error occurred while retrieving the image.")


def process_url_image_with_resize(image_url, max_pixels=DEFAULT_MAX_PIXELS):
    """Fallback downloader: plain `requests`, redirects walked and validated by hand.

    Returns (FileStorage, filename). The FileStorage may wrap a spooled temp
    file; the caller closes it.
    """
    response = None
    current_url = image_url
    with requests.Session() as session:
        session.headers.update({"User-Agent": APP_USER_AGENT})
        for _hop in range(URL_FETCH_MAX_REDIRECTS + 1):
            response = session.get(current_url, stream=True, timeout=URL_FETCH_READ_TIMEOUT, allow_redirects=False)
            if response.status_code not in (301, 302, 303, 307, 308):
                break
            location = response.headers.get("location")
            response.close()
            if not location:
                break
            current_url = urljoin(current_url, location)
            validate_redirect_target(current_url)
        else:
            raise BlockedDestination("Too many redirects while fetching the supplied image URL.", url=image_url)

        response.raise_for_status()

        spool = tempfile.SpooledTemporaryFile(max_size=SPOOL_MAX_IN_MEMORY_BYTES, mode="w+b")
        try:
            try:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        spool.write(chunk)
                spool.seek(0)
                content_type = response.headers.get("Content-Type", "image/jpeg")
                content_disposition = response.headers.get("Content-Disposition")
            finally:
                response.close()
        except BaseException:
            spool.close()
            raise

    filename = resolve_filename_from_response(
        image_url, final_url=current_url, content_disposition=content_disposition,
    )

    try:
        image = Image.open(spool)
        image.load()
    except BaseException:
        spool.close()
        raise

    width, height = image.size
    is_heic = bool(filename) and filename.lower().endswith((".heic", ".heif"))
    if width * height <= max_pixels and not is_heic:
        spool.seek(0)
        return FileStorage(stream=spool, filename=filename, content_type=content_type), filename

    spool.close()
    buffer, filename, out_type = encode_resized(image, filename, max_pixels)
    return FileStorage(stream=buffer, filename=filename, content_type=out_type), filename


def fetch_image(image_url, *, max_pixels=DEFAULT_MAX_PIXELS):
    """Primary downloader, then the fallback once. Returns (FileStorage, filename).

    Raises a TERMINAL_FETCH_FAILURES exception for settled verdicts, otherwise
    the last exception after both downloaders failed.
    """
    try:
        return smart_fetch_image_as_filestorage(
            image_url,
            max_pixels=max_pixels,
            max_retries=3,
            connect_timeout=15.0,
            read_timeout=90.0,
            per_try_base_delay=0.75,
            per_try_jitter=0.75,
            user_agent=APP_USER_AGENT,
            extra_headers={},
            destination_check=validate_redirect_target,
            terminal_exceptions=TERMINAL_FETCH_FAILURES,
            logger=logger,
        )
    except TERMINAL_FETCH_FAILURES:
        raise
    except Exception:
        logger.info("Primary image downloader failed; trying the fallback downloader.")
        return process_url_image_with_resize(image_url, max_pixels=max_pixels)


__all__ = [
    "APP_USER_AGENT", "TERMINAL_FETCH_FAILURES", "TerminalFetchError", "check_url_destination",
    "classify_fetch_failure", "fetch_image", "process_url_image_with_resize", "validate_redirect_target",
]
