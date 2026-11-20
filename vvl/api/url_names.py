"""POST /v1/url-names: the file name VoucherVisionLocal gives an image URL.

Names come from `vvl.fetch.url_name_parser`, the same parser the downloader
uses, so the UI never has to guess a name from a URL itself. This route does
pure string parsing: it never contacts the URL's server. (The parser's
`extract_filename_from_url` may send a HEAD request, so its request-free core,
`resolve_filename_from_response`, is called with no response metadata.) Once an
image has been fetched, the `filename` in the /process-url result is the final
word, because it can also use the server's response headers.
"""

from __future__ import annotations

import os

from flask import Blueprint, jsonify

from vvl.api.common import json_body
from vvl.errors import ApiError
from vvl.fetch.url_name_parser import resolve_filename_from_response

bp = Blueprint("vvl_url_names", __name__)

MAX_URLS = 5000
MAX_URL_LENGTH = 4096


def name_for_url(url: str) -> dict:
    """{"url", "name", "stem"} for one URL. Never raises and never uses the network."""
    try:
        name = resolve_filename_from_response(url)
    except Exception:  # noqa: BLE001 - one odd URL must not fail the batch
        name = resolve_filename_from_response("")
    return {"url": url, "name": name, "stem": os.path.splitext(name)[0] or name}


@bp.post("/v1/url-names")
def url_names():
    body = json_body()
    unknown = sorted(set(body) - {"urls"})
    if unknown:
        raise ApiError("INVALID_INPUT", "Unknown fields.", status=400, details={"unknown_keys": unknown})
    urls = body.get("urls")
    if not isinstance(urls, list):
        raise ApiError("INVALID_INPUT", "Send urls as a list, e.g. {\"urls\": [\"https://…\"]}.", status=400)
    if len(urls) > MAX_URLS:
        raise ApiError("INVALID_INPUT", f"Send at most {MAX_URLS} URLs in one request.", status=400,
                       details={"max_urls": MAX_URLS, "count": len(urls)})
    for index, url in enumerate(urls):
        if not isinstance(url, str):
            raise ApiError("INVALID_INPUT", "Every entry in urls must be text.", status=400,
                           details={"index": index})
        if len(url) > MAX_URL_LENGTH:
            raise ApiError("INVALID_INPUT", f"A URL may be at most {MAX_URL_LENGTH} characters long.", status=400,
                           details={"index": index, "max_url_length": MAX_URL_LENGTH})
    return jsonify({"names": [name_for_url(url) for url in urls]})
