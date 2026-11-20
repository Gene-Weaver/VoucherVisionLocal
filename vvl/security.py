"""Request gatekeeping (docs/API.md section 2).

  * Host header must be 127.0.0.1[:port], localhost[:port], or [::1][:port]
    (plus the explicit --host value when the user chose one): else 403.
  * Every route except the static UI and GET /health needs the session token
    in X-VVL-Token (aliases: Authorization: Bearer, X-API-Key): else 401.
    Compared with hmac.compare_digest; no token configured means every
    protected request is refused.
  * No Access-Control-* headers are ever sent.
"""

from __future__ import annotations

import hmac
import logging
import re

from flask import Flask, current_app, g, request

from vvl.errors import _error_response

logger = logging.getLogger(__name__)

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})
_HOST_RE = re.compile(r"^(?P<host>\[[0-9a-fA-F:.]+\]|[^:\[\]]+)(?::(?P<port>[0-9]{1,5}))?$")

PUBLIC_EXACT = frozenset({("GET", "/health"), ("HEAD", "/health"), ("GET", "/"), ("HEAD", "/")})
PUBLIC_PREFIXES = ("/ui/",)


def host_allowed(host_header: str | None, extra_hosts: frozenset[str] = frozenset()) -> bool:
    if not host_header:
        return False
    match = _HOST_RE.match(host_header.strip())
    if not match:
        return False
    host = match.group("host").lower()
    port = match.group("port")
    if port is not None and not (0 < int(port) < 65536):
        return False
    return host in LOOPBACK_HOSTS or host in extra_hosts


def is_public(method: str, path: str) -> bool:
    if (method, path) in PUBLIC_EXACT:
        return True
    return method in ("GET", "HEAD") and any(path.startswith(p) for p in PUBLIC_PREFIXES)


def presented_token() -> str | None:
    token = request.headers.get("X-VVL-Token")
    if token:
        return token.strip()
    auth = request.headers.get("Authorization", "")
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip() or None
    token = request.headers.get("X-API-Key")
    if token:
        return token.strip()
    return None


def token_matches(presented: str | None, expected: str | None) -> bool:
    if not expected or not presented:
        return False
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


def install(app: Flask) -> None:
    """Register the request gate and the response header policy on `app`."""

    @app.before_request
    def _vvl_gate():
        extra = frozenset(h.lower() for h in current_app.config.get("VVL_EXTRA_ALLOWED_HOSTS", ()))
        if not host_allowed(request.headers.get("Host"), extra):
            logger.warning("Rejected request with a non-local Host header.")
            return _error_response(
                "FORBIDDEN", "This server only answers requests addressed to 127.0.0.1 or localhost.",
                status=403,
            )
        if is_public(request.method, request.path):
            return None
        if not token_matches(presented_token(), current_app.config.get("VVL_TOKEN")):
            return _error_response(
                "UNAUTHENTICATED",
                "Missing or wrong session token. Open the app from the link printed by `vvl serve`.",
                status=401,
            )
        g.vvl_authenticated = True
        return None

    @app.after_request
    def _vvl_headers(response):
        for name in list(response.headers.keys()):
            if name.lower().startswith("access-control-"):
                del response.headers[name]
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers.setdefault("Cache-Control", "no-store")
        return response
