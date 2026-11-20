"""Shared fixtures for the backend tests.

Every test runs with VVL_HOME, VVL_PRIVATE_PATH, VVL_SETTINGS_PATH, and
VVL_PROMPTS_USER_DIR pointed into its own tmp_path (autouse), so no test can
create or change the real PRIVATE.yaml, settings.local.yaml, or prompts/user.
"""

from __future__ import annotations

import http.server
import json
import socket
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures"
FIXTURE_IMAGE = FIXTURES / "images" / "MICH_16205594_Poaceae_Jouvea_pilosa.jpg"
TEST_TOKEN = "test-session-token-0123456789abcdef"


@pytest.fixture(autouse=True)
def vvl_env(tmp_path, monkeypatch):
    home = tmp_path / "vvl_home"
    env = {
        "VVL_HOME": str(home),
        "VVL_PRIVATE_PATH": str(tmp_path / "PRIVATE.yaml"),
        "VVL_SETTINGS_PATH": str(tmp_path / "settings.yaml"),
        "VVL_PROMPTS_USER_DIR": str(tmp_path / "prompts_user"),
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("VVL_TOKEN", raising=False)
    monkeypatch.delenv("VVL_OWNER_PID", raising=False)
    return {key: Path(value) for key, value in env.items()}


# A modern glibc, so engine-build choices never depend on the test machine's C
# library (the lock's Linux CUDA builds need glibc 2.38). Tests about glibc pass
# an explicit value to vvl.llama.variants.assess or pin a different one.
HOST_GLIBC_FOR_TESTS = "2.39"


@pytest.fixture(autouse=True)
def pinned_host_glibc(monkeypatch):
    """Returns the real vvl.llama.variants.host_glibc, for the test of the seam itself."""
    from vvl.llama import variants

    real = variants.host_glibc
    monkeypatch.setattr(variants, "host_glibc", lambda: HOST_GLIBC_FOR_TESTS)
    return real


class ElevationSpy:
    """Stands in for the VVGO elevation lookup so tests never reach the network."""

    def __init__(self, value=None):
        self.value = value
        self.calls = []

    def __call__(self, record):
        self.calls.append(record)
        return self.value


@pytest.fixture
def elevation_spy():
    return ElevationSpy()


@pytest.fixture
def make_app(vvl_env, elevation_spy):
    import app as app_module

    def _make(**config):
        flask_app = app_module.create_app(token=TEST_TOKEN, config={"TESTING": True, **config})
        flask_app.extensions["vvl"]["processor"].elevation_lookup = elevation_spy
        return flask_app

    return _make


@pytest.fixture
def flask_app(make_app):
    return make_app()


@pytest.fixture
def client(flask_app):
    return flask_app.test_client()


@pytest.fixture
def auth():
    return {"X-VVL-Token": TEST_TOKEN}


# ------------------------------------------------------------ no network --

_LOOPBACK_NAMES = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


@pytest.fixture
def no_network(monkeypatch):
    """Refuse any connection or DNS lookup that is not loopback."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def _check(address):
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and host not in _LOOPBACK_NAMES and not host.startswith("127."):
            raise OSError(f"network access is disabled in tests (tried {host})")

    def guarded_connect(self, address):
        _check(address)
        return real_connect(self, address)

    def guarded_connect_ex(self, address):
        _check(address)
        return real_connect_ex(self, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if isinstance(host, (bytes, bytearray)):
            host = host.decode()
        if host not in _LOOPBACK_NAMES and not str(host).startswith("127."):
            raise socket.gaierror(f"DNS is disabled in tests (looked up {host})")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)


# ------------------------------------------------------------ fake server --

def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FileRoute:
    """A downloadable file for FakeServer.files, with optional faults.

    status        HTTP status to answer instead of the file (401, 403, 404, 500, ...)
    cut_after     close the connection after this many body bytes (once per entry
                  in the list; e.g. [1000, 5000] fails twice, then serves normally)
    range_support honor `Range: bytes=N-` (206); False answers 200 with the whole file
    redirect_to   answer 302 to this path on the same server
    chunk_delay   seconds to sleep between 64 KiB chunks (so a test can cancel mid-way)
    statuses      per-request statuses, used in order before falling back to `status`
    """

    def __init__(self, data: bytes, *, status: int = 200, cut_after=None, range_support: bool = True,
                 redirect_to: str | None = None, chunk_delay: float = 0.0, statuses=None):
        self.data = data
        self.status = status
        self.statuses = list(statuses or [])
        self.cut_after = list(cut_after or [])
        self.range_support = range_support
        self.redirect_to = redirect_to
        self.chunk_delay = chunk_delay
        self.hits = 0


class FakeServer:
    """A tiny HTTP server on 127.0.0.1. `routes` maps path -> (status, json_body)
    (or a list of them, answered in order, the last repeating); `bodies` keeps
    the request bodies; `files` maps path -> FileRoute (Range requests,
    interruptions, redirects)."""

    def __init__(self):
        self.routes: dict[str, tuple[int, object]] = {}
        self.files: dict[str, FileRoute] = {}
        self.requests: list[dict] = []
        self.bodies: list[bytes] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _file(self, route: FileRoute):
                route.hits += 1
                if route.redirect_to:
                    self.send_response(302)
                    target = route.redirect_to
                    self.send_header("Location", target if target.startswith("http") else outer.base_url + target)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                status = route.statuses.pop(0) if route.statuses else route.status
                if status != 200:
                    body = b'{"error": "refused"}'
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                data, start = route.data, 0
                header = self.headers.get("Range")
                if header and route.range_support and header.startswith("bytes="):
                    start = int(header[6:].split("-", 1)[0])
                    if start >= len(data):
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{len(data)}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    self.send_response(200)
                body = data[start:]
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                limit = route.cut_after.pop(0) if route.cut_after else None
                sent = 0
                step = 64 * 1024
                while sent < len(body):
                    piece = body[sent:sent + step]
                    if limit is not None and sent + len(piece) > limit:
                        piece = piece[:max(0, limit - sent)]
                        self.wfile.write(piece)
                        self.wfile.flush()
                        self.close_connection = True
                        self.connection.shutdown(socket.SHUT_RDWR)
                        return
                    self.wfile.write(piece)
                    sent += len(piece)
                    if route.chunk_delay:
                        import time as _time

                        _time.sleep(route.chunk_delay)

            def _answer(self):
                path, _, query = self.path.partition("?")
                outer.requests.append({"method": self.command, "path": path, "query": query,
                                       "headers": {k.lower(): v for k, v in self.headers.items()}})
                length = int(self.headers.get("Content-Length") or 0)
                raw_body = self.rfile.read(length) if length else b""
                if path in outer.files:
                    try:
                        self._file(outer.files[path])
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                entry = outer.routes.get(path, (404, {"error": "not found"}))
                if isinstance(entry, list):  # a sequence of answers; the last one repeats
                    entry = entry.pop(0) if len(entry) > 1 else entry[0]
                outer.bodies.append(raw_body)
                status, body = entry
                data = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = _answer
            do_POST = _answer

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def fake_server():
    server = FakeServer().start()
    yield server
    server.stop()
