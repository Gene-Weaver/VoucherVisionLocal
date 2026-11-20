"""docs/API.md section 2: Host header, session token, CORS, token never echoed."""

from __future__ import annotations

import pytest

from conftest import FIXTURE_IMAGE, TEST_TOKEN

pytestmark = pytest.mark.usefixtures("no_network")

PROTECTED_GETS = [
    "/v1/app", "/v1/system", "/v1/providers", "/v1/models", "/v1/settings", "/v1/prompts",
    "/v1/prompts/builtin/SLTPvM_default.yaml", "/v1/engine", "/prompts", "/api-costs",
]


def test_health_is_public_and_identifies_the_app(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["app"] == "VoucherVisionLocal"
    assert body["version"]
    assert isinstance(body["pid"], int)


@pytest.mark.parametrize("path", PROTECTED_GETS)
def test_token_required(client, path):
    response = client.get(path)
    assert response.status_code == 401
    body = response.get_json()
    assert body["error_code"] == "UNAUTHENTICATED"
    assert body["host"] == "VoucherVisionLocal"
    assert body["request_id"]


def test_wrong_token_rejected(client):
    assert client.get("/v1/app", headers={"X-VVL-Token": "nope"}).status_code == 401
    assert client.get("/v1/app", headers={"X-VVL-Token": TEST_TOKEN + "x"}).status_code == 401
    assert client.get("/v1/app", headers={"Authorization": "Basic " + TEST_TOKEN}).status_code == 401


@pytest.mark.parametrize("headers", [
    {"X-VVL-Token": TEST_TOKEN},
    {"Authorization": f"Bearer {TEST_TOKEN}"},
    {"X-API-Key": TEST_TOKEN},
])
def test_token_header_aliases(client, headers):
    assert client.get("/v1/app", headers=headers).status_code == 200


def test_post_routes_need_token(client):
    assert client.post("/process").status_code == 401
    assert client.post("/process-url", json={"image_url": "https://example.org/x.jpg"}).status_code == 401
    assert client.post("/v1/shutdown").status_code == 401
    assert client.put("/v1/keys/gemini", json={"fields": {"api_key": "fake"}}).status_code == 401


def test_fails_closed_without_configured_token(make_app):
    flask_app = make_app()
    flask_app.config["VVL_TOKEN"] = None
    client = flask_app.test_client()
    assert client.get("/v1/app", headers={"X-VVL-Token": TEST_TOKEN}).status_code == 401
    assert client.get("/v1/app", headers={"X-VVL-Token": ""}).status_code == 401
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize("base_url", [
    "http://evil.example.com", "http://evil.example.com:8642", "http://127.0.0.1.evil.example.com",
    "http://localhost.evil:8642", "http://10.0.0.5:8642", "http://0.0.0.0:8642",
])
def test_bad_host_rejected(client, auth, base_url):
    for path in ("/health", "/v1/app", "/"):
        response = client.get(path, headers=auth, base_url=base_url)
        assert response.status_code == 403, (base_url, path)
        assert response.get_json()["error_code"] == "FORBIDDEN"


@pytest.mark.parametrize("base_url", [
    "http://127.0.0.1", "http://127.0.0.1:8642", "http://localhost:8651", "http://LOCALHOST", "http://[::1]:8642",
])
def test_loopback_hosts_allowed(client, auth, base_url):
    assert client.get("/v1/app", headers=auth, base_url=base_url).status_code == 200


def test_no_cors_headers(client, auth):
    origin = {"Origin": "https://evil.example.com"}
    responses = [
        client.get("/health", headers=origin),
        client.get("/v1/app", headers={**auth, **origin}),
        client.get("/v1/app", headers=origin),
        client.options("/process", headers={**origin, "Access-Control-Request-Method": "POST"}),
        client.options("/v1/settings", headers={**auth, **origin, "Access-Control-Request-Method": "PUT"}),
    ]
    for response in responses:
        assert not any(h.lower().startswith("access-control-") for h in response.headers.keys())


def test_token_never_in_any_response(client, auth):
    responses = [client.get("/health")]
    responses += [client.get(path, headers=auth) for path in PROTECTED_GETS]
    responses += [client.get(path) for path in PROTECTED_GETS]
    responses.append(client.get("/prompts?format=text", headers=auth))
    responses.append(client.put("/v1/settings", headers=auth, json={"ui": {"theme": "light"}}))
    responses.append(client.put("/v1/selection", headers=auth, json={"parse": "demo:mock"}))
    responses.append(client.put("/v1/keys/openai", headers=auth, json={"fields": {"api_key": "sk-fake-test-value-1234"}}))
    responses.append(client.post("/v1/engine/start", headers=auth))
    responses.append(client.get("/no-such-route", headers=auth))
    with open(FIXTURE_IMAGE, "rb") as handle:
        responses.append(client.post(
            "/process", headers=auth, content_type="multipart/form-data",
            data={"file": (handle, "specimen.jpg"), "prompt": "SLTPvM_skeletal.yaml"},
        ))
    for response in responses:
        assert TEST_TOKEN not in response.get_data(as_text=True)
        assert all(TEST_TOKEN not in value for value in response.headers.values())


def test_static_ui_is_public_and_not_cached(client, tmp_path, monkeypatch):
    ui = tmp_path / "ui"
    (ui / "js").mkdir(parents=True)
    (ui / "index.html").write_text("<!doctype html><title>VVL</title>", encoding="utf-8")
    (ui / "js" / "app.js").write_text("console.log('vvl');", encoding="utf-8")
    monkeypatch.setattr("vvl.paths.ui_dir", lambda: ui)

    index = client.get("/")
    assert index.status_code == 200
    assert b"<title>VVL</title>" in index.data
    assert index.headers["Cache-Control"] == "no-store"
    script = client.get("/ui/js/app.js")
    assert script.status_code == 200
    assert script.headers["Cache-Control"] == "no-store"
    assert client.get("/ui/../app.py").status_code == 404
    assert client.get("/ui/%2e%2e/app.py").status_code == 404
    assert client.get("/ui/missing.js").status_code == 404


def test_security_headers(client, auth):
    response = client.get("/v1/app", headers=auth)
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["X-Request-ID"]


def test_unknown_route_uses_envelope(client, auth):
    response = client.get("/admin", headers=auth)
    assert response.status_code == 404
    body = response.get_json()
    assert body["error_code"] == "NOT_FOUND"
    assert list(body)[:2] == ["error", "error_code"]
