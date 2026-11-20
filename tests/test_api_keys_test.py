"""POST /v1/keys/<id>/test and model-list refresh, against a local fake server."""

from __future__ import annotations

import pytest
import yaml

from conftest import free_port

pytestmark = pytest.mark.usefixtures("no_network")

FAKE_KEY = "sk-FAKE-local-test-key-abcd1234"


def write_registry(path, base_url, *, auth="bearer", test_path="/models", models_path="/models"):
    path.write_text(yaml.safe_dump({
        "version": 1,
        "providers": [{
            "id": "fakeai",
            "label": "Fake AI",
            "role": "model_provider",
            "kind": "openai_compat",
            "purpose": "A local fake used by the tests.",
            "sends_images": True,
            "optional": True,
            "base_url": base_url,
            "key_url": "https://example.org/keys",
            "docs_url": "https://example.org/docs",
            "fields": [{"name": "api_key", "label": "Fake key", "placeholder": "fake"}],
            "test": {"method": "GET", "url": "{base_url}" + test_path, "auth": auth, "ok_status": [200]},
            "models": {"source": "list_endpoint", "url": "{base_url}" + models_path, "auth": auth,
                       "id_path": "data[].id", "strip_prefix": "models/"},
            "enabled": True,
        }],
    }), encoding="utf-8")
    return path


@pytest.fixture
def fake_app(make_app, tmp_path, fake_server):
    registry = write_registry(tmp_path / "providers.yaml", fake_server.base_url)
    flask_app = make_app(VVL_PROVIDERS_PATH=str(registry))
    return flask_app


def save_key(client, auth, value=FAKE_KEY):
    response = client.put("/v1/keys/fakeai", headers=auth, json={"fields": {"api_key": value}})
    assert response.status_code == 200
    return response


def test_ok(fake_app, fake_server, auth):
    fake_server.routes["/models"] = (200, {"data": [{"id": "m1"}], "secret_echo": FAKE_KEY})
    client = fake_app.test_client()
    save_key(client, auth)
    response = client.post("/v1/keys/fakeai/test", headers=auth)
    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is True
    assert body["status"] == "ok"
    assert body["http_status"] == 200
    assert body["message"] == "Key accepted."
    assert body["tested_at"].endswith("Z")
    assert FAKE_KEY not in response.get_data(as_text=True)
    assert "secret_echo" not in response.get_data(as_text=True)
    sent = fake_server.requests[-1]
    assert sent["headers"]["authorization"] == f"Bearer {FAKE_KEY}"

    provider = next(p for p in client.get("/v1/providers", headers=auth).get_json()["providers"] if p["id"] == "fakeai")
    assert provider["last_test"]["status"] == "ok"
    # Saving a key clears the previous test result.
    save_key(client, auth, FAKE_KEY + "-new")
    provider = next(p for p in client.get("/v1/providers", headers=auth).get_json()["providers"] if p["id"] == "fakeai")
    assert provider["last_test"] is None


def test_invalid_key(fake_app, fake_server, auth):
    fake_server.routes["/models"] = (401, {"error": {"message": f"bad key {FAKE_KEY}"}})
    client = fake_app.test_client()
    save_key(client, auth)
    response = client.post("/v1/keys/fakeai/test", headers=auth)
    body = response.get_json()
    assert body["ok"] is False
    assert body["status"] == "invalid_key"
    assert body["http_status"] == 401
    assert body["message"] == "The service rejected this key."
    assert FAKE_KEY not in response.get_data(as_text=True)


def test_other_http_error(fake_app, fake_server, auth):
    fake_server.routes["/models"] = (500, {"error": "boom"})
    client = fake_app.test_client()
    save_key(client, auth)
    body = client.post("/v1/keys/fakeai/test", headers=auth).get_json()
    assert body["status"] == "error"
    assert body["http_status"] == 500


def test_connection_refused(make_app, tmp_path, auth, caplog):
    port = free_port()  # nothing listens here
    registry = write_registry(tmp_path / "providers.yaml", f"http://127.0.0.1:{port}", auth="query:key")
    client = make_app(VVL_PROVIDERS_PATH=str(registry)).test_client()
    save_key(client, auth)
    response = client.post("/v1/keys/fakeai/test", headers=auth)
    body = response.get_json()
    assert body["ok"] is False
    assert body["status"] == "network_error"
    assert body["http_status"] is None
    # The key travels in the query string here; it must not reach the log.
    assert FAKE_KEY not in caplog.text
    assert FAKE_KEY not in response.get_data(as_text=True)


def test_not_configured(fake_app, auth):
    body = fake_app.test_client().post("/v1/keys/fakeai/test", headers=auth).get_json()
    assert body["status"] == "not_configured"
    assert body["ok"] is False


def test_query_and_header_auth(make_app, tmp_path, fake_server, auth):
    fake_server.routes["/q"] = (200, {})
    registry = write_registry(tmp_path / "providers.yaml", fake_server.base_url, auth="query:key", test_path="/q")
    client = make_app(VVL_PROVIDERS_PATH=str(registry)).test_client()
    save_key(client, auth)
    assert client.post("/v1/keys/fakeai/test", headers=auth).get_json()["status"] == "ok"
    assert fake_server.requests[-1]["query"] == f"key={FAKE_KEY}"

    fake_server.routes["/h"] = (200, {})
    registry = write_registry(tmp_path / "providers2.yaml", fake_server.base_url, auth="header:x-api-key", test_path="/h")
    client = make_app(VVL_PROVIDERS_PATH=str(registry)).test_client()
    save_key(client, auth)
    assert client.post("/v1/keys/fakeai/test", headers=auth).get_json()["status"] == "ok"
    assert fake_server.requests[-1]["headers"]["x-api-key"] == FAKE_KEY


def test_unknown_provider(fake_app, auth):
    assert fake_app.test_client().post("/v1/keys/nope/test", headers=auth).status_code == 404


def test_models_refresh(fake_app, fake_server, auth):
    fake_server.routes["/models"] = (200, {"data": [{"id": "models/alpha"}, {"id": "beta"}, {"id": "beta"}]})
    client = fake_app.test_client()

    entry = client.post("/v1/providers/fakeai/models/refresh", headers=auth).get_json()
    assert entry["models_state"] == "no_key"
    assert fake_server.requests == []

    save_key(client, auth)
    response = client.post("/v1/providers/fakeai/models/refresh", headers=auth)
    entry = response.get_json()
    assert entry["models_state"] == "loaded"
    assert [m["model_id"] for m in entry["models"]] == ["api:fakeai:alpha", "api:fakeai:beta"]
    assert entry["runnable"] is False
    assert FAKE_KEY not in response.get_data(as_text=True)

    catalog = client.get("/v1/models", headers=auth).get_json()
    api = {p["id"]: p for p in catalog["api"]["providers"]}
    assert api["fakeai"]["models_state"] == "loaded"
    assert [s["model_id"] for s in catalog["selectable"]["ocr"] if s["group"] == "API"] == [
        "api:fakeai:alpha", "api:fakeai:beta"]

    # Selecting a listed model works; an unlisted one is rejected.
    assert client.put("/v1/selection", headers=auth, json={"parse": "api:fakeai:beta"}).status_code == 200
    assert client.put("/v1/selection", headers=auth, json={"parse": "api:fakeai:gamma"}).status_code == 400

    fake_server.routes["/models"] = (500, {})
    entry = client.post("/v1/providers/fakeai/models/refresh", headers=auth).get_json()
    assert entry["models_state"] == "error"
    assert entry["models_error"]
    assert entry["models"] == []


def test_refresh_rejected_for_non_model_provider(client, auth):
    response = client.post("/v1/providers/vvgo/models/refresh", headers=auth)
    assert response.status_code == 400
    assert response.get_json()["error_code"] == "INVALID_INPUT"
