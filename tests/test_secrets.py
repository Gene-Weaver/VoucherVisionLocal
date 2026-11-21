"""PRIVATE.yaml store: round trip, masking, atomic writes, values never leak."""

from __future__ import annotations

import logging
import os
import stat
import sys

import pytest
import yaml

from vvl.secrets.store import SecretsFileError, SecretStore, SecretValueError, mask

pytestmark = pytest.mark.usefixtures("no_network")

FAKE_GEMINI = "AIzaFAKE-test-key-not-real-000a1b2"
FAKE_OPENAI = "sk-FAKE-test-openai-key-9999wxyz"


def test_mask():
    assert mask(None) is None
    assert mask("") is None
    assert mask("short") == "••••"
    assert mask("12345678901") == "••••"
    assert mask("123456789012") == "••••9012"
    assert mask(FAKE_GEMINI) == "••••" + FAKE_GEMINI[-4:]


def test_round_trip(vvl_env):
    store = SecretStore()
    assert store.path == vvl_env["VVL_PRIVATE_PATH"]
    assert store.get_value("gemini", "api_key") is None
    store.set_fields("gemini", {"api_key": f"  {FAKE_GEMINI}  "}, ["api_key"])
    assert store.get_value("gemini", "api_key") == FAKE_GEMINI
    status = store.field_status("gemini", ["api_key"])
    assert status == {"api_key": {"saved": True, "masked": mask(FAKE_GEMINI)}}
    assert store.is_configured("gemini", ["api_key"])
    data = yaml.safe_load(store.path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["credentials"]["gemini"]["api_key"] == FAKE_GEMINI


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_file_mode_0600(vvl_env):
    store = SecretStore()
    store.set_fields("openai", {"api_key": FAKE_OPENAI}, ["api_key"])
    assert stat.S_IMODE(os.stat(store.path).st_mode) == 0o600


def test_other_providers_and_keys_preserved(vvl_env):
    path = vvl_env["VVL_PRIVATE_PATH"]
    path.write_text(yaml.safe_dump({
        "version": 1,
        "credentials": {"anthropic": {"api_key": "sk-ant-FAKE-keep-me-1234"}, "hyperbolic": {"api_key": ""}},
        "notes": {"kept": True},
    }), encoding="utf-8")
    store = SecretStore()
    store.set_fields("gemini", {"api_key": FAKE_GEMINI}, ["api_key"])
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["credentials"]["anthropic"]["api_key"] == "sk-ant-FAKE-keep-me-1234"
    assert data["credentials"]["hyperbolic"] == {"api_key": ""}
    assert data["notes"] == {"kept": True}
    assert data["credentials"]["gemini"]["api_key"] == FAKE_GEMINI


def test_empty_string_removes_and_delete(vvl_env):
    store = SecretStore()
    store.set_fields("gemini", {"api_key": FAKE_GEMINI}, ["api_key"])
    store.set_fields("openai", {"api_key": FAKE_OPENAI}, ["api_key"])
    store.set_fields("gemini", {"api_key": ""}, ["api_key"])
    data = yaml.safe_load(store.path.read_text(encoding="utf-8"))
    assert "gemini" not in data["credentials"]
    assert not store.is_configured("gemini", ["api_key"])
    store.delete_provider("openai")
    data = yaml.safe_load(store.path.read_text(encoding="utf-8"))
    assert data["credentials"] == {}


def test_atomic_write_keeps_old_file_on_failure(vvl_env, monkeypatch):
    store = SecretStore()
    store.set_fields("gemini", {"api_key": FAKE_GEMINI}, ["api_key"])
    before = store.path.read_bytes()

    def broken_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr("vvl.fileutil.os.replace", broken_replace)
    with pytest.raises(OSError):
        store.set_fields("openai", {"api_key": FAKE_OPENAI}, ["api_key"])
    assert store.path.read_bytes() == before
    leftovers = [p.name for p in store.path.parent.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_rejects_bad_input(vvl_env):
    store = SecretStore()
    with pytest.raises(SecretValueError):
        store.set_fields("gemini", {"token": "x"}, ["api_key"])
    with pytest.raises(SecretValueError):
        store.set_fields("gemini", {"api_key": 12345}, ["api_key"])
    with pytest.raises(SecretValueError) as excinfo:
        store.set_fields("gemini", {"api_key": "line1\nline2-secret"}, ["api_key"])
    assert "line2-secret" not in str(excinfo.value)
    with pytest.raises(SecretValueError):
        store.set_fields("gemini", {}, ["api_key"])


def test_unparsable_file_is_never_overwritten(vvl_env):
    path = vvl_env["VVL_PRIVATE_PATH"]
    path.write_text("credentials: [unclosed\n  api_key: SECRET-IN-BROKEN-FILE", encoding="utf-8")
    store = SecretStore()
    with pytest.raises(SecretsFileError) as excinfo:
        store.set_fields("gemini", {"api_key": FAKE_GEMINI}, ["api_key"])
    assert "SECRET-IN-BROKEN-FILE" not in str(excinfo.value)
    assert "SECRET-IN-BROKEN-FILE" in path.read_text(encoding="utf-8")


# ------------------------------------------------------------- API routes --

def test_put_and_delete_keys_routes(client, auth, vvl_env):
    response = client.put("/v1/keys/gemini", headers=auth, json={"fields": {"api_key": FAKE_GEMINI}})
    assert response.status_code == 200
    body = response.get_json()
    assert body["id"] == "gemini"
    assert body["configured"] is True
    assert body["fields"][0]["saved"] is True
    assert body["fields"][0]["masked"] == mask(FAKE_GEMINI)
    assert body["last_test"] is None
    assert FAKE_GEMINI not in response.get_data(as_text=True)

    listing = client.get("/v1/providers", headers=auth)
    assert FAKE_GEMINI not in listing.get_data(as_text=True)
    gemini = next(p for p in listing.get_json()["providers"] if p["id"] == "gemini")
    assert gemini["configured"] is True

    response = client.delete("/v1/keys/gemini", headers=auth)
    assert response.status_code == 200
    assert response.get_json()["configured"] is False
    assert response.get_json()["fields"][0] == {
        "name": "api_key", "label": "Gemini API key", "placeholder": "AIza…",
        "saved": False, "masked": None,
    }


def test_keys_route_validation(client, auth):
    assert client.put("/v1/keys/nope", headers=auth, json={"fields": {"api_key": "x"}}).status_code == 404
    bad_field = client.put("/v1/keys/gemini", headers=auth, json={"fields": {"password": "x"}})
    assert bad_field.status_code == 400
    assert bad_field.get_json()["error_code"] == "INVALID_INPUT"
    assert client.put("/v1/keys/gemini", headers=auth, json={"api_key": "x"}).status_code == 400
    assert client.put("/v1/keys/gemini", headers=auth, data="not json").status_code == 400


def test_values_never_in_responses_or_logs(client, auth, caplog):
    caplog.set_level(logging.DEBUG)
    responses = [
        client.put("/v1/keys/gemini", headers=auth, json={"fields": {"api_key": FAKE_GEMINI}}),
        client.put("/v1/keys/openai", headers=auth, json={"fields": {"api_key": FAKE_OPENAI}}),
        client.put("/v1/keys/huggingface", headers=auth, json={"fields": {"token": "hf_FAKEtokenvalue12345"}}),
        client.get("/v1/providers", headers=auth),
        client.get("/v1/models", headers=auth),
        client.get("/v1/app", headers=auth),
        client.get("/v1/settings", headers=auth),
        client.delete("/v1/keys/openai", headers=auth),
    ]
    secrets = (FAKE_GEMINI, FAKE_OPENAI, "hf_FAKEtokenvalue12345")
    for response in responses:
        text = response.get_data(as_text=True)
        assert not any(s in text for s in secrets)
    assert not any(s in caplog.text for s in secrets)
    assert "Credentials updated for gemini" in caplog.text
