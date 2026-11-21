"""Settings: defaults, allowlisted partial updates, atomic writes, selection checks."""

from __future__ import annotations

import os

import pytest
import yaml

from vvl import settings

pytestmark = pytest.mark.usefixtures("no_network")


def test_defaults_without_file(client, auth, vvl_env):
    body = client.get("/v1/settings", headers=auth).get_json()
    assert set(body) == {"models_dir", "output_dir", "selection", "enrichment", "ui", "engine", "generation", "openrouter", "setup"}
    assert body["setup"] == {"usage": None}
    assert body["generation"] == {
        "ocr": {"reasoning": "off", "strength": "high", "output_length": "standard", "image_detail": "standard"},
        "parse": {"reasoning": "off", "strength": "high", "output_length": "standard"}, "context": {}}
    assert body["engine"] == {"variant": "auto", "context_size": "auto", "parallel": "auto", "gpu_devices": "auto",
                              "idle_stop_minutes": 15, "staged_batches": False}
    assert body["selection"] == {"ocr": ["demo:mock"], "parse": "demo:mock"}
    assert body["enrichment"] == {"elevation": True, "wfo_source": "off"}
    assert body["ui"]["theme"] in settings.THEMES
    assert body["models_dir"] == str(vvl_env["VVL_HOME"] / "models")
    assert body["output_dir"] == str(vvl_env["VVL_HOME"] / "runs")
    assert not vvl_env["VVL_SETTINGS_PATH"].exists()


def test_partial_update_merges(client, auth, vvl_env, tmp_path):
    models_dir = str(tmp_path / "models_here")
    response = client.put("/v1/settings", headers=auth, json={
        "models_dir": models_dir, "enrichment": {"wfo_source": "vvgo"}, "ui": {"theme": "light"},
    })
    assert response.status_code == 200
    body = response.get_json()
    assert body["models_dir"] == models_dir
    assert body["enrichment"] == {"elevation": True, "wfo_source": "vvgo"}
    assert body["ui"] == {"theme": "light"}
    assert isinstance(body["warnings"], list)
    assert any(w["code"] == "MODELS_DIR_MISSING" for w in body["warnings"])

    body = client.put("/v1/settings", headers=auth, json={"enrichment": {"elevation": False}}).get_json()
    assert body["enrichment"] == {"elevation": False, "wfo_source": "vvgo"}
    assert body["models_dir"] == models_dir
    stored = yaml.safe_load(vvl_env["VVL_SETTINGS_PATH"].read_text(encoding="utf-8"))
    assert stored["enrichment"] == {"wfo_source": "vvgo", "elevation": False}
    assert client.get("/v1/settings", headers=auth).get_json()["ui"]["theme"] == "light"


@pytest.mark.parametrize("payload", [
    {"secret_key": 1},
    {"models_dir": "relative/path"},
    {"models_dir": ""},
    {"models_dir": 42},
    {"enrichment": {"elevation": "yes"}},
    {"enrichment": {"wfo_source": "gbif"}},
    {"enrichment": {"other": True}},
    {"ui": {"theme": "neon"}},
    {"ui": {"font": "x"}},
    {"selection": {"ocr": []}},
    {"selection": {"ocr": ["demo:mock", "demo:mock"]}},
    {"selection": {"parse": "gemini-2.5-flash"}},
    {"selection": {"parse": "local:not-a-build"}},
    {"selection": {"parse": "api:nobody:model"}},
    {"selection": {"other": "x"}},
    {"setup": {"usage": "cloud"}},
    {"setup": {"usage": True}},
    {"setup": {"other": "x"}},
    {"setup": "local"},
    [1, 2, 3],
])
def test_rejected_updates(client, auth, vvl_env, payload):
    response = client.put("/v1/settings", headers=auth, json=payload)
    assert response.status_code == 400
    assert response.get_json()["error_code"] == "INVALID_INPUT"
    assert not vvl_env["VVL_SETTINGS_PATH"].exists()


def test_unknown_keys_reported(client, auth):
    body = client.put("/v1/settings", headers=auth, json={"theme": "dark", "models_dir": "/x"}).get_json()
    assert body["details"]["unknown_keys"] == ["theme"]


def test_unmanaged_keys_in_file_preserved(client, auth, vvl_env):
    path = vvl_env["VVL_SETTINGS_PATH"]
    path.write_text("# hand-written\nmodels_dir: /opt/vvl_models\nexperimental: {flag: true}\n", encoding="utf-8")
    assert client.get("/v1/settings", headers=auth).get_json()["models_dir"] == os.path.normpath("/opt/vvl_models")
    client.put("/v1/settings", headers=auth, json={"ui": {"theme": "system"}})
    stored = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert stored["experimental"] == {"flag": True}
    assert stored["models_dir"] == "/opt/vvl_models"
    assert stored["ui"] == {"theme": "system"}
    assert [p.name for p in path.parent.iterdir() if p.name.endswith(".tmp")] == []


def test_unreadable_file_is_set_aside(client, auth, vvl_env):
    path = vvl_env["VVL_SETTINGS_PATH"]
    path.write_text("models_dir: [unclosed\n", encoding="utf-8")
    assert client.get("/v1/settings", headers=auth).status_code == 200
    assert client.put("/v1/settings", headers=auth, json={"ui": {"theme": "dark"}}).status_code == 200
    backups = [p for p in path.parent.iterdir() if p.name.startswith(path.name + ".unreadable-")]
    assert len(backups) == 1
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["ui"] == {"theme": "dark"}


def test_synced_folder_warning(client, auth):
    body = client.put("/v1/settings", headers=auth, json={"models_dir": "/home/u/Dropbox/models"}).get_json()
    assert any(w["code"] == "MODELS_DIR_SYNCED" for w in body["warnings"])
    body = client.put("/v1/settings", headers=auth, json={"models_dir": "/home/u/OneDrive - Org/models"}).get_json()
    assert any(w["code"] == "MODELS_DIR_SYNCED" for w in body["warnings"])
    body = client.put("/v1/settings", headers=auth, json={"models_dir": "/datac/VVL_models"}).get_json()
    assert not any(w["code"] == "MODELS_DIR_SYNCED" for w in body["warnings"])


def test_selection_route(client, auth):
    response = client.put("/v1/selection", headers=auth, json={"ocr": ["demo:mock", "local:qwen3-vl-8b-q8"], "parse": "local:qwen3-vl-2b-q8"})
    assert response.status_code == 200
    assert response.get_json() == {"selection": {"ocr": ["demo:mock", "local:qwen3-vl-8b-q8"], "parse": "local:qwen3-vl-2b-q8"}}
    assert client.get("/v1/settings", headers=auth).get_json()["selection"]["parse"] == "local:qwen3-vl-2b-q8"
    # An API model id is accepted when its provider exists and no list has been fetched.
    assert client.put("/v1/selection", headers=auth, json={"parse": "api:gemini:gemini-3.1-flash-lite"}).status_code == 200
    bad = client.put("/v1/selection", headers=auth, json={"ocr": ["api:vvgo:anything"]})
    assert bad.status_code == 400
    assert client.put("/v1/selection", headers=auth, json={"ocr": "demo:mock", "extra": 1}).status_code == 400


@pytest.mark.parametrize("usage", ["openrouter", "local", "both"])
def test_setup_usage_saved(client, auth, vvl_env, usage):
    body = client.put("/v1/settings", headers=auth, json={"setup": {"usage": usage}}).get_json()
    assert body["setup"] == {"usage": usage}
    assert client.get("/v1/settings", headers=auth).get_json()["setup"] == {"usage": usage}
    assert yaml.safe_load(vvl_env["VVL_SETTINGS_PATH"].read_text(encoding="utf-8"))["setup"] == {"usage": usage}
    # null puts the question back.
    body = client.put("/v1/settings", headers=auth, json={"setup": {"usage": None}}).get_json()
    assert body["setup"] == {"usage": None}


def test_setup_usage_invalid_on_disk_reads_as_null(vvl_env):
    vvl_env["VVL_SETTINGS_PATH"].write_text(yaml.safe_dump({"setup": {"usage": "cloud"}}), encoding="utf-8")
    assert settings.load()["setup"] == {"usage": None}
    vvl_env["VVL_SETTINGS_PATH"].write_text(yaml.safe_dump({"setup": "both"}), encoding="utf-8")
    assert settings.load()["setup"] == {"usage": None}


def test_load_ignores_invalid_stored_values(vvl_env):
    vvl_env["VVL_SETTINGS_PATH"].write_text(yaml.safe_dump({
        "models_dir": "not/absolute", "selection": {"ocr": 5, "parse": ""},
        "enrichment": {"elevation": "maybe", "wfo_source": "x"}, "ui": {"theme": 3},
    }), encoding="utf-8")
    current = settings.load()
    assert current["models_dir"] == str(vvl_env["VVL_HOME"] / "models")
    assert current["selection"] == settings.DEFAULTS["selection"]
    assert current["enrichment"] == settings.DEFAULTS["enrichment"]
    assert current["ui"] == settings.DEFAULTS["ui"]
