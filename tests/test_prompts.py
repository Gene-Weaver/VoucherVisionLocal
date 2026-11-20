"""Prompts: built-ins list as valid, import validation, name confinement, legacy /prompts."""

from __future__ import annotations

import io

import pytest
import yaml

from vvl import paths, prompts

pytestmark = pytest.mark.usefixtures("no_network")

GOOD_PROMPT = {
    "prompt_author": "Test Author",
    "prompt_author_institution": "Test Herbarium",
    "prompt_name": "Tiny test prompt",
    "prompt_version": "v-1",
    "prompt_description": "Three fields for testing.",
    "LLM": "General Purpose",
    "instructions": "Refactor the unstructured text into a JSON dictionary.",
    "json_formatting_instructions": "Return valid JSON only.",
    "rules": {
        "catalogNumber": "Barcode identifier.",
        "scientificName": "Full scientific name.",
        "country": "Country name.",
    },
    "mapping": {"TAXONOMY": ["scientificName"], "GEOGRAPHY": ["catalogNumber", "country"]},
}


def good_yaml(**changes):
    data = dict(GOOD_PROMPT)
    data.update(changes)
    return yaml.safe_dump(data, sort_keys=False)


def test_all_builtins_valid(client, auth):
    body = client.get("/v1/prompts", headers=auth).get_json()
    builtins = [p for p in body["prompts"] if p["source"] == "builtin"]
    assert len(builtins) == 18
    for item in builtins:
        assert item["valid"] is True, (item["name"], item["error"])
        assert item["error"] is None
        assert item["n_fields"] > 0
        assert item["prompt_name"]
    assert set(builtins[0]) == {
        "name", "source", "prompt_name", "prompt_author", "prompt_author_institution", "prompt_version",
        "prompt_description", "llm", "n_fields", "valid", "error",
    }
    default = next(p for p in builtins if p["name"] == "SLTPvM_default.yaml")
    assert default["n_fields"] == 30
    assert default["llm"] == "General Purpose"


def test_prompt_detail(client, auth):
    response = client.get("/v1/prompts/builtin/SLTPvM_default.yaml", headers=auth)
    assert response.status_code == 200
    body = response.get_json()
    assert body["name"] == "SLTPvM_default.yaml"
    assert body["source"] == "builtin"
    assert body["valid"] is True
    assert len(body["fields"]) == 30
    first = body["fields"][0]
    assert first["name"] == "catalogNumber"
    assert first["group"] == "GEOGRAPHY"
    assert first["rule"]
    assert "TAXONOMY" in body["mapping"]
    assert body["instructions"]
    assert body["yaml_text"].startswith(
        (paths.prompts_builtin_dir() / "SLTPvM_default.yaml").read_text(encoding="utf-8")[:40])


def test_import_json_then_list_detail_delete(client, auth, vvl_env):
    response = client.post("/v1/prompts/import", headers=auth, json={"name": "Tiny_test-1.yaml", "yaml_text": good_yaml()})
    assert response.status_code == 201, response.get_json()
    summary = response.get_json()
    assert summary["name"] == "Tiny_test-1.yaml"
    assert summary["source"] == "user"
    assert summary["valid"] is True
    assert summary["n_fields"] == 3
    assert (vvl_env["VVL_PROMPTS_USER_DIR"] / "Tiny_test-1.yaml").is_file()

    listing = client.get("/v1/prompts", headers=auth).get_json()["prompts"]
    assert [p["name"] for p in listing if p["source"] == "user"] == ["Tiny_test-1.yaml"]
    detail = client.get("/v1/prompts/user/Tiny_test-1.yaml", headers=auth).get_json()
    assert [f["name"] for f in detail["fields"]] == ["catalogNumber", "scientificName", "country"]

    again = client.post("/v1/prompts/import", headers=auth, json={"name": "Tiny_test-1.yaml", "yaml_text": good_yaml()})
    assert again.status_code == 409
    assert again.get_json()["error_code"] == "CONFLICT"

    assert client.delete("/v1/prompts/user/Tiny_test-1.yaml", headers=auth).status_code == 200
    assert not (vvl_env["VVL_PROMPTS_USER_DIR"] / "Tiny_test-1.yaml").exists()
    assert client.delete("/v1/prompts/user/Tiny_test-1.yaml", headers=auth).status_code == 404


def test_import_multipart(client, auth):
    response = client.post(
        "/v1/prompts/import", headers=auth, content_type="multipart/form-data",
        data={"file": (io.BytesIO(good_yaml().encode("utf-8")), "uploaded.yml")},
    )
    assert response.status_code == 201
    assert response.get_json()["name"] == "uploaded.yml"


@pytest.mark.parametrize("yaml_text", [
    "",
    "just a string",
    "- a\n- list\n",
    "prompt_name: [unclosed",
    good_yaml(rules=["catalogNumber", "country"]),
    good_yaml(rules={}),
    good_yaml(mapping="GEOGRAPHY"),
    good_yaml(instructions="   "),
    good_yaml(prompt_name=""),
    good_yaml(json_formatting_instructions=None),
    yaml.safe_dump({k: v for k, v in GOOD_PROMPT.items() if k != "prompt_author"}),
])
def test_import_rejects_bad_yaml(client, auth, vvl_env, yaml_text):
    response = client.post("/v1/prompts/import", headers=auth, json={"name": "bad.yaml", "yaml_text": yaml_text})
    assert response.status_code == 400
    assert response.get_json()["error_code"] == "INVALID_INPUT"
    assert not (vvl_env["VVL_PROMPTS_USER_DIR"] / "bad.yaml").exists()


def test_import_rejects_large_and_non_utf8(client, auth):
    big = good_yaml() + "# " + "x" * (prompts.USER_PROMPTS_MAX_BYTES + 10) + "\n"
    response = client.post("/v1/prompts/import", headers=auth, json={"name": "big.yaml", "yaml_text": big})
    assert response.status_code == 413
    response = client.post(
        "/v1/prompts/import", headers=auth, content_type="multipart/form-data",
        data={"file": (io.BytesIO(b"prompt_name: \xff\xfe\n"), "latin.yaml")},
    )
    assert response.status_code == 400


@pytest.mark.parametrize("name", [
    "../evil.yaml", "../../etc/passwd.yaml", "a/b.yaml", "a\\b.yaml", "evil.txt", "evil", ".hidden.yaml",
    "spaces in name.yaml", "x.yaml\x00.txt", "", "%2e%2e%2fevil.yaml",
])
def test_import_rejects_bad_names(client, auth, vvl_env, tmp_path, name):
    response = client.post("/v1/prompts/import", headers=auth, json={"name": name, "yaml_text": good_yaml()})
    assert response.status_code == 400
    assert not any(tmp_path.rglob("evil*"))


def test_import_rejects_traversal_filename_in_multipart(client, auth, tmp_path):
    response = client.post(
        "/v1/prompts/import", headers=auth, content_type="multipart/form-data",
        data={"file": (io.BytesIO(good_yaml().encode("utf-8")), "../../escape.yaml")},
    )
    assert response.status_code == 400
    assert not any(tmp_path.rglob("escape.yaml"))


def test_import_rejects_builtin_collision(client, auth):
    for name in ("SLTPvM_default.yaml", "sltpvm_default.yaml"):
        response = client.post("/v1/prompts/import", headers=auth, json={"name": name, "yaml_text": good_yaml()})
        assert response.status_code == 409


def test_detail_and_delete_confinement(client, auth):
    assert client.get("/v1/prompts/builtin/../app.py", headers=auth).status_code == 404
    assert client.get("/v1/prompts/builtin/..%2F..%2Fapp.yaml", headers=auth).status_code in (400, 404)
    assert client.get("/v1/prompts/builtin/nope.yaml", headers=auth).status_code == 404
    assert client.get("/v1/prompts/builtin/bad name.yaml", headers=auth).status_code == 400
    assert client.get("/v1/prompts/other/SLTPvM_default.yaml", headers=auth).status_code == 404
    response = client.delete("/v1/prompts/builtin/SLTPvM_default.yaml", headers=auth)
    assert response.status_code == 403
    assert (paths.prompts_builtin_dir() / "SLTPvM_default.yaml").is_file()


def test_symlink_out_of_user_folder_is_ignored(client, auth, vvl_env, tmp_path):
    outside = tmp_path / "outside.yaml"
    outside.write_text(good_yaml(), encoding="utf-8")
    user_dir = vvl_env["VVL_PROMPTS_USER_DIR"]
    user_dir.mkdir(parents=True, exist_ok=True)
    try:
        (user_dir / "linked.yaml").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available")
    names = [p["name"] for p in client.get("/v1/prompts", headers=auth).get_json()["prompts"]]
    assert "linked.yaml" not in names
    assert client.get("/v1/prompts/user/linked.yaml", headers=auth).status_code == 404


def test_legacy_prompts_route(client, auth):
    body = client.get("/prompts", headers=auth).get_json()
    assert body["status"] == "success"
    assert body["count"] == 18
    entry = body["prompts"][0]
    for key in ("filename", "description", "version", "author", "institution", "name", "source",
                "is_user_generated", "prompt_ref"):
        assert key in entry
    one = client.get("/prompts?prompt=SLTPvM_default.yaml", headers=auth).get_json()
    assert one["prompt"]["details"]["parsed_data"]["prompt_name"]
    missing = client.get("/prompts?prompt=nope.yaml", headers=auth)
    assert missing.status_code == 404
    assert missing.get_json()["error_code"] == "PROMPT_NOT_FOUND"
    text = client.get("/prompts?format=text", headers=auth)
    assert text.status_code == 200
    assert "SLTPvM_default.yaml" in text.get_data(as_text=True)
