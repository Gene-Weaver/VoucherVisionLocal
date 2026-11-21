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
    "prompt_name": "Tiny_test-1",
    "prompt_version": "v-1",
    "prompt_description": "Three fields for testing.",
    "LLM": "General Purpose",
    "instructions": "Refactor the unstructured text into a JSON dictionary.",
    "json_formatting_instructions": prompts.CANONICAL_JSON_FORMATTING_INSTRUCTIONS,
    "rules": {
        "catalogNumber": "Barcode identifier.",
        "scientificName": "Full scientific name.",
        "country": "Country name.",
    },
    "mapping": {
        "GEOGRAPHY": ["catalogNumber", "country"],
        "TAXONOMY": ["scientificName"],
        "COLLECTING": [],
    },
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
        data={"file": (io.BytesIO(good_yaml(prompt_name="uploaded").encode("utf-8")), "uploaded.yml")},
    )
    assert response.status_code == 201
    assert response.get_json()["name"] == "uploaded.yml"


@pytest.mark.parametrize("yaml_text", [
    "",
    "just a string",
    "- a\n- list\n",
    "prompt_name: [unclosed",
    good_yaml(prompt_name="bad", rules=["catalogNumber", "country"]),
    good_yaml(prompt_name="bad", rules={}),
    good_yaml(prompt_name="bad", mapping="GEOGRAPHY"),
    good_yaml(prompt_name="bad", mapping={"GEOGRAPHY": ["catalogNumber", "country"], "TAXONOMY": ["scientificName"]}),
    good_yaml(prompt_name="bad", mapping={"GEOGRAPHY": ["catalogNumber"], "TAXONOMY": ["scientificName"], "COLLECTING": []}),
    good_yaml(prompt_name="bad", instructions="   "),
    good_yaml(prompt_name=""),
    good_yaml(prompt_name="has spaces"),
    good_yaml(prompt_name="bad.dot"),
    good_yaml(prompt_name="different_stem"),
    good_yaml(prompt_name="bad", prompt_author="   "),
    good_yaml(prompt_name="bad", prompt_version=""),
    good_yaml(prompt_name="bad", prompt_description="   "),
    good_yaml(prompt_name="bad", json_formatting_instructions=None),
    good_yaml(prompt_name="bad", json_formatting_instructions="   "),
    yaml.safe_dump({k: v for k, v in GOOD_PROMPT.items() if k != "prompt_author"}),
    # Duplicate YAML key rejected
    good_yaml(prompt_name="bad") + "\nprompt_name: bad\n",
    # Non-JSON-compatible YAML values (unquoted date / NaN) rejected
    good_yaml(prompt_name="bad") + "\nextra_date: 2025-01-01\n",
    good_yaml(prompt_name="bad") + "\nextra_num: .nan\n",
])
def test_import_rejects_bad_yaml(client, auth, vvl_env, yaml_text):
    response = client.post("/v1/prompts/import", headers=auth, json={"name": "bad.yaml", "yaml_text": yaml_text})
    assert response.status_code == 400
    assert response.get_json()["error_code"] == "INVALID_INPUT"
    assert response.get_json()["error"]
    assert not (vvl_env["VVL_PROMPTS_USER_DIR"] / "bad.yaml").exists()


def test_import_rejects_large_and_non_utf8(client, auth):
    big = good_yaml(prompt_name="big") + "# " + "x" * (prompts.USER_PROMPTS_MAX_BYTES + 10) + "\n"
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
        data={"file": (io.BytesIO(good_yaml(prompt_name="escape").encode("utf-8")), "../../escape.yaml")},
    )
    assert response.status_code == 400
    assert not any(tmp_path.rglob("escape.yaml"))


def test_import_rejects_builtin_collision(client, auth):
    for name in ("SLTPvM_default.yaml", "sltpvm_default.yaml"):
        stem = name.rsplit(".", 1)[0]
        response = client.post("/v1/prompts/import", headers=auth, json={"name": name, "yaml_text": good_yaml(prompt_name=stem)})
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


# ------------------------------------------------------------ Designer API --

def _designer_payload(**overrides):
    base = {
        "prompt_author": "Test Author",
        "prompt_author_institution": "Test Herbarium",
        "prompt_name": "Designer_Test_Prompt",
        "prompt_version": "v-1-0",
        "prompt_description": "Prompt built in the Designer.",
        "LLM": "Thinking models",
        "instructions": "1. Extract fields into JSON.",
        "json_formatting_instructions": "Return valid JSON.",
        "rules": {
            "catalogNumber": "Specimen barcode.",
            "scientificName": "Scientific name.",
            "collectedBy": "Primary collector.",
            "country": "Country name.",
        },
        "mapping": {
            "GEOGRAPHY": ["catalogNumber", "country"],
            "TAXONOMY": ["scientificName"],
            "COLLECTING": ["collectedBy"],
        },
        "checklist": [
            "1. [VoucherVision] - Accept unreviewed fields",
            "2. [OCR] - Review disagreements",
        ],
        "field_default_values": {
            "country": ["no data"],
            "collectedBy": ["not present", "illegible"],
        },
        "review_not_required": ["catalogNumber"],
        "custom_flags": ["(Review) Needs check"],
    }
    base.update(overrides)
    return base


def test_detail_includes_editor_sections(client, auth):
    body = client.get("/v1/prompts/builtin/SLTPvM_full.yaml", headers=auth).get_json()
    assert len(body["checklist"]) >= 10
    assert "collectedBy" in body["field_default_values"]
    assert "scientificNameAuthorship" in body["review_not_required"]
    assert len(body["custom_flags"]) >= 1


def test_designer_create_update_render_and_roundtrip(client, auth, vvl_env):
    from vvl.prompting import PromptBuilder

    payload = _designer_payload(json_formatting_instructions="Attempted custom edit")
    rendered = client.post("/v1/prompts/render", headers=auth, json={"prompt": payload})
    assert rendered.status_code == 200
    assert rendered.get_json()["valid"] is True
    assert "GEOGRAPHY:" in rendered.get_json()["yaml_text"]
    # Formatting instructions are locked to the canonical standard text.
    assert prompts.CANONICAL_JSON_FORMATTING_INSTRUCTIONS in rendered.get_json()["yaml_text"]
    assert "Attempted custom edit" not in rendered.get_json()["yaml_text"]

    # File name must match prompt_name.
    mismatch = client.post("/v1/prompts/user", headers=auth, json={"name": "Wrong_File.yaml", "prompt": payload})
    assert mismatch.status_code == 400

    created = client.post("/v1/prompts/user", headers=auth, json={"name": "Designer_Test_Prompt.yaml", "prompt": payload})
    assert created.status_code == 201, created.get_json()
    body = created.get_json()
    assert body["name"] == "Designer_Test_Prompt.yaml"
    assert body["source"] == "user"
    assert body["json_formatting_instructions"] == prompts.CANONICAL_JSON_FORMATTING_INSTRUCTIONS
    assert body["checklist"] == payload["checklist"]
    assert body["field_default_values"] == payload["field_default_values"]
    assert body["review_not_required"] == ["catalogNumber"]
    assert body["custom_flags"] == ["(Review) Needs check"]

    # Saved file works with PromptBuilder (real inference pipeline) and passes strict re-import.
    saved_path = vvl_env["VVL_PROMPTS_USER_DIR"] / "Designer_Test_Prompt.yaml"
    assert saved_path.is_file()
    pb = PromptBuilder(saved_path)
    assert list(pb.rules) == ["catalogNumber", "scientificName", "collectedBy", "country"]

    # Update in place (with an added custom ALL-CAPS category AND rename).
    updated_payload = _designer_payload(
        prompt_name="Designer_Renamed-2",
        prompt_version="v-1-1",
        rules={**payload["rules"], "notes": "Extra notes."},
        mapping={**payload["mapping"], "NOTES": ["notes"]},
    )
    upd = client.put("/v1/prompts/user/Designer_Test_Prompt.yaml", headers=auth, json={"prompt": updated_payload})
    assert upd.status_code == 200, upd.get_json()
    assert upd.get_json()["name"] == "Designer_Renamed-2.yaml"
    assert upd.get_json()["prompt_version"] == "v-1-1"
    assert "NOTES" in upd.get_json()["mapping"]
    assert not saved_path.exists()
    assert (vvl_env["VVL_PROMPTS_USER_DIR"] / "Designer_Renamed-2.yaml").is_file()

    # Built-in cannot be edited.
    forbid = client.put("/v1/prompts/builtin/SLTPvM_full.yaml", headers=auth, json={"prompt": payload})
    assert forbid.status_code == 403


def test_designer_prevents_name_overlaps(client, auth):
    payload = _designer_payload(prompt_name="Unique_One")
    assert client.post("/v1/prompts/user", headers=auth, json={"name": "Unique_One.yaml", "prompt": payload}).status_code == 201

    # Duplicate filename / prompt_name (case-insensitive) rejected.
    assert client.post(
        "/v1/prompts/user", headers=auth,
        json={"name": "unique_one.yaml", "prompt": _designer_payload(prompt_name="unique_one")},
    ).status_code == 409

    # Built-in filename & prompt_name collision rejected.
    assert client.post(
        "/v1/prompts/user", headers=auth,
        json={"name": "SLTPvM_full.yaml", "prompt": _designer_payload(prompt_name="SLTPvM_full")},
    ).status_code == 409


@pytest.mark.parametrize("mutator", [
    lambda p: p["mapping"].pop("COLLECTING"),
    lambda p: p["mapping"].update({"A": [], "B": [], "C": [], "D": []}),
    lambda p: p["mapping"].update({"lowercase": []}),
    lambda p: p["mapping"]["GEOGRAPHY"].remove("country"),
    lambda p: p["mapping"]["TAXONOMY"].append("country"),
    lambda p: p["mapping"]["GEOGRAPHY"].append("nonexistentField"),
    lambda p: p["field_default_values"].update({"ghostField": ["x"]}),
    lambda p: p["review_not_required"].append("ghostField"),
    lambda p: p["rules"].update({"bad field": "has space"}),
    lambda p: p.update({"prompt_version": ""}),
    lambda p: p.update({"prompt_description": ""}),
    lambda p: p.update({"prompt_name": "Has Spaces"}),
    lambda p: p.update({"prompt_name": "Bad.Punctuation!"}),
])
def test_designer_rejects_invalid_prompts(client, auth, mutator):
    payload = _designer_payload(prompt_name="Invalid_Check")
    mutator(payload)
    res = client.post("/v1/prompts/user", headers=auth, json={"name": "Invalid_Check.yaml", "prompt": payload})
    assert res.status_code == 400
    assert res.get_json()["error_code"] == "INVALID_INPUT"
