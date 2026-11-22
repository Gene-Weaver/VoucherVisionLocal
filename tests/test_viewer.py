"""Viewer API (vvl/viewer.py): project listing, aggregate stats, record view, and base64 image decoding."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pytest

from conftest import FIXTURE_IMAGE

pytestmark = pytest.mark.usefixtures("no_network")


def _post_image(client, auth, batch_id, filename="MICH_16205594.jpg"):
    return client.post(
        "/process",
        headers=auth,
        data={"output_batch": batch_id, "file": (io.BytesIO(FIXTURE_IMAGE.read_bytes()), filename)},
        content_type="multipart/form-data",
    )


def test_viewer_projects_stats_record_and_independent_image_decoding(client, auth, vvl_env):
    # Create a JSON-only run (no external input_image/ or label_collage/ folders on disk!).
    batch = client.post(
        "/v1/output/batches",
        headers=auth,
        json={"name": "viewer_demo", "save": {"xlsx": False, "label_collage": False, "input_image": False, "ocr_txt": False}},
    ).get_json()
    assert _post_image(client, auth, batch["id"], "spec_a.jpg").status_code == 200
    assert _post_image(client, auth, batch["id"], "spec_b.jpg").status_code == 200
    client.post(f"/v1/output/batches/{batch['id']}/finish", headers=auth)

    folder = Path(batch["path"])
    assert not (folder / "input_image").exists()
    assert not (folder / "label_collage").exists()

    # Project listing finds the run.
    listed = client.get("/v1/viewer/projects", headers=auth).get_json()
    assert len(listed["projects"]) == 1
    assert listed["projects"][0]["json_count"] == 2

    # Open project and verify aggregate stats + lightweight index.
    opened = client.post("/v1/viewer/open", headers=auth, json={"path": str(folder)}).get_json()
    pid = opened["project"]["id"]
    assert opened["stats"]["total_records"] == 2
    assert opened["stats"]["parsed_records"] == 2
    assert len(opened["records"]) == 2
    assert opened["records"][0]["file"] == "spec_a.json"
    assert opened["records"][0]["has_input_image"] is True
    assert opened["records"][0]["has_collage_image"] is True

    # Record endpoint strips giant base64 strings while preserving formatted_json, ocr, metrics.
    rec = client.get(f"/v1/viewer/projects/{pid}/records/spec_a.json", headers=auth).get_json()
    assert rec["images"]["has_input"] is True
    assert rec["images"]["has_collage"] is True
    assert rec["record"]["collage_info"]["base64image_input_resized"].startswith("<base64 image")
    assert rec["record"]["collage_info"]["base64image_text_collage"].startswith("<base64 image")
    assert isinstance(rec["record"]["formatted_json"], dict)
    assert isinstance(rec["record"]["ocr"], str)
    assert "total_tokens" in rec["metrics"]

    # Images are decoded directly from the JSON file (independent worker process), even with no external image files.
    in_resp = client.get(f"/v1/viewer/projects/{pid}/images/spec_a.json/input", headers=auth)
    assert in_resp.status_code == 200
    assert in_resp.mimetype == "image/jpeg"
    assert in_resp.data[:2] == b"\xff\xd8"

    col_resp = client.get(f"/v1/viewer/projects/{pid}/images/spec_a.json/collage", headers=auth)
    assert col_resp.status_code == 200
    assert col_resp.mimetype == "image/jpeg"
    assert col_resp.data[:2] == b"\xff\xd8"

    # Confinement & invalid kind checks.
    assert client.get(f"/v1/viewer/projects/{pid}/images/spec_a.json/other", headers=auth).status_code == 400
    assert client.get(f"/v1/viewer/projects/{pid}/records/..%2Fevil.json", headers=auth).status_code in (400, 404)
    assert client.get(f"/v1/viewer/projects/{pid}/records/missing.json", headers=auth).status_code == 404


def test_viewer_opens_standalone_json_folder_and_aggregates_costs(client, auth, tmp_path):
    raw_jpg = FIXTURE_IMAGE.read_bytes()
    b64 = base64.b64encode(raw_jpg).decode("ascii")
    doc = {
        "filename": "herb_001.jpg",
        "prompt": "SLTPvM_full.yaml",
        "ocr": "Acer rubrum L.",
        "formatted_json": {"scientificName": "Acer rubrum", "catalogNumber": "12345", "decimalLatitude": "42.28", "decimalLongitude": "-83.74"},
        "ocr_info": {
            "api:openrouter:demo-ocr": {"tokens_in": 1000, "tokens_out": 200, "thinking_tokens": 50, "cost_in": 0.001, "cost_out": 0.002, "total_cost": 0.003},
        },
        "parsing_info": {
            "model": "api:openrouter:demo-parse", "input": 500, "output": 150, "thinking_tokens": 100,
            "cost_in": 0.0005, "cost_out": 0.0015, "thinking_cost": 0.0005, "total_cost": 0.0025,
        },
        "total_request_cost_usd": 0.0055,
        "collage_info": {"base64image_input_resized": b64, "base64image_text_collage": b64},
        "success": {"ocr": "True", "llm": "True", "text_collage": "True"},
    }
    (tmp_path / "herb_001.json").write_text(json.dumps(doc), encoding="utf-8")

    opened = client.post("/v1/viewer/open", headers=auth, json={"path": str(tmp_path)}).get_json()
    stats = opened["stats"]
    assert stats["total_records"] == 1
    assert stats["has_coords"] == 1
    assert stats["total_tokens"] == 2000
    assert stats["total_cost_usd"] == pytest.approx(0.0055)
    assert stats["ocr_cost_usd"] == pytest.approx(0.003)
    assert stats["parse_cost_usd"] == pytest.approx(0.0025)
    assert stats["ocr_models"] == ["api:openrouter:demo-ocr"]
    assert stats["parse_models"] == ["api:openrouter:demo-parse"]
    assert stats["prompts"] == ["SLTPvM_full.yaml"]
