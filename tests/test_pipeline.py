"""/process and /process-url end to end with the demo engine on the fixture image."""

from __future__ import annotations

import io

import pytest
import yaml

from conftest import FIXTURE_IMAGE
from vvl import paths
from vvl.engines.mock import DEMO_NOTICE, SAMPLE_RECORD

pytestmark = pytest.mark.usefixtures("no_network")

RESULT_KEYS = [
    "filename", "url_source", "host", "prompt", "ocr_info", "WFO_info", "COP90_elevation_m", "ocr",
    "formatted_json", "formatted_md", "parsing_info", "impact", "collage_info", "collage_image_format",
    "success", "total_request_cost_usd", "engine_notice",
]


def post_image(client, auth, data=None, filename="MICH_16205594.jpg", payload=None):
    body = dict(data or {})
    body["file"] = (io.BytesIO(payload if payload is not None else FIXTURE_IMAGE.read_bytes()), filename)
    return client.post("/process", headers=auth, data=body, content_type="multipart/form-data")


def rules_of(name):
    data = yaml.safe_load((paths.prompts_builtin_dir() / name).read_text(encoding="utf-8"))
    return list(data["rules"])


@pytest.mark.parametrize("prompt", ["SLTPvM_default.yaml", "SLTPvM_skeletal.yaml", "NBC_Lepidoptera.yaml"])
def test_process_demo_record_has_exactly_the_prompt_fields(client, auth, prompt, vvl_env):
    response = post_image(client, auth, {"prompt": prompt, "engines": "demo:mock", "llm_model": "demo:mock"})
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert list(body) == RESULT_KEYS
    record = body["formatted_json"]
    assert list(record) == rules_of(prompt)
    lowered = {k.lower(): v for k, v in SAMPLE_RECORD.items()}
    for field, value in record.items():
        expected = lowered.get(field.lower(), "")
        assert value == expected.replace("\n", " ").strip() or value == expected or expected == "", field
    assert body["engine_notice"] == DEMO_NOTICE
    assert body["host"] == "VoucherVisionLocal"
    assert body["prompt"] == prompt
    assert body["filename"] == "MICH_16205594.jpg"
    assert body["impact"] is None
    assert "payment_inference" not in body
    assert list(body["ocr_info"]) == ["demo:mock"]
    assert body["ocr_info"]["demo:mock"]["tokens_in"] == 0
    assert "KATHRYN KALMBACH HERBARIUM" in body["ocr"]
    assert body["parsing_info"]["model"] == "demo:mock"
    assert body["parsing_info"]["total_cost"] == 0
    assert body["total_request_cost_usd"] == 0
    assert body["success"] == {"image_available": "True", "text_collage": "True", "text_collage_resize": "gemini",
                               "ocr": "True", "llm": "True"}
    collage = body["collage_info"]
    assert collage["base64image_text_collage"] and collage["base64image_input_resized"]
    assert collage["position_original"]["label"]
    # Temp files are always removed.
    tmp_dir = vvl_env["VVL_HOME"] / "tmp"
    assert not tmp_dir.exists() or list(tmp_dir.iterdir()) == []


def test_process_defaults_to_saved_selection_and_default_prompt(client, auth):
    body = post_image(client, auth).get_json()
    assert body["prompt"] == "SLTPvM_full.yaml"
    assert list(body["formatted_json"]) == rules_of("SLTPvM_full.yaml")
    assert body["parsing_info"]["model"] == "demo:mock"


def test_process_values_from_sample(client, auth):
    record = post_image(client, auth, {"prompt": "SLTPvM_default.yaml"}).get_json()["formatted_json"]
    assert record["catalogNumber"] == "KHD00041592"
    assert record["scientificName"] == "Galax aphylla L."
    assert record["country"] == "United States"
    assert record["decimalLatitude"] == ""


def test_process_ocr_only_and_notebook(client, auth):
    body = post_image(client, auth, {"ocr_only": "true"}).get_json()
    assert body["formatted_json"] == ""
    assert body["prompt"] == "verbatim_with_annotations"
    assert body["success"]["llm"] == "False"
    assert body["parsing_info"]["model"] == ""
    assert body["engine_notice"] == DEMO_NOTICE

    body = post_image(client, auth, {"notebook_mode": "true", "llm_model": "local:qwen3-vl-8b-q8"}).get_json()
    assert body["prompt"] == "verbatim_notebook"
    assert body["formatted_md"].startswith("# KATHRYN KALMBACH HERBARIUM")
    assert not body["ocr"].startswith("#")  # Markdown headings flattened in the plain copy
    assert body["success"]["text_collage"] == "False"


def test_skip_label_collage(client, auth):
    body = post_image(client, auth, {"skip_label_collage": "true"}).get_json()
    assert body["success"]["text_collage"] == "False"
    assert body["collage_info"]["position_original"] == {}


@pytest.mark.parametrize("field,value", [
    ("engines", "local:qwen3-vl-8b-q8"),
    ("llm_model", "local:qwen3-vl-30b-a3b-q4"),
    ("engines", "api:gemini:gemini-3.1-flash-lite"),
    ("llm_model", "api:anthropic:claude-sonnet"),
])
def test_process_non_runnable_model_is_409(client, auth, field, value):
    response = post_image(client, auth, {field: value})
    assert response.status_code == 409
    body = response.get_json()
    assert body["error_code"] == "MODEL_NOT_AVAILABLE"
    assert body["details"]["model_id"] == value
    assert body["details"]["unavailable_reason"] in (
        "Download this model and install the llama.cpp engine first (Models and Engine tabs).",
        "This provider is not available in this version yet. Use OpenRouter for hosted models.")
    assert body["message"] == body["details"]["unavailable_reason"]


def test_process_saved_selection_not_runnable_is_409(client, auth):
    assert client.put("/v1/selection", headers=auth, json={"parse": "local:qwen3-vl-4b-q8"}).status_code == 200
    assert post_image(client, auth).status_code == 409
    assert post_image(client, auth, {"llm_model": "demo:mock"}).status_code == 200


@pytest.mark.parametrize("field,value", [
    ("engines", "gemini-2.5-flash"), ("engines", "local:not-a-build"), ("llm_model", "api:nobody:m"),
    ("engines", "demo:mock,demo:mock,a:b,c:d,e:f"),
])
def test_process_unknown_model_is_400(client, auth, field, value):
    response = post_image(client, auth, {field: value})
    assert response.status_code == 400
    assert response.get_json()["error_code"] == "INVALID_INPUT"


def test_process_request_errors(client, auth):
    assert client.post("/process", headers=auth, data={}, content_type="multipart/form-data").status_code == 400
    assert post_image(client, auth, filename="notes.txt", payload=b"hello").status_code == 415
    corrupt = post_image(client, auth, filename="broken.jpg", payload=b"\xff\xd8 not really a jpeg")
    assert corrupt.status_code == 400
    assert corrupt.get_json()["error_code"] == "INVALID_INPUT"
    missing = post_image(client, auth, {"prompt": "nope.yaml"})
    assert missing.status_code == 404
    assert missing.get_json()["error_code"] == "PROMPT_NOT_FOUND"
    assert post_image(client, auth, {"prompt": "../app.py"}).status_code == 400
    bad_level = post_image(client, auth, {"ocr_thinking_level": "max"})
    assert bad_level.status_code == 400
    assert bad_level.get_json()["details"]["allowed_thinking_levels"] == ["low", "medium", "high"]


def test_process_uses_user_prompt(client, auth):
    data = yaml.safe_load((paths.prompts_builtin_dir() / "SLTPvM_skeletal.yaml").read_text(encoding="utf-8"))
    data["rules"] = {"catalogNumber": "Barcode.", "myCustomField": "Anything."}
    data["mapping"] = {"MISC": ["catalogNumber", "myCustomField"]}
    assert client.post("/v1/prompts/import", headers=auth,
                       json={"name": "custom.yaml", "yaml_text": yaml.safe_dump(data)}).status_code == 201
    body = post_image(client, auth, {"prompt": "custom.yaml"}).get_json()
    assert body["formatted_json"] == {"catalogNumber": "KHD00041592", "myCustomField": ""}


def test_process_png_and_heic_paths(client, auth):
    from PIL import Image

    image = Image.open(FIXTURE_IMAGE)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    response = post_image(client, auth, filename="sheet.png", payload=buffer.getvalue())
    assert response.status_code == 200
    assert response.get_json()["filename"] == "sheet.png"

    big = image.resize((image.width * 2, image.height * 2))
    buffer = io.BytesIO()
    big.save(buffer, format="JPEG")
    response = post_image(client, auth, filename="big.jpg", payload=buffer.getvalue())
    assert response.status_code == 200

    try:
        import pillow_heif

        heif = pillow_heif.from_pillow(image)
        buffer = io.BytesIO()
        heif.save(buffer, quality=80)
    except Exception:
        pytest.skip("HEIF encoder not available")
    response = post_image(client, auth, filename="phone.heic", payload=buffer.getvalue())
    assert response.status_code == 200
    assert response.get_json()["filename"] == "phone.jpg"


def test_process_pdf(client, auth):
    import pymupdf

    doc = pymupdf.open()
    for _ in range(2):
        page = doc.new_page(width=400, height=500)
        page.insert_image(page.rect, filename=str(FIXTURE_IMAGE))
    pdf_bytes = doc.tobytes()
    doc.close()

    response = post_image(client, auth, {"prompt": "SLTPvM_skeletal.yaml"}, filename="two.pdf", payload=pdf_bytes)
    assert response.status_code == 200
    body = response.get_json()
    assert list(body) == ["source_pdf", "page_count", "host", "pages", "total_request_cost_usd", "engine_notice"]
    assert body["source_pdf"] == "two.pdf"
    assert body["page_count"] == 2
    assert [p["filename"] for p in body["pages"]] == ["two__page_0001.jpg", "two__page_0002.jpg"]
    assert list(body["pages"][0]["formatted_json"]) == rules_of("SLTPvM_skeletal.yaml")
    assert body["engine_notice"] == DEMO_NOTICE

    bad = post_image(client, auth, filename="broken.pdf", payload=b"%PDF-1.4 nonsense")
    assert bad.status_code == 400


def test_upload_size_cap(make_app, auth):
    flask_app = make_app()
    flask_app.config["MAX_CONTENT_LENGTH"] = 1024
    response = post_image(flask_app.test_client(), auth)
    assert response.status_code == 413
    assert response.get_json()["error_code"] == "PAYLOAD_TOO_LARGE"


def test_elevation_enrichment_called_only_when_enabled(client, auth, elevation_spy, monkeypatch):
    from vvl.engines import mock

    record = dict(mock.SAMPLE_RECORD, decimalLatitude="38.95", decimalLongitude="-81.70")
    monkeypatch.setattr(mock, "SAMPLE_RECORD", record)
    elevation_spy.value = 1234
    body = post_image(client, auth, {"prompt": "SLTPvM_default.yaml"}).get_json()
    assert body["COP90_elevation_m"] == 1234
    assert len(elevation_spy.calls) == 1

    body = post_image(client, auth, {"prompt": "SLTPvM_default.yaml", "include_cop90": "false"}).get_json()
    assert body["COP90_elevation_m"] == ""
    client.put("/v1/settings", headers=auth, json={"enrichment": {"elevation": False}})
    body = post_image(client, auth, {"prompt": "SLTPvM_default.yaml"}).get_json()
    assert body["COP90_elevation_m"] == ""
    assert len(elevation_spy.calls) == 1


# ------------------------------------------------------------ process-url --

@pytest.mark.parametrize("url,code,status", [
    ("http://127.0.0.1:8642/x.jpg", "URL_BLOCKED", 403),
    ("http://localhost/x.jpg", "URL_BLOCKED", 403),
    ("http://10.0.0.1/x.jpg", "URL_BLOCKED", 403),
    ("file:///etc/passwd", "URL_INVALID", 400),
    ("not a url", "URL_INVALID", 400),
    ("https://user:pw@example.org/x.jpg", None, None),
])
def test_process_url_rejects_unsafe_urls(client, auth, url, code, status):
    response = client.post("/process-url", headers=auth, json={"image_url": url})
    body = response.get_json()
    if code:
        assert response.status_code == status, body
        assert body["error_code"] == code
    else:
        assert response.status_code in (400, 403)
    assert body["details"]["url_source"] == url


def test_process_url_input_errors(client, auth):
    assert client.post("/process-url", headers=auth, json={}).status_code == 400
    assert client.post("/process-url", headers=auth, data={"x": "1"}).status_code == 400
    response = client.post("/process-url", headers=auth, json={"image_url": "https://example.org/a.jpg",
                                                               "engines": ["local:qwen3-vl-8b-q8"]})
    assert response.status_code == 409


def test_process_url_success_with_stubbed_fetch(client, auth, monkeypatch):
    from werkzeug.datastructures import FileStorage

    import app as app_module

    monkeypatch.setattr(app_module, "check_url_destination", lambda url: None)
    closed = []

    class Tracked(io.BytesIO):
        def close(self):
            closed.append(True)
            super().close()

    def fake_fetch(url, **kwargs):
        return FileStorage(stream=Tracked(FIXTURE_IMAGE.read_bytes()), filename="remote.jpg",
                           content_type="image/jpeg"), "remote.jpg"

    monkeypatch.setattr(app_module, "fetch_image", fake_fetch)
    response = client.post("/process-url", headers=auth,
                           json={"image_url": "https://images.example.org/remote.jpg", "prompt": "SLTPvM_skeletal.yaml"})
    assert response.status_code == 200
    body = response.get_json()
    assert body["url_source"] == "https://images.example.org/remote.jpg"
    assert body["filename"] == "remote.jpg"
    assert list(body["formatted_json"]) == rules_of("SLTPvM_skeletal.yaml")
    assert closed


def test_no_detector_uses_the_whole_image(make_app, auth):
    """Before the label detector is downloaded, an image is processed whole (no 503)."""
    from vvl.label_detector import DetectorNotInstalled

    def not_installed():
        raise DetectorNotInstalled("not downloaded yet")

    client = make_app(VVL_DETECTOR_FACTORY=not_installed).test_client()
    response = post_image(client, auth, {})
    assert response.status_code == 200
    body = response.get_json()
    assert body["success"]["text_collage"] == "False"
    assert body["collage_info"]["position_original"] == {}
    assert body["collage_info"]["base64image_text_collage"]


def test_detector_load_error_is_kept_until_reset(make_app):
    calls = []

    def broken():
        calls.append(1)
        raise RuntimeError("bad package")

    processor = make_app(VVL_DETECTOR_FACTORY=broken).extensions["vvl"]["processor"]
    assert processor.collage_engine is None and processor.collage_engine is None
    assert len(calls) == 1
    assert processor.collage_status() == {"loaded": False, "loaded_runtime": None, "load_error": "RuntimeError: bad package"}
