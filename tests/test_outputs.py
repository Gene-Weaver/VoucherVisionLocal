"""Saved results (vvl/outputs.py): the folder per run, its files, and the spreadsheet guards."""

from __future__ import annotations

import base64
import io
import json
import re
from collections import OrderedDict

import pytest
from openpyxl import load_workbook

from conftest import FIXTURE_IMAGE
from vvl import outputs

pytestmark = pytest.mark.usefixtures("no_network")

STAMP = r"\d{8}-\d{2}-\d{2}-\d{2}"


def post_image(client, auth, data=None, filename="MICH_16205594.jpg"):
    body = dict(data or {})
    body["file"] = (io.BytesIO(FIXTURE_IMAGE.read_bytes()), filename)
    return client.post("/process", headers=auth, data=body, content_type="multipart/form-data")


def start(client, auth, **body):
    response = client.post("/v1/output/batches", headers=auth, json=body)
    assert response.status_code == 201, response.get_json()
    return response.get_json()


def finish(client, auth, batch):
    response = client.post(f"/v1/output/batches/{batch['id']}/finish", headers=auth)
    assert response.status_code == 200
    return response.get_json()


def sheet_rows(path):
    sheet = load_workbook(path)["results"]
    return [[cell.value for cell in row] for row in sheet.iter_rows()], sheet


def is_text(cell):
    """A text cell with the Text format (an empty one is stored as an inline string)."""
    return cell.number_format == "@" and cell.data_type in ("s", "inlineStr") and (cell.value is None or isinstance(cell.value, str))


# ------------------------------------------------------------- the folder --

def test_default_folder_is_runs_in_the_data_folder(client, auth, vvl_env):
    body = client.get("/v1/output", headers=auth).get_json()
    assert body["dir"] == str(vvl_env["VVL_HOME"] / "runs") == body["default_dir"]
    assert body["is_default"] is True
    assert body["subfolders"] == ["json", "xlsx", "label_collage", "input_image", "ocr_txt"]
    assert client.get("/v1/settings", headers=auth).get_json()["output_dir"] == body["dir"]


def test_output_dir_setting(client, auth, tmp_path):
    chosen = tmp_path / "my results"
    saved = client.put("/v1/settings", headers=auth, json={"output_dir": str(chosen)}).get_json()
    assert saved["output_dir"] == str(chosen)
    assert client.get("/v1/output", headers=auth).get_json()["is_default"] is False
    assert client.put("/v1/settings", headers=auth, json={"output_dir": "relative/path"}).status_code == 400
    back = client.put("/v1/settings", headers=auth, json={"output_dir": None}).get_json()
    assert back["output_dir"].endswith("runs")


def test_each_run_gets_a_new_timestamped_folder(client, auth, vvl_env):
    first = start(client, auth)
    second = start(client, auth)
    assert re.fullmatch(STAMP, first["folder"])
    assert first["path"] != second["path"]          # same second: the second one gets "-2"
    assert re.fullmatch(STAMP + r"(-\d+)?", second["folder"])
    named = start(client, auth, name="Herbarium A")
    assert re.fullmatch("Herbarium A_" + STAMP, named["folder"])
    for batch in (first, second, named):
        for sub in outputs.SUBDIRS:
            assert (vvl_env["VVL_HOME"] / "runs" / batch["folder"] / sub).is_dir()


def test_adding_to_a_folder_uses_the_name_as_it_is(client, auth, vvl_env):
    batch = start(client, auth, name="Herbarium A", append=True)
    assert batch["folder"] == "Herbarium A" and batch["existed"] is False and batch["append"] is True
    again = start(client, auth, name="Herbarium A", append=True)
    assert again["path"] == batch["path"] and again["existed"] is True
    response = client.post("/v1/output/batches", headers=auth, json={"append": True})
    assert response.status_code == 400 and "Name the folder" in response.get_json()["error"]


@pytest.mark.parametrize("name", ["a/b", "..", "a\\b", "x:y", "bad?", "CON", "trailing.", ".hidden", "n" * 101, 7])
def test_bad_folder_names_are_refused(client, auth, name):
    response = client.post("/v1/output/batches", headers=auth, json={"name": name})
    assert response.status_code == 400
    assert response.get_json()["error_code"] == "INVALID_INPUT"


def test_unknown_batch_is_404_and_nothing_runs(client, auth):
    response = post_image(client, auth, {"output_batch": "x" * 24})
    assert response.status_code == 404
    assert response.get_json()["error_code"] == "OUTPUT_BATCH_NOT_FOUND"


def test_a_folder_that_cannot_be_made(client, auth, tmp_path):
    blocker = tmp_path / "a-file"
    blocker.write_text("x")
    client.put("/v1/settings", headers=auth, json={"output_dir": str(blocker / "below")})
    response = client.post("/v1/output/batches", headers=auth, json={})
    assert response.status_code == 400
    assert response.get_json()["error_code"] == "OUTPUT_DIR_UNAVAILABLE"


# -------------------------------------------------------------- the files --

def test_a_run_saves_all_five_kinds_of_files(client, auth):
    batch = start(client, auth, name="demo")
    response = post_image(client, auth, {"output_batch": batch["id"]})
    assert response.status_code == 200
    result = response.get_json()
    summary = finish(client, auth, batch)
    assert summary["errors"] == []
    assert summary["saved"] == {"json": 1, "xlsx": 1, "label_collage": 1, "input_image": 1, "ocr_txt": 1}

    from pathlib import Path
    folder = Path(batch["path"])
    saved = json.loads((folder / "json" / "MICH_16205594.json").read_text(encoding="utf-8"))
    assert saved == result                                              # the result, as the API returned it
    assert list(saved) == list(result)
    assert (folder / "ocr_txt" / "MICH_16205594.txt").read_text(encoding="utf-8") == result["ocr"]
    collage = (folder / "label_collage" / "MICH_16205594.jpg").read_bytes()
    assert collage == base64.b64decode(result["collage_info"]["base64image_text_collage"])
    assert collage[:2] == b"\xff\xd8"
    assert (folder / "input_image" / "MICH_16205594.jpg").read_bytes()[:2] == b"\xff\xd8"

    rows, sheet = sheet_rows(folder / "xlsx" / "results.xlsx")
    assert rows[0] == ["filename", *result["formatted_json"].keys()]
    assert rows[1][0] == "MICH_16205594"
    # Empty cells read back as None; everything else is the record's value as text.
    assert [v or "" for v in rows[1][1:]] == ["" if v is None else str(v) for v in result["formatted_json"].values()]
    assert all(is_text(cell) for row in sheet.iter_rows() for cell in row)


def test_without_output_batch_nothing_is_saved(client, auth, vvl_env):
    assert post_image(client, auth).status_code == 200
    assert not (vvl_env["VVL_HOME"] / "runs").exists()


def test_same_file_name_twice_in_one_run_keeps_both(client, auth):
    from pathlib import Path
    batch = start(client, auth)
    for _ in range(2):
        assert post_image(client, auth, {"output_batch": batch["id"]}).status_code == 200
    summary = finish(client, auth, batch)
    names = sorted(p.name for p in (Path(batch["path"]) / "json").iterdir())
    assert names == ["MICH_16205594.json", "MICH_16205594_2.json"]
    assert summary["saved"]["xlsx"] == 2


def test_ocr_only_saves_files_but_no_spreadsheet_row(client, auth):
    from pathlib import Path
    batch = start(client, auth)
    assert post_image(client, auth, {"output_batch": batch["id"], "ocr_only": "true"}).status_code == 200
    summary = finish(client, auth, batch)
    assert summary["saved"]["json"] == 1 and summary["saved"]["ocr_txt"] == 1 and summary["saved"]["xlsx"] == 0
    assert not (Path(batch["path"]) / "xlsx" / "results.xlsx").exists()


def test_two_pass_run_saves_images_in_the_first_pass_and_the_record_in_the_second(client, auth):
    from pathlib import Path
    batch = start(client, auth)
    first = post_image(client, auth, {"output_batch": batch["id"], "stage": "ocr"}).get_json()
    assert first["staged"] is True
    folder = Path(batch["path"])
    assert (folder / "input_image" / "MICH_16205594.jpg").is_file()
    assert (folder / "label_collage" / "MICH_16205594.jpg").is_file()
    assert not (folder / "json" / "MICH_16205594.json").exists()
    done = client.post("/v1/process/finish", headers=auth,
                       json={"stage_token": first["stage_token"], "output_batch": batch["id"]})
    assert done.status_code == 200
    summary = finish(client, auth, batch)
    assert summary["saved"] == {"json": 1, "xlsx": 1, "label_collage": 1, "input_image": 1, "ocr_txt": 1}
    assert json.loads((folder / "json" / "MICH_16205594.json").read_text(encoding="utf-8")) == done.get_json()


def test_adding_to_a_folder_safely_preserves_existing_and_can_optionally_replace(client, auth):
    from pathlib import Path
    first = start(client, auth, name="set", append=True)
    assert post_image(client, auth, {"output_batch": first["id"]}, filename="a.jpg").status_code == 200
    assert post_image(client, auth, {"output_batch": first["id"]}, filename="b.jpg").status_code == 200
    finish(client, auth, first)

    # Default append is safe: duplicate filenames are auto-numbered instead of overwritten.
    safe_second = start(client, auth, name="set", append=True)
    assert safe_second["replace_existing"] is False
    assert post_image(client, auth, {"output_batch": safe_second["id"]}, filename="b.jpg").status_code == 200
    assert post_image(client, auth, {"output_batch": safe_second["id"]}, filename="c.jpg").status_code == 200
    safe_summary = finish(client, auth, safe_second)
    folder = Path(safe_second["path"])
    assert sorted(p.name for p in (folder / "json").iterdir()) == ["a.json", "b.json", "b_2.json", "c.json"]
    rows, _ = sheet_rows(folder / "xlsx" / "results.xlsx")
    assert [r[0] for r in rows] == ["filename", "a", "b", "b_2", "c"]
    assert safe_summary["saved"]["xlsx"] == 4

    # Explicit replace_existing=True overwrites same-named files while keeping other rows.
    replace_third = start(client, auth, name="set", append=True, replace_existing=True)
    assert post_image(client, auth, {"output_batch": replace_third["id"]}, filename="b.jpg").status_code == 200
    replace_summary = finish(client, auth, replace_third)
    assert sorted(p.name for p in (folder / "json").iterdir()) == ["a.json", "b.json", "b_2.json", "c.json"]
    assert replace_summary["saved"]["xlsx"] == 4


def test_optional_output_types_keep_json_required(client, auth):
    from pathlib import Path

    # Turn off all optional outputs: only json/ is created and written.
    batch = start(client, auth, name="json_only", save={
        "xlsx": False, "label_collage": False, "input_image": False, "ocr_txt": False,
    })
    assert batch["enabled"] == ["json"]
    assert post_image(client, auth, {"output_batch": batch["id"]}).status_code == 200
    summary = finish(client, auth, batch)
    assert summary["saved"] == {"json": 1, "xlsx": 0, "label_collage": 0, "input_image": 0, "ocr_txt": 0}
    folder = Path(batch["path"])
    assert (folder / "json" / "MICH_16205594.json").is_file()
    for skipped in ("xlsx", "label_collage", "input_image", "ocr_txt"):
        assert not (folder / skipped).exists()

    # Settings output_types apply as default when batch omits `save`.
    upd = client.put("/v1/settings", headers=auth, json={"output_types": {"input_image": False, "ocr_txt": False}})
    assert upd.status_code == 200
    assert upd.get_json()["output_types"] == {"xlsx": True, "label_collage": True, "input_image": False, "ocr_txt": False}
    b2 = start(client, auth, name="from_settings")
    assert b2["enabled"] == ["json", "xlsx", "label_collage"]
    assert post_image(client, auth, {"output_batch": b2["id"]}).status_code == 200
    s2 = finish(client, auth, b2)
    assert s2["saved"] == {"json": 1, "xlsx": 1, "label_collage": 1, "input_image": 0, "ocr_txt": 0}


# ------------------------------------------------------- spreadsheet guards --

def test_spreadsheet_guards(tmp_path):
    rows = OrderedDict([
        ("first", OrderedDict([
            ("formula", "=HYPERLINK(\"http://x\")"), ("plus", "+1+1"), ("at", "@SUM(A1)"), ("minus", "-12.5"),
            ("number", 1250), ("decimal", 42.5), ("none", None), ("flag", True),
            ("list", ["a", 2, "c"]), ("dict", {"b": 1, "a": 2}),
            ("control", "bad\x00\x07char\nnew\tline"), ("long", "x" * 40000),
            ("date_like", "1/2"), ("leading_zero", "00123"), ("padded", "   =1+1"),
        ])),
        ("second", OrderedDict([("extra", "only here"), ("formula", "plain")])),
    ])
    path = tmp_path / "results.xlsx"
    outputs.write_xlsx(path, rows)
    data, sheet = sheet_rows(path)
    assert sheet.title == "results"
    header = data[0]
    assert header[0] == "filename" and header[-1] == "extra"            # union of fields, first seen first
    first = dict(zip(header, data[1]))
    second = dict(zip(header, data[2]))
    assert first["filename"] == "first"
    assert first["formula"] == "'=HYPERLINK(\"http://x\")"              # formula injection
    assert first["plus"] == "'+1+1" and first["at"] == "'@SUM(A1)"
    assert first["minus"] == "-12.5"                                    # VoucherVisionGO keeps "-" (coordinates)
    assert first["padded"] == "   '=1+1"
    assert first["number"] == "1250" and first["decimal"] == "42.5" and first["flag"] == "True"
    assert first["none"] in ("", None)
    assert first["list"] == "a | 2 | c"
    assert first["dict"] == '{"a": 2, "b": 1}'
    assert first["control"] == "bad  char new line"                     # XML-illegal characters and line breaks
    assert len(first["long"]) == 32767 and first["long"].endswith("…")  # Excel's cell limit
    assert first["date_like"] == "1/2" and first["leading_zero"] == "00123"
    assert second["extra"] == "only here" and second["number"] in ("", None)
    for row in sheet.iter_rows():
        for cell in row:
            assert is_text(cell)                                        # text cells: Excel converts nothing
    assert not [p for p in tmp_path.iterdir() if p.name != "results.xlsx"]   # no temp file left


def test_spreadsheet_that_cannot_be_replaced_goes_to_a_second_file(tmp_path, monkeypatch):
    store = outputs.OutputStore()
    batch = store.create(tmp_path, name="locked")
    real = outputs.write_xlsx

    def locked(path, rows):
        if path.name == outputs.XLSX_NAME:
            raise PermissionError("in use")
        return real(path, rows)

    monkeypatch.setattr(outputs, "write_xlsx", locked)
    batch.save_result("one", {"ocr": "text", "formatted_json": {"catalogNumber": "1"}})
    summary = batch.finish()
    others = [p.name for p in (batch.path / "xlsx").iterdir()]
    assert len(others) == 1 and re.fullmatch(r"results_" + STAMP + r"\.xlsx", others[0])
    assert summary["saved"]["xlsx"] == 1
    assert any("could not be replaced" in e for e in summary["errors"])


def test_safe_stem():
    assert outputs.safe_stem("MICH-V-1122841.jpg") == "MICH-V-1122841"
    assert outputs.safe_stem("../../etc/passwd") == "passwd"
    assert outputs.safe_stem("C:\\Users\\x\\a b.tiff") == "a b"
    assert outputs.safe_stem('we?ird:na*me.png') == "we_ird_na_me"
    assert outputs.safe_stem("") == "image" and outputs.safe_stem("...") == "image"
    assert outputs.safe_stem("CON.jpg") == "image_CON"
    assert outputs.safe_stem("Señorita_ñ.jpg") == "Señorita_ñ"
