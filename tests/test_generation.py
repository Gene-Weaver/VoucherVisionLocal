"""Per-step model settings (vvl/generation.py, model_settings.json) and GET/PUT /v1/generation."""

from __future__ import annotations

import struct

import pytest
import yaml

from vvl import generation, settings


def chosen(ocr=None, parse=None, context=None, selection=None, engine=None):
    base = {"selection": selection or {"ocr": ["local:qwen3.6-27b-q8"], "parse": "local:qwen3.6-27b-q8"},
            "generation": {"ocr": {"reasoning": "off", "output_length": "standard", "image_detail": "standard"},
                           "parse": {"reasoning": "off", "output_length": "standard"}, "context": {}}}
    base["generation"]["ocr"].update(ocr or {})
    base["generation"]["parse"].update(parse or {})
    base["generation"]["context"].update(context or {})
    if engine:
        base["engine"] = engine
    return base


def test_data_file_covers_every_family_with_builds():
    from vvl.registry.loader import load_local_models

    data = generation.load()
    registry = load_local_models()
    for family in registry.families:
        for build in family.builds:
            assert generation.entry_for(f"local:{build.id}", data, registry), build.id


def test_qwen_reasoning_switch_and_card_sampling():
    off = generation.step_request("local:qwen3.6-27b-q8", "ocr", chosen())
    assert off["reasoning"] == "off"
    assert off["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert off["sampling"]["temperature"] == 0.7 and off["sampling"]["top_p"] == 0.8
    assert "repeat_penalty" in off["sampling"] and "repetition_penalty" not in off["sampling"], \
        "card names are translated to llama-server's names"
    assert off["max_tokens"] == 3072 and off["prompt_suffix"] == ""
    on = generation.step_request("local:qwen3.6-27b-q8", "parse", chosen(parse={"reasoning": "on", "output_length": "long"}))
    assert on["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert on["sampling"]["temperature"] == 1.0 and on["max_tokens"] == 16384


def test_each_family_maps_off_its_own_way():
    muse = generation.step_request("local:muse-glimmer-30b-q8", "ocr", chosen())
    assert muse["extra_body"] == {"chat_template_kwargs": {"reasoning_strength": "low"}}
    mimo_vl = generation.step_request("local:mimo-vl-7b-rl-2508-q8", "ocr", chosen())
    assert mimo_vl["prompt_suffix"] == " /no_think" and mimo_vl["extra_body"] == {}
    mimo_vl_on = generation.step_request("local:mimo-vl-7b-rl-2508-q8", "ocr", chosen(ocr={"reasoning": "on"}))
    assert mimo_vl_on["prompt_suffix"] == ""
    gemma = generation.step_request("local:gemma-4-12b-q4", "ocr", chosen())
    assert gemma["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}, \
        "Gemma's switch is always sent: the server would otherwise turn thinking on"
    q4 = generation.step_request("local:mimo-v2.6-distill-qwen-9b-q4", "ocr", chosen())
    assert q4["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}, "same_as entries resolve"
    vl = generation.step_request("local:qwen3-vl-32b-q8", "ocr", chosen(ocr={"reasoning": "on"}))
    assert vl["reasoning"] is None and vl["extra_body"] == {} and vl["max_tokens"] == 3072, \
        "a model that cannot reason ignores the reasoning choice"
    assert vl["sampling"] == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.0,
                              "presence_penalty": 1.5}


def test_image_detail_only_where_supported():
    high = chosen(ocr={"image_detail": "high"})
    assert generation.engine_params("local:qwen3-vl-32b-q8", high) == {"image_min_tokens": 1024, "image_max_tokens": 8192}
    assert generation.engine_params("local:qwen3-vl-32b-q8", chosen())["image_max_tokens"] == 4096
    assert generation.engine_params("local:gemma-4-12b-q4", high) == {}
    assert generation.engine_params("local:muse-glimmer-30b-q8", high) == {}


def test_context_rules():
    vl = "local:qwen3-vl-32b-q8"
    sel = {"ocr": [vl], "parse": vl}
    auto = generation.context_rule(vl, chosen(selection=sel), model_max=262144)
    assert auto == {"mode": "auto", "tokens": 8192, "needed": 8192, "max": 262144}
    long = generation.context_rule(vl, chosen(selection=sel, ocr={"output_length": "long"}), model_max=262144)
    # image 4096 + prompt 400 + OCR 8192; parsing 2000 + 8192 + 2048 -> 12288
    assert long["mode"] == "auto" and long["tokens"] == 13312
    high = generation.context_rule(vl, chosen(selection=sel, ocr={"output_length": "long", "image_detail": "high"}),
                                   model_max=262144)
    assert high["tokens"] == 17408
    thinking = generation.context_rule("local:qwen3.6-27b-q8", chosen(parse={"reasoning": "on"}), model_max=262144)
    assert thinking["mode"] == "auto_max" and thinking["tokens"] is None
    fixed = generation.context_rule(vl, chosen(selection=sel, context={vl: 400000}), model_max=131072)
    assert fixed["mode"] == "fixed" and fixed["tokens"] == 131072, "clamped to the model's maximum"
    engine_wide = generation.context_rule(vl, chosen(selection=sel, engine={"context_size": 20480}), model_max=262144)
    assert engine_wide["mode"] == "fixed" and engine_wide["tokens"] == 20480
    # A model that cannot reason never gets the "as large as fits" rule.
    assert generation.context_rule(vl, chosen(selection=sel, ocr={"reasoning": "on"}), model_max=262144)["mode"] == "auto"


def test_gguf_header_reads_scalars_and_skips_arrays(tmp_path):
    def s(text):
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    body = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 4)
    body += s("general.architecture") + struct.pack("<I", 8) + s("qwen35")
    body += s("tokenizer.ggml.tokens") + struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", 2) + s("a") + s("bb")
    body += s("qwen35.context_length") + struct.pack("<I", 4) + struct.pack("<I", 262144)
    body += s("qwen35.layers") + struct.pack("<I", 9) + struct.pack("<I", 5) + struct.pack("<Q", 3) + struct.pack("<3i", 1, 2, 3)
    path = tmp_path / "x.gguf"
    path.write_bytes(body)
    header = generation.gguf_header(path)
    assert header == {"general.architecture": "qwen35", "qwen35.context_length": 262144}


def test_settings_validation():
    with pytest.raises(settings.SettingsError):
        settings.validate_update({"generation": {"ocr": {"reasoning": "maybe"}}})
    with pytest.raises(settings.SettingsError):
        settings.validate_update({"generation": {"parse": {"image_detail": "high"}}})
    with pytest.raises(settings.SettingsError):
        settings.validate_update({"generation": {"context": {"local:x": 4096}}})
    with pytest.raises(settings.SettingsError):
        settings.validate_update({"generation": {"context": {"demo:mock": 16384}}})


# ------------------------------------------------------------------ route --

def test_generation_route(client, auth, vvl_env):
    vvl_env["VVL_SETTINGS_PATH"].write_text(yaml.safe_dump({
        "selection": {"ocr": ["local:qwen3-vl-32b-q8", "demo:mock"], "parse": "local:qwen3.6-27b-q8"}}))
    body = client.get("/v1/generation", headers=auth).get_json()
    assert set(body) == {"ocr", "parse", "context"}
    assert [m["model_id"] for m in body["ocr"]["models"]] == ["local:qwen3-vl-32b-q8"], "only local models"
    assert body["ocr"]["applicable"] == {"reasoning": False, "strength": False, "output_length": True, "image_detail": True}
    assert body["parse"]["applicable"] == {"reasoning": True, "strength": False, "output_length": True, "image_detail": False}
    parse = body["parse"]["models"][0]
    assert parse["controls"]["reasoning"]["literal"]["on"] == {"request": {"chat_template_kwargs": {"enable_thinking": True}}}
    assert parse["sent"]["chat_template_kwargs"] == {"enable_thinking": False} and parse["sent"]["max_tokens"] == 2048
    assert [c["model_id"] for c in body["context"]["models"]] == ["local:qwen3-vl-32b-q8", "local:qwen3.6-27b-q8"]
    assert body["context"]["min"] == 8192 and body["context"]["models"][0]["value"] == "auto"

    body = client.put("/v1/generation", headers=auth, json={
        "parse": {"reasoning": "on"}, "context": {"local:qwen3-vl-32b-q8": 32768}}).get_json()
    assert body["parse"]["values"]["reasoning"] == "on" and body["parse"]["values"]["output_length"] == "standard"
    assert body["parse"]["models"][0]["sent"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["parse"]["models"][0]["sent"]["max_tokens"] == 8192
    ctx = {c["model_id"]: c for c in body["context"]["models"]}
    assert ctx["local:qwen3-vl-32b-q8"]["value"] == 32768
    assert ctx["local:qwen3.6-27b-q8"]["mode"] == "auto_max", "reasoning on: as large as fits"
    saved = yaml.safe_load(vvl_env["VVL_SETTINGS_PATH"].read_text())
    assert saved["generation"]["context"] == {"local:qwen3-vl-32b-q8": 32768}

    body = client.put("/v1/generation", headers=auth, json={"context": {"local:qwen3-vl-32b-q8": "auto"}}).get_json()
    assert {c["model_id"]: c["value"] for c in body["context"]["models"]}["local:qwen3-vl-32b-q8"] == "auto"
    assert client.put("/v1/generation", headers=auth, json={"ocr": {"reasoning": "sideways"}}).status_code == 400
    assert client.put("/v1/generation", headers=auth, json={"surprise": 1}).status_code == 400
    assert client.get("/v1/generation").status_code == 401


def test_muse_strength_is_sent_while_reasoning_is_on():
    muse = "local:muse-glimmer-30b-q8"
    on = generation.step_request(muse, "ocr", chosen(ocr={"reasoning": "on", "strength": "medium"}))
    assert on["extra_body"] == {"chat_template_kwargs": {"reasoning_strength": "medium"}}
    assert on["strength"] == "medium"
    default = generation.step_request(muse, "parse", chosen(parse={"reasoning": "on"}))
    assert default["extra_body"] == {"chat_template_kwargs": {"reasoning_strength": "high"}}, \
        "no strength chosen: the model's default level"
    off = generation.step_request(muse, "ocr", chosen(ocr={"strength": "xhigh"}))
    assert off["extra_body"] == {"chat_template_kwargs": {"reasoning_strength": "low"}}, \
        "reasoning off sends the lowest level whatever strength is chosen"
    qwen = generation.step_request("local:qwen3.6-27b-q8", "ocr", chosen(ocr={"reasoning": "on", "strength": "low"}))
    assert qwen["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}} and qwen["strength"] is None


def test_a_model_that_cannot_stop_reasoning_gets_reasoning_token_limits():
    muse = "local:muse-glimmer-30b-q8"
    off = generation.step_request(muse, "ocr", chosen())
    assert off["reasoning"] == "off" and off["max_tokens"] == 8192, "Muse still reasons at low strength"
    assert generation.step_request(muse, "parse", chosen(parse={"output_length": "long"}))["max_tokens"] == 16384
    assert generation.step_request("local:qwen3.6-27b-q8", "ocr", chosen())["max_tokens"] == 3072
    assert generation.context_rule(muse, chosen(selection={"ocr": [muse], "parse": muse}))["mode"] == "auto", \
        "the context window only goes to the maximum when the user turns reasoning on"


def test_describe_lists_strength_only_for_models_with_levels():
    muse = generation.describe_step("local:muse-glimmer-30b-q8", "ocr", chosen())
    assert muse["controls"]["strength"]["options"] == ["low", "medium", "high", "xhigh"]
    assert muse["controls"]["reasoning"]["can_turn_off"] is False
    assert muse["controls"]["output_length"]["max_tokens"] == {"standard": 8192, "long": 16384}
    qwen = generation.describe_step("local:qwen3.6-27b-q8", "ocr", chosen())
    assert "strength" not in qwen["controls"] and qwen["controls"]["reasoning"]["can_turn_off"] is True


def test_settings_accept_only_known_strength_levels():
    ok = settings.validate_update({"generation": {"ocr": {"strength": "xhigh"}}})
    assert ok["generation"]["ocr"]["strength"] == "xhigh"
    with pytest.raises(settings.SettingsError):
        settings.validate_update({"generation": {"parse": {"strength": "max"}}})
