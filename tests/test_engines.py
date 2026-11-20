"""Engine seam: model id parsing, availability, factory registration, JSON schema."""

from __future__ import annotations

import pytest

from vvl import prompts
from vvl.engines import (
    ModelIdError,
    ModelNotAvailable,
    ParseRequest,
    ResolveContext,
    availability,
    json_schema_from_prompt,
    json_schema_from_rules,
    parse_model_id,
    register_factory,
    resolve_engine,
    unregister_factory,
)
from vvl.engines.mock import DEMO_NOTICE, SAMPLE_RECORD, MockEngine
from vvl.prompting import PromptBuilder, ocr_prompt_text, wrap_parse_prompt
from vvl.registry import loader


@pytest.fixture
def context():
    return ResolveContext(providers=loader.load_providers(), local_models=loader.load_local_models())


@pytest.mark.parametrize("model_id,source,name,provider", [
    ("demo:mock", "demo", "mock", None),
    ("local:qwen3-vl-8b-q8", "local", "qwen3-vl-8b-q8", None),
    ("api:gemini:gemini-3.1-flash-lite", "api", "gemini-3.1-flash-lite", "gemini"),
    ("api:openrouter:qwen/qwen3-vl-32b-instruct:free", "api", "qwen/qwen3-vl-32b-instruct:free", "openrouter"),
    ("api:openrouter:~anthropic/claude-opus-latest", "api", "~anthropic/claude-opus-latest", "openrouter"),
    (" demo:mock ", "demo", "mock", None),
])
def test_parse_model_id(model_id, source, name, provider):
    ref = parse_model_id(model_id)
    assert (ref.source, ref.name, ref.provider) == (source, name, provider)


@pytest.mark.parametrize("model_id", [
    "", None, "gemini-2.5-flash", "demo:other", "local:", "local:Bad Id", "api:gemini", "api::x",
    "api:gemini:", "cloud:x", "local:../../etc",
])
def test_parse_model_id_rejects(model_id):
    with pytest.raises(ModelIdError):
        parse_model_id(model_id)


def test_availability(context):
    assert availability("demo:mock", context) == (True, None)
    assert availability("local:qwen3-vl-8b-q8", context) == (
        False, "Download this model and install the llama.cpp engine first (Models and Engine tabs).")
    assert availability("api:openai:gpt-4o", context) == (False, "This provider is not available in this version yet. Use OpenRouter for hosted models.")
    with pytest.raises(ModelIdError):
        availability("local:not-a-build", context)
    with pytest.raises(ModelIdError):
        availability("api:vvgo:x", context)  # enrichment provider, not a model provider
    with pytest.raises(ModelIdError):
        availability("api:nobody:x", context)


def test_resolve(context):
    engine = resolve_engine("demo:mock", context)
    assert isinstance(engine, MockEngine)
    assert engine.notice == DEMO_NOTICE
    with pytest.raises(ModelNotAvailable) as excinfo:
        resolve_engine("local:qwen3-vl-2b-q8", context)
    assert excinfo.value.reason == "Download this model and install the llama.cpp engine first (Models and Engine tabs)."
    with pytest.raises(ModelNotAvailable):
        resolve_engine("api:anthropic:claude-x", context)


def test_register_factory_makes_a_source_runnable(context):
    made = []

    def factory(ref, ctx):
        made.append(ref.model_id)
        return MockEngine()

    register_factory("api", factory, kind="openai_compat")
    try:
        assert availability("api:openai:gpt-x", context) == (True, None)
        assert availability("api:gemini:g", context)[0] is False  # different kind
        resolve_engine("api:openai:gpt-x", context)
        assert made == ["api:openai:gpt-x"]
    finally:
        unregister_factory("api", kind="openai_compat")
    assert availability("api:openai:gpt-x", context)[0] is False


def test_mock_parse_uses_template_fields_only():
    template = {"catalogNumber": "", "SCIENTIFICNAME": "", "notAField": ""}
    result = MockEngine().parse(ParseRequest(prompt_text="x", template=template, json_schema={}))
    assert list(result.record) == ["catalogNumber", "SCIENTIFICNAME", "notAField"]
    assert result.record["catalogNumber"] == SAMPLE_RECORD["catalogNumber"]
    assert result.record["SCIENTIFICNAME"] == SAMPLE_RECORD["scientificName"]
    assert result.record["notAField"] == ""
    assert result.tokens_in == result.tokens_out == 0


# ------------------------------------------------------------- schema --

def test_schema_from_rules():
    schema = json_schema_from_rules({"b": "Second.", "a": None, "c": {"nested": 1}, "d": "  "}, title="T")
    assert schema["type"] == "object"
    assert list(schema["properties"]) == ["b", "a", "c", "d"]
    assert schema["required"] == ["b", "a", "c", "d"]
    assert schema["additionalProperties"] is False
    assert schema["title"] == "T"
    assert schema["properties"]["b"] == {"type": "string", "description": "Second."}
    assert schema["properties"]["a"] == {"type": "string"}
    assert schema["properties"]["c"]["description"] == '{"nested": 1}'
    assert schema["properties"]["d"] == {"type": "string"}


@pytest.mark.parametrize("rules", [None, {}, [], "x", {"": "empty name"}, {5: "number name"}])
def test_schema_rejects_bad_rules(rules):
    with pytest.raises(ValueError):
        json_schema_from_rules(rules)


def test_schema_for_every_builtin_prompt():
    import yaml

    for name in prompts.builtin_names():
        data = yaml.safe_load((prompts.source_dir("builtin") / name).read_text(encoding="utf-8"))
        schema = json_schema_from_prompt(data)
        assert schema["required"] == list(data["rules"])
        assert all(p["type"] == "string" for p in schema["properties"].values())


# -------------------------------------------------------- prompt text --

def test_parse_prompt_text_matches_vvgo():
    path = prompts.source_dir("builtin") / "SLTPvM_default.yaml"
    builder = PromptBuilder(path)
    assert list(builder.JSON_dict_structure) == list(builder.rules)
    assert all(v == "" for v in builder.JSON_dict_structure.values())
    text = builder.parse_prompt_text('Label: "Galax" 12:30')
    assert text.startswith("Answer the user query.\nReturn a JSON object.\nPlease help me complete this text parsing task")
    assert text.endswith("\n")
    # prompt_SLTP strips colons and double quotes from the OCR text.
    assert "Label Galax 1230" in text
    assert wrap_parse_prompt("Q") == "Answer the user query.\nReturn a JSON object.\nQ\n"
    assert builder.json_schema()["required"] == list(builder.rules)


def test_ocr_prompt_selection():
    key, text = ocr_prompt_text(None)
    assert key == "verbatim_with_annotations"
    assert text.startswith("Please perform OCR on this scientific image")
    assert ocr_prompt_text("verbatim_notebook")[0] == "verbatim_notebook"
    assert ocr_prompt_text("anything else")[0] == "verbatim_with_annotations"
