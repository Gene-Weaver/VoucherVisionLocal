"""OpenRouter: the catalog, role and cost rules, the engine, the request limit, and the routes.

Nothing here reaches openrouter.ai: the catalog and the chat endpoint are served
by the fake loopback server.
"""

from __future__ import annotations

import json
import threading
import time

import pytest
import yaml

from conftest import FIXTURE_IMAGE
from vvl import openrouter
from vvl.engines import ResolveContext, availability, resolve_engine
from vvl.engines.base import EngineError, ModelNotAvailable, ModelRef, OCRRequest, ParseRequest
from vvl.engines.openrouter import OpenRouterEngine
from vvl.registry import loader
from vvl.secrets.store import SecretStore

pytestmark = pytest.mark.usefixtures("no_network")

CHAT = "/api/v1/chat/completions"
MODELS = "/api/v1/models"
FAKE_KEY = "sk-or-v1-FAKE-0000000000000000"


def raw_model(slug, name, *, inputs=("text", "image"), outputs=("text",), prompt="0.000002", completion="0.000006",
              params=("max_tokens", "reasoning", "structured_outputs", "response_format"), created=1790000000,
              max_out=64000):
    return {"id": slug, "name": name, "created": created, "description": "A model. " * 100,
            "context_length": 200000,
            "architecture": {"input_modalities": list(inputs), "output_modalities": list(outputs)},
            "pricing": {"prompt": prompt, "completion": completion},
            "top_provider": {"max_completion_tokens": max_out}, "supported_parameters": list(params)}


RAW = {"data": [
    raw_model("anthropic/claude-test", "Anthropic: Claude Test"),
    raw_model("mistralai/text-only", "Mistral: Text Only", inputs=("text",), params=("max_tokens",)),
    raw_model("qwen/qwen-vl:free", "Qwen: Qwen VL (free)", prompt="0", completion="0", params=("max_tokens",)),
    raw_model("openai/image-maker", "OpenAI: Image Maker", outputs=("image", "text")),
    raw_model("openrouter/auto", "Auto Router", prompt="-1", completion="-1"),
    {"id": 7},
]}


ZDR = {"data": [
    {"model_id": "anthropic/claude-test", "provider_name": "Amazon Bedrock", "tag": "amazon-bedrock"},
    {"model_id": "anthropic/claude-test", "provider_name": "Google", "tag": "google-vertex/global"},
    {"model_id": "anthropic/claude-test", "provider_name": "Google", "tag": "google-vertex/global/flex"},
    {"model_id": "someone/else", "provider_name": "X"},
]}


@pytest.fixture(autouse=True)
def fresh_catalog(monkeypatch, vvl_env):
    """Each test gets its own catalog (cached under the test's VVL_HOME) and limiter."""
    monkeypatch.setattr(openrouter, "catalog", openrouter.Catalog())
    monkeypatch.setattr(openrouter, "limiter", openrouter.RequestLimiter())


def loaded_catalog():
    openrouter.catalog._set(openrouter.normalize_catalog(RAW), time.time())
    return openrouter.catalog


# ------------------------------------------------------------------ catalog --

def test_normalize_keeps_only_usable_models_and_sets_roles():
    models = {m["id"]: m for m in openrouter.normalize_catalog(RAW)}
    assert set(models) == {"anthropic/claude-test", "mistralai/text-only", "qwen/qwen-vl:free"}
    claude = models["anthropic/claude-test"]
    assert claude["roles"] == ["ocr", "parse"] and claude["vision"] is True
    assert (claude["owner"], claude["owner_label"], claude["name"]) == ("anthropic", "Anthropic", "Claude Test")
    assert (claude["price_in"], claude["price_out"]) == (2.0, 6.0)        # USD per 1,000,000 tokens
    assert claude["reasoning"] is True and claude["structured_output"] is True
    assert claude["model_id"] == "api:openrouter:anthropic/claude-test"
    assert len(claude["description"]) <= openrouter.DESCRIPTION_CHARS + 1
    text_only = models["mistralai/text-only"]
    assert text_only["roles"] == ["parse"] and text_only["cost_1k"]["ocr"] is None
    assert models["qwen/qwen-vl:free"]["free"] is True


def test_cost_per_thousand_uses_the_vvgo_calculator_tokens():
    # OCR typical: 1,300 in + 150 out; worst case: 1,500 in + 1,000 out.
    assert openrouter.estimate_per_thousand(2.0, 6.0, "ocr") == {"typical": 3.5, "high": 9.0}
    # Parsing typical: 2,350 in + 650 out; worst case: 3,000 in + 1,000 out.
    assert openrouter.estimate_per_thousand(2.0, 6.0, "parse") == {"typical": 8.6, "high": 12.0}


def test_catalog_fetches_caches_and_survives_a_failed_refresh(vvl_env):
    calls = []
    clock = {"t": 1000.0}

    def get(url, timeout):
        if url.endswith("/endpoints/zdr"):
            return ZDR
        calls.append(url)
        if len(calls) == 2:
            raise RuntimeError("HTTP 503")
        return RAW

    cat = openrouter.Catalog(now=lambda: clock["t"], http_get=get)
    snap = cat.snapshot("https://openrouter.ai/api/v1")
    assert calls == ["https://openrouter.ai/api/v1/models"]
    assert len(snap["models"]) == 3 and snap["error"] is None and snap["stale"] is False
    assert cat.snapshot("https://openrouter.ai/api/v1")["fetched_at"] == 1000.0 and len(calls) == 1   # fresh
    assert cat.cache_path().exists()

    again = openrouter.Catalog(now=lambda: clock["t"], http_get=get)             # a new process
    assert again.find("anthropic/claude-test")["vision"] is True and len(calls) == 1
    assert again.find("anthropic/claude-test")["private_providers"] == ["Amazon Bedrock", "Google"]
    assert again.find("mistralai/text-only")["private"] is False

    clock["t"] += openrouter.CATALOG_MAX_AGE_SECONDS + 1
    stale = again.snapshot("https://openrouter.ai/api/v1")                        # refresh fails
    assert len(calls) == 2 and len(stale["models"]) == 3
    assert stale["error"] == "OpenRouter's model list could not be read." and stale["stale"] is True


# ------------------------------------------------------------------- limiter --

@pytest.mark.parametrize("limit", [1, 2, 4])
def test_limiter_never_runs_more_than_the_limit(limit):
    limiter = openrouter.RequestLimiter()
    lock = threading.Lock()
    state = {"now": 0, "peak": 0}

    def work():
        with limiter.slot(limit):
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            time.sleep(0.02)
            with lock:
                state["now"] -= 1

    threads = [threading.Thread(target=work) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert state["peak"] == limit and limiter.status() == {"active": 0, "waiting": 0}


def test_limit_is_clamped_to_one_through_four():
    assert openrouter.settings_block({"openrouter": {"concurrency": 99}})["concurrency"] == 4
    assert openrouter.settings_block({"openrouter": {"concurrency": 0}})["concurrency"] == 1
    assert openrouter.settings_block({})["concurrency"] == 4
    assert openrouter.concurrency_for(["local:x", "api:openrouter:a/b"], {"openrouter": {"concurrency": 2}}) == 2
    assert openrouter.concurrency_for(["local:x", "demo:mock"], {}) is None


# -------------------------------------------------------------------- engine --

def reply(content, usage=None):
    return 200, {"id": "gen-1", "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
                 "usage": usage or {"prompt_tokens": 1300, "completion_tokens": 400,
                                    "completion_tokens_details": {"reasoning_tokens": 250}}}


def make_engine(server, slug="anthropic/claude-test", settings=None):
    model = loaded_catalog().find(slug)
    return OpenRouterEngine(ModelRef(openrouter.model_id_for(slug), "api", slug, provider="openrouter"),
                            api_key=FAKE_KEY, base_url=server.base_url + "/api/v1", model=model,
                            settings=settings, sleep=lambda s: None)


SCHEMA = {"type": "object", "properties": {"genus": {"type": "string"}}, "required": ["genus"],
          "additionalProperties": False}


def test_ocr_request_and_cost(fake_server):
    fake_server.routes[CHAT] = reply("Jouvea pilosa")
    engine = make_engine(fake_server)
    assert engine.capabilities.to_dict() == {"vision": True, "thinking_levels": ["low", "medium", "high"],
                                             "structured_output": True}
    result = engine.ocr(OCRRequest(image_path=str(FIXTURE_IMAGE), prompt_text="Please OCR.", prompt_key="k",
                                   thinking_level="medium"))
    assert result.text == "Jouvea pilosa" and result.model_id == "api:openrouter:anthropic/claude-test"
    sent = json.loads(fake_server.bodies[-1])
    assert sent["model"] == "anthropic/claude-test"
    assert sent["reasoning"] == {"effort": "medium", "exclude": True}
    assert sent["max_tokens"] == 16384 and "response_format" not in sent and "provider" not in sent
    assert sent["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert fake_server.requests[-1]["headers"]["authorization"] == f"Bearer {FAKE_KEY}"
    # 1,300 in at $2 and 400 out at $6 per million; 250 of the 400 were reasoning.
    assert result.cost_in == pytest.approx(0.0026)
    assert result.cost_out == pytest.approx(0.0009)
    assert result.thinking_cost == pytest.approx(0.0015)
    assert result.total_cost == pytest.approx(0.005)
    assert (result.rates_in, result.rates_out, result.thinking_budget_requested) == (2.0, 6.0, "medium")


def test_parse_sends_schema_and_falls_back_without_it(fake_server):
    fake_server.routes[CHAT] = [(404, {"error": {"code": 404, "message": "No endpoints found that support the requested parameters"}}),
                                reply('{"genus": "Jouvea"}')]
    engine = make_engine(fake_server)
    result = engine.parse(ParseRequest(prompt_text="Parse.", template={"genus": ""}, json_schema=SCHEMA))
    assert result.record == {"genus": "Jouvea"}
    first, second = (json.loads(b) for b in fake_server.bodies[-2:])
    assert first["response_format"]["json_schema"]["schema"] == SCHEMA
    assert first["provider"] == {"require_parameters": True}
    assert "response_format" not in second and "provider" not in second


def test_model_without_reasoning_or_schema_sends_neither(fake_server):
    fake_server.routes[CHAT] = reply('{"genus": "Jouvea"}', usage={"prompt_tokens": 10, "completion_tokens": 5})
    engine = make_engine(fake_server, "qwen/qwen-vl:free")
    assert engine.capabilities.thinking_levels == ()
    result = engine.parse(ParseRequest(prompt_text="Parse.", template={"genus": ""}, json_schema=SCHEMA))
    sent = json.loads(fake_server.bodies[-1])
    assert "reasoning" not in sent and "response_format" not in sent and sent["max_tokens"] == 8192
    assert result.cost_in + result.cost_out + result.thinking_cost == 0 and result.thinking_budget_requested == ""


def test_private_only_restricts_routing(fake_server):
    fake_server.routes[CHAT] = [reply("ok"), reply('{"genus": "Jouvea"}'),
                                (404, {"error": {"code": 404, "message": "No endpoints found matching your data policy"}})]
    engine = make_engine(fake_server, settings={"openrouter": {"private_only": True}})
    engine.ocr(OCRRequest(image_path=str(FIXTURE_IMAGE), prompt_text="x", prompt_key="k"))
    assert json.loads(fake_server.bodies[-1])["provider"] == {"zdr": True}
    engine.parse(ParseRequest(prompt_text="Parse.", template={"genus": ""}, json_schema=SCHEMA))
    assert json.loads(fake_server.bodies[-1])["provider"] == {"zdr": True, "require_parameters": True}
    with pytest.raises(EngineError) as excinfo:
        engine.ocr(OCRRequest(image_path=str(FIXTURE_IMAGE), prompt_text="x", prompt_key="k"))
    assert "No private provider offers this model" in str(excinfo.value)


def test_a_provider_can_be_pinned(fake_server):
    fake_server.routes[CHAT] = reply("ok")
    model = loaded_catalog().find("anthropic/claude-test")
    engine = OpenRouterEngine(ModelRef("api:openrouter:anthropic/claude-test", "api", "anthropic/claude-test",
                                       provider="openrouter"), api_key=FAKE_KEY,
                              base_url=fake_server.base_url + "/api/v1", model=model, settings=None,
                              provider={"only": ["amazon-bedrock"], "allow_fallbacks": False})
    engine.ocr(OCRRequest(image_path=str(FIXTURE_IMAGE), prompt_text="x", prompt_key="k"))
    assert json.loads(fake_server.bodies[-1])["provider"] == {"only": ["amazon-bedrock"], "allow_fallbacks": False}


def test_text_only_model_refuses_ocr(fake_server):
    engine = make_engine(fake_server, "mistralai/text-only")
    with pytest.raises(EngineError) as excinfo:
        engine.ocr(OCRRequest(image_path=str(FIXTURE_IMAGE), prompt_text="x", prompt_key="k"))
    assert excinfo.value.status == 409 and fake_server.bodies == []


@pytest.mark.parametrize("status,code,text", [
    (401, "UPSTREAM_AUTH_FAILED", "rejected the saved key"),
    (402, "UPSTREAM_QUOTA_EXHAUSTED", "enough credits"),
])
def test_account_errors_are_plain(fake_server, status, code, text):
    fake_server.routes[CHAT] = (status, {"error": {"code": status, "message": "raw upstream text"}})
    with pytest.raises(EngineError) as excinfo:
        make_engine(fake_server).ocr(OCRRequest(image_path=str(FIXTURE_IMAGE), prompt_text="x", prompt_key="k"))
    assert excinfo.value.code == code and text in str(excinfo.value) and FAKE_KEY not in str(excinfo.value)


def test_rate_limit_is_retried(fake_server):
    fake_server.routes[CHAT] = [(429, {"error": {"code": 429, "message": "slow down"}}), reply("ok")]
    assert make_engine(fake_server).ocr(
        OCRRequest(image_path=str(FIXTURE_IMAGE), prompt_text="x", prompt_key="k")).text == "ok"
    assert len(fake_server.bodies) == 2


def test_error_inside_a_200_reply_is_reported(fake_server):
    fake_server.routes[CHAT] = (200, {"error": {"code": 502, "message": "Provider returned error"}})
    with pytest.raises(EngineError) as excinfo:
        make_engine(fake_server).ocr(OCRRequest(image_path=str(FIXTURE_IMAGE), prompt_text="x", prompt_key="k"))
    assert "Provider returned error" in str(excinfo.value)


def test_engine_requests_respect_the_concurrency_setting(fake_server):
    fake_server.routes[CHAT] = reply("ok")
    engine = make_engine(fake_server, settings={"openrouter": {"concurrency": 2}})
    peak = {"value": 0}
    real_slot = openrouter.limiter.slot

    def watched(limit):
        assert limit == 2
        peak["value"] = max(peak["value"], openrouter.limiter.status()["active"] + 1)
        return real_slot(limit)

    engine._limiter.slot = watched
    request = OCRRequest(image_path=str(FIXTURE_IMAGE), prompt_text="x", prompt_key="k")
    threads = [threading.Thread(target=engine.ocr, args=(request,)) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert len(fake_server.bodies) == 6 and peak["value"] <= 3


# ----------------------------------------------------------- resolver + app --

def test_openrouter_needs_a_key(vvl_env):
    context = ResolveContext(providers=loader.load_providers(), local_models=loader.load_local_models(),
                             secrets=SecretStore())
    assert availability("api:openrouter:anthropic/claude-test", context) == (
        False, "Add an OpenRouter key first (Keys tab).")
    with pytest.raises(ModelNotAvailable):
        resolve_engine("api:openrouter:anthropic/claude-test", context)
    SecretStore().set_fields("openrouter", {"api_key": FAKE_KEY}, ["api_key"])
    assert availability("api:openrouter:anthropic/claude-test", context) == (True, None)
    assert isinstance(resolve_engine("api:openrouter:anthropic/claude-test", context), OpenRouterEngine)


@pytest.fixture
def or_app(make_app, fake_server, tmp_path):
    """The app with OpenRouter's base URL pointed at the fake server."""
    data = yaml.safe_load((loader.paths.registry_dir() / "providers.yaml").read_text(encoding="utf-8"))
    for provider in data["providers"]:
        if provider["id"] == "openrouter":
            provider["base_url"] = fake_server.base_url + "/api/v1"
    path = tmp_path / "providers.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    fake_server.routes[MODELS] = (200, RAW)
    fake_server.routes["/api/v1/endpoints/zdr"] = (200, ZDR)
    return make_app(VVL_PROVIDERS_PATH=str(path))


def test_catalog_route_and_adding_models(or_app, auth, fake_server):
    client = or_app.test_client()
    body = client.get("/v1/openrouter/catalog", headers=auth).get_json()
    assert [m["id"] for m in body["models"]] == ["anthropic/claude-test", "mistralai/text-only", "qwen/qwen-vl:free"]
    assert body["added"] == [] and body["configured"] is False and body["concurrency"] == 4
    assert body["private_only"] is False
    assert [m["private"] for m in body["models"]] == [True, False, False]
    assert body["models"][0]["private_providers"] == ["Amazon Bedrock", "Google"]
    assert body["owners"][0]["id"] in ("anthropic", "mistralai", "qwen") and body["concurrency_range"] == [1, 4]
    assert all("authorization" not in r["headers"] for r in fake_server.requests)   # the lists are public

    bad = client.put("/v1/openrouter/models", headers=auth, json={"add": ["nobody/nothing"]})
    assert bad.status_code == 400
    added = client.put("/v1/openrouter/models", headers=auth,
                       json={"add": ["anthropic/claude-test", "mistralai/text-only"]}).get_json()
    assert added["added"] == ["anthropic/claude-test", "mistralai/text-only"]

    models = client.get("/v1/models", headers=auth).get_json()
    entry = next(p for p in models["api"]["providers"] if p["id"] == "openrouter")
    assert entry["runnable"] is False and entry["unavailable_reason"] == "Add an OpenRouter key first (Keys tab)."
    ocr = [p["model_id"] for p in models["selectable"]["ocr"] if p["family_id"] == "openrouter"]
    parse = [p["model_id"] for p in models["selectable"]["parse"] if p["family_id"] == "openrouter"]
    assert ocr == ["api:openrouter:anthropic/claude-test"]                 # text-only models cannot do OCR
    assert parse == ["api:openrouter:anthropic/claude-test", "api:openrouter:mistralai/text-only"]

    # Text-only for OCR is refused; removing a chosen model resets the selection.
    refused = client.put("/v1/selection", headers=auth, json={"ocr": ["api:openrouter:mistralai/text-only"]})
    assert refused.status_code == 400
    ok = client.put("/v1/selection", headers=auth, json={"ocr": ["api:openrouter:anthropic/claude-test"],
                                                         "parse": "api:openrouter:mistralai/text-only"})
    assert ok.status_code == 200
    after = client.put("/v1/openrouter/models", headers=auth, json={"remove": ["mistralai/text-only"]}).get_json()
    assert after["added"] == ["anthropic/claude-test"]
    assert after["selection"] == {"ocr": ["api:openrouter:anthropic/claude-test"], "parse": "demo:mock"}


def test_concurrency_setting_is_one_to_four(or_app, auth):
    client = or_app.test_client()
    assert client.put("/v1/settings", headers=auth, json={"openrouter": {"concurrency": 5}}).status_code == 400
    saved = client.put("/v1/settings", headers=auth, json={"openrouter": {"concurrency": 2}})
    assert saved.status_code == 200
    assert client.get("/v1/openrouter/catalog", headers=auth).get_json()["concurrency"] == 2
    assert client.put("/v1/settings", headers=auth, json={"openrouter": {"private_only": "yes"}}).status_code == 400
    assert client.put("/v1/settings", headers=auth, json={"openrouter": {"private_only": True}}).status_code == 200
    assert client.get("/v1/openrouter/catalog", headers=auth).get_json()["private_only"] is True


def test_process_through_openrouter(or_app, auth, fake_server):
    client = or_app.test_client()
    client.get("/v1/openrouter/catalog", headers=auth)
    client.put("/v1/keys/openrouter", headers=auth, json={"fields": {"api_key": FAKE_KEY}})
    client.put("/v1/openrouter/models", headers=auth, json={"add": ["anthropic/claude-test"]})
    record = {"catalogNumber": "1122841", "scientificName": "Jouvea pilosa"}
    fake_server.routes[CHAT] = [reply("MICH 1122841 Jouvea pilosa"), reply(json.dumps(record))]
    with open(FIXTURE_IMAGE, "rb") as handle:
        response = client.post("/process", headers=auth, content_type="multipart/form-data", data={
            "file": (handle, "sheet.jpg"), "engines": "api:openrouter:anthropic/claude-test",
            "llm_model": "api:openrouter:anthropic/claude-test", "ocr_thinking_level": "high"})
    assert response.status_code == 200, response.get_data(as_text=True)[:500]
    body = response.get_json()
    assert body["formatted_json"]["scientificName"] == "Jouvea pilosa"
    info = body["ocr_info"]["api:openrouter:anthropic/claude-test"]
    assert info["ocr_text"] == "MICH 1122841 Jouvea pilosa" and info["total_cost"] > 0
    assert body["total_request_cost_usd"] > 0
    ocr_sent, parse_sent = (json.loads(b) for b in fake_server.bodies[-2:])
    assert ocr_sent["reasoning"]["effort"] == "high" and "response_format" in parse_sent
    assert FAKE_KEY not in response.get_data(as_text=True)
