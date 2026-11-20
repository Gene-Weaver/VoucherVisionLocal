"""The generic OpenAI-compatible engine against a fake chat-completions server."""

from __future__ import annotations

import base64
import json
import socket
import threading

import pytest

from conftest import FIXTURE_IMAGE
from vvl.engines.base import EngineCapabilities, EngineError, OCRRequest, ParseRequest
from vvl.engines.openai_compat import OpenAICompatConfig, OpenAICompatEngine, StepOptions, parse_json_reply

pytestmark = pytest.mark.usefixtures("no_network")

PATH = "/v1/chat/completions"


def reply(content, *, finish="stop", usage=None):
    return 200, {"id": "x", "object": "chat.completion",
                 "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": content}}],
                 "usage": usage or {"prompt_tokens": 1500, "completion_tokens": 80, "total_tokens": 1580}}


def make_engine(server, **overrides):
    config = OpenAICompatConfig(
        base_url=server.base_url + "/v1", model="test-model", model_id="local:test-a", label="Test A",
        api_key="engine-key-123", ocr=StepOptions(max_tokens=3072, sampling={"temperature": 0.1, "top_k": 20}),
        parse=StepOptions(max_tokens=2048, sampling={"temperature": 0.0}), timeout_seconds=5, max_retries=2,
        retry_backoff_seconds=(0.0,))
    for key, value in overrides.items():
        setattr(config, key, value)
    return OpenAICompatEngine(config, sleep=lambda s: None)


def body(server, index=-1):
    return json.loads(server.bodies[index])


SCHEMA = {"type": "object", "properties": {"catalogNumber": {"type": "string"}, "genus": {"type": "string"}},
          "required": ["catalogNumber", "genus"], "additionalProperties": False}


def test_ocr_payload_shape(fake_server):
    fake_server.routes[PATH] = reply("Label text\n«handwritten»")
    engine = make_engine(fake_server)
    result = engine.ocr(OCRRequest(image_path=str(FIXTURE_IMAGE), prompt_text="Please OCR.", prompt_key="k"))
    assert result.text == "Label text\n«handwritten»"
    assert (result.tokens_in, result.tokens_out, result.thinking_tokens) == (1500, 80, 0)
    assert result.total_cost == 0 and result.model_id == "local:test-a"
    sent = body(fake_server)
    assert sent["model"] == "test-model" and sent["stream"] is False
    assert sent["max_tokens"] == 3072
    assert sent["temperature"] == 0.1 and sent["top_k"] == 20
    assert "response_format" not in sent
    content = sent["messages"][0]["content"]
    assert sent["messages"][0]["role"] == "user"
    assert content[0]["type"] == "image_url"
    url = content[0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == FIXTURE_IMAGE.read_bytes()
    assert content[1] == {"type": "text", "text": "Please OCR."}
    headers = fake_server.requests[-1]["headers"]
    assert headers["authorization"] == "Bearer engine-key-123"


def test_ocr_text_first_and_request_cap(fake_server):
    fake_server.routes[PATH] = reply("x")
    engine = make_engine(fake_server, ocr=StepOptions(max_tokens=3072, image_first=False))
    engine.ocr(OCRRequest(image_path=str(FIXTURE_IMAGE), prompt_text="p", prompt_key="k", max_output_tokens=1000))
    sent = body(fake_server)
    assert [part["type"] for part in sent["messages"][0]["content"]] == ["text", "image_url"]
    assert sent["max_tokens"] == 1000


def test_parse_sends_schema_and_returns_record(fake_server):
    fake_server.routes[PATH] = reply(json.dumps({"catalogNumber": "1122841", "genus": "Jouvea"}))
    engine = make_engine(fake_server)
    result = engine.parse(ParseRequest(prompt_text="Parse this.", template={"catalogNumber": "", "genus": ""},
                                       json_schema=SCHEMA))
    assert result.record == {"catalogNumber": "1122841", "genus": "Jouvea"}
    sent = body(fake_server)
    assert sent["messages"] == [{"role": "user", "content": "Parse this."}]
    assert sent["response_format"] == {"type": "json_schema",
                                       "json_schema": {"name": "vvl_record", "strict": True, "schema": SCHEMA}}
    assert sent["max_tokens"] == 2048 and sent["temperature"] == 0.0


def test_parse_without_structured_output(fake_server):
    fake_server.routes[PATH] = reply('{"genus": "Jouvea"}')
    engine = make_engine(fake_server, capabilities=EngineCapabilities(vision=True, structured_output=False))
    engine.parse(ParseRequest(prompt_text="p", template={}, json_schema=SCHEMA))
    assert "response_format" not in body(fake_server)


@pytest.mark.parametrize("content,expected", [
    ('```json\n{"genus": "Jouvea"}\n```', {"genus": "Jouvea"}),
    ('Here you go: {"genus": "Jouvea"} done', {"genus": "Jouvea"}),
    ('<think>hmm</think>{"genus": "Jouvea"}', {"genus": "Jouvea"}),
    ("Sorry, no JSON.", None),
    ('{"genus": "Jouv', None),
    ("[1, 2]", None),
])
def test_malformed_replies(fake_server, content, expected):
    fake_server.routes[PATH] = reply(content, finish="length" if expected is None else "stop")
    result = make_engine(fake_server).parse(ParseRequest(prompt_text="p", template={}, json_schema=SCHEMA))
    assert result.record == expected
    assert result.raw_text


def test_parse_json_reply_helper():
    assert parse_json_reply("") is None
    assert parse_json_reply('{"a": "b"}') == {"a": "b"}


def test_retry_on_transient_status(fake_server):
    fake_server.routes[PATH] = [(503, {"error": {"message": "Loading model"}}), (502, {}), reply("ok")]
    result = make_engine(fake_server).ocr(OCRRequest(image_path=str(FIXTURE_IMAGE), prompt_text="p", prompt_key="k"))
    assert result.text == "ok"
    assert len([r for r in fake_server.requests if r["path"] == PATH]) == 3


def test_retry_is_bounded(fake_server):
    fake_server.routes[PATH] = (503, {"error": {"code": 503, "message": "busy"}})
    with pytest.raises(EngineError) as excinfo:
        make_engine(fake_server).ocr(OCRRequest(image_path=str(FIXTURE_IMAGE), prompt_text="p", prompt_key="k"))
    assert excinfo.value.code == "ENGINE_ERROR" and excinfo.value.status == 502
    assert "HTTP 503" in str(excinfo.value) and "busy" in str(excinfo.value)
    assert len([r for r in fake_server.requests if r["path"] == PATH]) == 3   # 1 + 2 retries


def test_client_errors_are_not_retried(fake_server):
    fake_server.routes[PATH] = (400, {"error": {"code": 400, "type": "exceed_context_size_error",
                                                "message": "the request exceeds the available context size",
                                                "n_prompt_tokens": 9000, "n_ctx": 8192}})
    with pytest.raises(EngineError) as excinfo:
        make_engine(fake_server).ocr(OCRRequest(image_path=str(FIXTURE_IMAGE), prompt_text="p", prompt_key="k"))
    message = str(excinfo.value)
    assert "context window" in message and "9000 tokens needed, 8192 available" in message
    assert len([r for r in fake_server.requests if r["path"] == PATH]) == 1


def test_reply_without_choices(fake_server):
    fake_server.routes[PATH] = (200, {"choices": []})
    with pytest.raises(EngineError, match="without any choices"):
        make_engine(fake_server).parse(ParseRequest(prompt_text="p", template={}, json_schema=SCHEMA))


def test_connection_refused_retries_then_fails():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = OpenAICompatConfig(base_url=f"http://127.0.0.1:{port}/v1", model="m", model_id="local:x", label="X",
                                max_retries=2, retry_backoff_seconds=(0.0,), timeout_seconds=2,
                                service_label="The local engine")
    calls = []
    engine = OpenAICompatEngine(config, sleep=lambda s: calls.append(s))
    with pytest.raises(EngineError, match="Could not reach the local engine"):
        engine.parse(ParseRequest(prompt_text="p", template={}, json_schema=SCHEMA))
    assert len(calls) == 2


def test_read_timeout_is_not_retried():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(5)
    accepted = []

    def accept():
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            accepted.append(conn)

    threading.Thread(target=accept, daemon=True).start()
    try:
        config = OpenAICompatConfig(base_url=f"http://127.0.0.1:{listener.getsockname()[1]}/v1", model="m",
                                    model_id="local:x", label="X", timeout_seconds=0.5, max_retries=2,
                                    retry_backoff_seconds=(0.0,))
        with pytest.raises(EngineError, match="did not answer within"):
            OpenAICompatEngine(config, sleep=lambda s: None).parse(
                ParseRequest(prompt_text="p", template={}, json_schema=SCHEMA))
        assert len(accepted) == 1
    finally:
        listener.close()
        for conn in accepted:
            conn.close()


def test_usage_with_reasoning_tokens(fake_server):
    fake_server.routes[PATH] = reply("text", usage={"prompt_tokens": 10, "completion_tokens": 50,
                                                    "completion_tokens_details": {"reasoning_tokens": 30}})
    result = make_engine(fake_server).ocr(OCRRequest(image_path=str(FIXTURE_IMAGE), prompt_text="p", prompt_key="k"))
    assert (result.tokens_in, result.tokens_out, result.thinking_tokens) == (10, 50, 30)


def test_extra_body_and_headers_for_hosted_apis(fake_server):
    fake_server.routes[PATH] = reply("x")
    engine = make_engine(fake_server, extra_headers={"X-Org": "lab"}, extra_body={"seed": 7})
    engine.ocr(OCRRequest(image_path=str(FIXTURE_IMAGE), prompt_text="p", prompt_key="k"))
    assert body(fake_server)["seed"] == 7
    assert fake_server.requests[-1]["headers"]["x-org"] == "lab"


def test_png_mime(tmp_path, fake_server):
    from PIL import Image

    path = tmp_path / "x.jpg"            # wrong extension on purpose: the bytes decide
    Image.new("RGB", (8, 8), "white").save(path, format="PNG")
    fake_server.routes[PATH] = reply("x")
    make_engine(fake_server).ocr(OCRRequest(image_path=str(path), prompt_text="p", prompt_key="k"))
    assert body(fake_server)["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
