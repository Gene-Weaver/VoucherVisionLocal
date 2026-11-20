"""Engine routes, engine settings, the catalog's Phase 2 fields, and /process with a
local model end to end (fake llama-server, fake model files)."""

from __future__ import annotations

import json
import os
import time

import pytest

from conftest import FIXTURE_IMAGE, free_port
from p2util import FakeModels, fake_engine_install, load_report, wait_until, write_settings

pytestmark = [pytest.mark.usefixtures("no_network"),
              pytest.mark.skipif(os.name == "nt", reason="the fake engine wrapper is a POSIX shell script")]


@pytest.fixture
def local(tmp_path, vvl_env, make_app, monkeypatch):
    monkeypatch.delenv("FAKE_LLAMA_MODE", raising=False)
    models_dir = tmp_path / "models_dir"
    state = tmp_path / "fake_state"
    state.mkdir()
    fake = FakeModels(tmp_path)
    fake.install_on_disk(models_dir, "test-a")
    fake.install_on_disk(models_dir, "test-b")
    fake_engine_install(models_dir, state_dir=state)
    write_settings(vvl_env["VVL_SETTINGS_PATH"], models_dir)
    base = free_port()
    report = load_report()
    app = make_app(VVL_LOCAL_MODELS_PATH=str(fake.registry_path), VVL_ENGINE_PORT_RANGE=(base, base + 30),
                   VVL_ENGINE_LOAD_TIMEOUT=15, VVL_SYSTEM_REPORT=lambda refresh=False: report)
    rt = app.extensions["vvl"]["runtime"]
    yield {"client": app.test_client(), "rt": rt, "fake": fake, "state": state, "models_dir": models_dir,
           "app": app}
    rt.shutdown(timeout=5)


def _requests(state):
    path = state / "requests.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def wait_engine(client, auth, *states, timeout=15):
    return wait_until(lambda: (lambda b: b if b["managed"]["state"] in states else None)(
        client.get("/v1/engine", headers=auth).get_json()), timeout)


def test_engine_start_logs_stop(local, auth):
    client = local["client"]
    body = client.get("/v1/engine", headers=auth).get_json()
    assert body["available"] is True and body["unavailable_reason"] is None
    assert body["managed"]["state"] == "stopped" and body["managed"]["installed"] is True
    assert body["managed"]["variant"] == {"id": "linux-x64-cuda12", "label": "Fake linux-x64-cuda12",
                                          "backend": "cuda"}
    assert [v["id"] for v in body["variants"]] == ["linux-x64-cuda13", "linux-x64-cuda12", "linux-x64-vulkan",
                                                   "linux-x64-cpu"]
    assert next(v for v in body["variants"] if v["id"] == "linux-x64-cuda12")["installed"] is True
    assert body["recommended_variant"]["id"] == "linux-x64-cuda12"
    response = client.post("/v1/engine/start", headers=auth, json={"model_id": "local:test-a"})
    assert response.status_code == 202
    assert response.get_json()["managed"]["state"] in ("starting", "loading_model", "ready")
    body = wait_engine(client, auth, "ready")
    managed = body["managed"]
    assert managed["loaded_model"] == "local:test-a" and managed["running"] is True
    # With automatic settings the pool runs one engine on each of the two cards.
    assert managed["gpu_devices"] == [0, 1] and managed["context_size"] == 8192   # the smallest context VVL uses
    assert [e["gpu_devices"] for e in managed["engines"]] == [[0], [1]] or \
        [e["gpu_devices"] for e in managed["engines"]] == [[1], [0]]
    assert all(e["model_id"] == "local:test-a" and e["state"] == "ready" for e in managed["engines"])
    assert managed["parallel_total"] == sum(e["parallel"] for e in managed["engines"])
    assert managed["loaded_models"] == ["local:test-a"] and managed["layout"]["mode"] == "single"
    assert managed["idle_stop_at"] is not None
    logs = client.get("/v1/engine/logs", headers=auth).get_json()
    assert logs["lines"] and logs["next"] == logs["lines"][-1]["seq"] + 1
    assert {"seq", "ts", "text"} == set(logs["lines"][0])
    more = client.get(f"/v1/engine/logs?after={logs['next']}&limit=5", headers=auth).get_json()
    assert more["lines"] == [] and more["next"] == logs["next"]
    page = client.get("/v1/engine/logs?after=1&limit=2", headers=auth).get_json()
    assert [line["seq"] for line in page["lines"]] == [1, 2] and page["next"] == 3
    assert client.get("/v1/engine/logs?after=x", headers=auth).status_code == 400
    # Uninstall is refused while running.
    assert client.post("/v1/engine/uninstall", headers=auth, json={}).status_code == 409
    # Deleting the loaded model is refused.
    response = client.delete("/v1/models/local/test-a", headers=auth)
    assert response.status_code == 409 and "loaded" in response.get_json()["error"]
    body = client.post("/v1/engine/stop", headers=auth).get_json()
    assert body["managed"]["state"] == "stopped" and body["managed"]["pid"] is None


def test_start_switches_models(local, auth):
    client = local["client"]
    client.post("/v1/engine/start", headers=auth, json={"model_id": "local:test-a"})
    first = wait_engine(client, auth, "ready")["managed"]["pid"]
    client.post("/v1/engine/start", headers=auth, json={"model_id": "local:test-b"})
    body = wait_until(lambda: (lambda b: b if b["managed"]["state"] == "ready"
                               and b["managed"]["loaded_model"] == "local:test-b" else None)(
        client.get("/v1/engine", headers=auth).get_json()))
    assert body["managed"]["pid"] != first


def test_start_errors(local, auth):
    client = local["client"]
    assert client.post("/v1/engine/start", headers=auth, json={"model_id": "local:nope"}).status_code == 400
    assert client.post("/v1/engine/start", headers=auth, data="x").status_code == 400
    local["rt"].models.delete("test-b")
    response = client.post("/v1/engine/start", headers=auth, json={"model_id": "local:test-b"})
    assert response.status_code == 409 and response.get_json()["error_code"] == "MODEL_NOT_AVAILABLE"


def test_engine_error_state_is_reported(local, auth, monkeypatch):
    monkeypatch.setenv("FAKE_LLAMA_MODE", "oom_on_start")
    client = local["client"]
    client.post("/v1/engine/start", headers=auth, json={"model_id": "local:test-a"})
    managed = wait_engine(client, auth, "error")["managed"]
    assert managed["last_error"]["code"] == "ENGINE_FAILED"
    assert "ran out of memory" in managed["last_error"]["message"]
    assert managed["running"] is False


# ------------------------------------------------------------- settings --

def test_engine_settings(local, auth):
    client = local["client"]
    response = client.put("/v1/settings", headers=auth, json={"engine": {
        "variant": "linux-x64-vulkan", "context_size": 16384, "parallel": 4, "gpu_devices": [0],
        "idle_stop_minutes": 0}})
    assert response.status_code == 200
    assert response.get_json()["engine"] == {"variant": "linux-x64-vulkan", "context_size": 16384, "parallel": 4,
                                             "gpu_devices": [0], "idle_stop_minutes": 0, "staged_batches": False}
    assert client.get("/v1/engine", headers=auth).get_json()["settings"]["parallel"] == 4
    # A preferred variant that is not installed falls back to an installed one.
    assert local["rt"].installer.active_install("linux-x64-vulkan")["variant"] == "linux-x64-cuda12"
    body = client.put("/v1/settings", headers=auth, json={"engine": {"context_size": "auto",
                                                                    "gpu_devices": "auto"}}).get_json()
    assert body["engine"]["context_size"] == "auto" and body["engine"]["gpu_devices"] == "auto"
    assert body["engine"]["parallel"] == 4   # partial update keeps the rest


@pytest.mark.parametrize("engine", [
    {"context_size": 100}, {"context_size": "big"}, {"parallel": 0}, {"parallel": 99}, {"parallel": True},
    {"gpu_devices": []}, {"gpu_devices": [0, 0]}, {"gpu_devices": [-1]}, {"gpu_devices": "all"},
    {"idle_stop_minutes": -1}, {"variant": "../x"}, {"variant": ""}, {"nope": 1},
])
def test_engine_settings_rejected(local, auth, engine):
    response = local["client"].put("/v1/settings", headers=auth, json={"engine": engine})
    assert response.status_code == 400


# -------------------------------------------------------------- catalog --

def test_catalog_runnable_and_recommended(local, auth):
    body = local["client"].get("/v1/models", headers=auth).get_json()
    builds = body["local"]["families"][0]["builds"]
    assert [b["install_state"] for b in builds] == ["installed", "installed"]
    assert all(b["runnable"] and b["unavailable_reason"] is None for b in builds)
    assert [b["recommended"] for b in builds] == [False, True]   # most capable that fits a GPU
    picks = [p for p in body["selectable"]["ocr"] if p["group"] == "Local"]
    assert all(p["runnable"] for p in picks)
    assert local["client"].put("/v1/selection", headers=auth,
                               json={"ocr": ["local:test-a"], "parse": "local:test-a"}).status_code == 200


def test_recommended_rules():
    from types import SimpleNamespace

    from vvl.catalog import recommended_builds

    family = SimpleNamespace(builds=[SimpleNamespace(id=i) for i in ("s", "m", "l")])
    v = lambda verdict, label="": {"verdict": verdict, "label": label}  # noqa: E731
    assert recommended_builds(family, {"s": v("fits_gpu"), "m": v("fits_gpu"), "l": v("too_large")}) == "m"
    assert recommended_builds(family, {"s": v("fits_split", "Partly on GPU"), "m": v("fits_split", "Partly on GPU"),
                                       "l": v("too_large")}) == "s"
    assert recommended_builds(family, {"s": v("fits_split", "Split across GPUs"),
                                       "m": v("fits_split", "Split across GPUs"), "l": v("too_large")}) == "m"
    assert recommended_builds(family, {"s": v("cpu_only"), "m": v("cpu_only"), "l": v("too_large")}) == "s"
    assert recommended_builds(family, {"s": v("unknown"), "m": v("unknown"), "l": v("unknown")}) is None


def test_vulkan_index_mapping():
    from vvl.llama.supervisor import vulkan_indexes

    report = load_report()
    same = [{"name": "Vulkan0", "description": "NVIDIA RTX 6000 Ada Generation"},
            {"name": "Vulkan1", "description": "NVIDIA RTX 6000 Ada Generation"}]
    assert vulkan_indexes([1], report, same) == [1]
    assert vulkan_indexes([0], report, same) == [0]
    assert vulkan_indexes([0, 1], report, same) == [0, 1]
    # An iGPU enumerated first by Vulkan: names decide.
    mixed = [{"name": "Vulkan0", "description": "AMD Radeon Graphics (RADV RAPHAEL_MENDOCINO)"},
             {"name": "Vulkan1", "description": "NVIDIA RTX 6000 Ada Generation"},
             {"name": "Vulkan2", "description": "NVIDIA RTX 6000 Ada Generation"}]
    assert vulkan_indexes([1], report, mixed) == [2]
    assert vulkan_indexes([0], report, None) == [0]


# ------------------------------------------------------- /process end to end --

def _post(client, auth, data):
    with open(FIXTURE_IMAGE, "rb") as handle:
        return client.post("/process", headers=auth, content_type="multipart/form-data",
                           data={"file": (handle, "MICH_16205594.jpg"), "include_cop90": "false", **data})


def test_process_with_local_model(local, auth):
    client, state = local["client"], local["state"]
    response = _post(client, auth, {"engines": "local:test-a", "llm_model": "local:test-a",
                                    "prompt": "SLTPvM_default.yaml"})
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["engine_notice"] is None
    assert body["ocr"].startswith("FAKE OCR TEXT")
    info = body["ocr_info"]["local:test-a"]
    assert info["tokens_in"] == 1234 and info["tokens_out"] == 56 and info["total_cost"] == 0
    record = body["formatted_json"]
    import yaml

    from vvl import paths

    rules = yaml.safe_load((paths.prompts_builtin_dir() / "SLTPvM_default.yaml").read_text(encoding="utf-8"))["rules"]
    assert list(record) == list(rules)
    assert record["catalogNumber"] == "fake catalogNumber"
    assert body["parsing_info"]["model"] == "local:test-a"
    assert body["parsing_info"]["input"] == 1234 and body["parsing_info"]["total_cost"] == 0
    assert body["success"]["llm"] == "True" and body["total_request_cost_usd"] == 0
    ocr_request, parse_request = _requests(state)
    content = ocr_request["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1]["text"].startswith("Please perform OCR on this scientific image")
    # Output length "standard" with reasoning off; a model without an entry in
    # model_settings.json gets no sampling fields and no reasoning switch.
    assert ocr_request["max_tokens"] == 3072 and "temperature" not in ocr_request
    assert "chat_template_kwargs" not in ocr_request
    schema = parse_request["response_format"]["json_schema"]["schema"]
    assert schema["required"] == list(rules) and schema["additionalProperties"] is False
    assert parse_request["messages"][0]["content"].startswith("Answer the user query.")
    # The engine was started for this request (auto-start).
    managed = client.get("/v1/engine", headers=auth).get_json()["managed"]
    assert managed["state"] == "ready" and managed["loaded_model"] == "local:test-a"


def test_process_ocr_only_and_model_switch(local, auth):
    client = local["client"]
    response = _post(client, auth, {"engines": "local:test-a", "ocr_only": "true"})
    assert response.status_code == 200
    response = _post(client, auth, {"engines": "local:test-a", "llm_model": "local:test-b",
                                    "prompt": "SLTPvM_default.yaml"})
    assert response.status_code == 200
    assert response.get_json()["parsing_info"]["model"] == "local:test-b"
    assert client.get("/v1/engine", headers=auth).get_json()["managed"]["loaded_model"] == "local:test-b"


def test_process_engine_failure_is_503(local, auth, monkeypatch):
    monkeypatch.setenv("FAKE_LLAMA_MODE", "crash_on_start")
    response = _post(local["client"], auth, {"engines": "local:test-a", "ocr_only": "true"})
    assert response.status_code == 503
    body = response.get_json()
    assert body["error_code"] == "ENGINE_NOT_READY"
    assert "exit code 3" in body["error"]


def test_process_garbage_parse_reply(local, auth, monkeypatch):
    monkeypatch.setenv("FAKE_LLAMA_MODE", "chat_garbage")
    response = _post(local["client"], auth, {"engines": "local:test-a", "llm_model": "local:test-a",
                                             "prompt": "SLTPvM_default.yaml"})
    assert response.status_code == 200
    body = response.get_json()
    assert body["success"]["llm"] == "False"


def test_process_retries_transient_engine_error(local, auth, monkeypatch):
    monkeypatch.setenv("FAKE_LLAMA_MODE", "chat_500_once")
    response = _post(local["client"], auth, {"engines": "local:test-a", "ocr_only": "true"})
    assert response.status_code == 200
    assert len(_requests(local["state"])) == 2


def test_process_model_not_downloaded_is_409(local, auth):
    local["rt"].models.delete("test-b")
    response = _post(local["client"], auth, {"engines": "local:test-b", "ocr_only": "true"})
    assert response.status_code == 409
    body = response.get_json()
    assert body["error_code"] == "MODEL_NOT_AVAILABLE"
    assert body["details"]["unavailable_reason"] == "Download this model first (Models tab)."


def test_shutdown_route_stops_engine(local, auth):
    import threading

    client, rt = local["client"], local["rt"]
    exited = threading.Event()
    local["app"].config["VVL_EXIT_FUNC"] = lambda code: exited.set()
    client.post("/v1/engine/start", headers=auth, json={"model_id": "local:test-a"})
    pid = wait_engine(client, auth, "ready")["managed"]["pid"]
    assert client.post("/v1/shutdown", headers=auth).get_json() == {"ok": True}
    assert exited.wait(20)
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_doctor_reports_engine_and_models(local, vvl_env):
    from vvl.doctor import run_checks

    checks = {c["name"]: c for c in run_checks()}
    assert checks["engine"]["status"] == "ok"
    assert "linux-x64-cuda12" in checks["engine"]["detail"]
    # The doctor reads the real registry; the fake builds are not in it.
    assert checks["local models"]["status"] in ("ok", "warn")
