"""GET /v1/models shape, the system-checker fallback, and the other /v1 info routes."""

from __future__ import annotations

import platform
import sys
import types

import pytest

pytestmark = pytest.mark.usefixtures("no_network")


@pytest.fixture
def no_system(monkeypatch):
    """Make `vvl.system` look absent, whatever its state in the working tree."""
    fake = types.ModuleType("vvl.system")
    monkeypatch.setitem(sys.modules, "vvl.system", fake)
    import vvl

    monkeypatch.setattr(vvl, "system", fake, raising=False)
    return fake


@pytest.fixture
def fake_system(monkeypatch):
    calls = {"report": 0, "assess": []}
    pkg = types.ModuleType("vvl.system")
    fit = types.ModuleType("vvl.system.fit")

    def get_report(refresh=False):
        calls["report"] += 1
        return {"schema": 1, "capability": {"summary": "fake box"}, "warnings": [], "probes": []}

    def assess(report, build):
        calls["assess"].append(build["id"])
        return {"verdict": "fits_gpu", "label": "Fits on GPU", "required_bytes": build["total_size_bytes"],
                "detail": "fake"}

    pkg.get_report = get_report
    pkg.fit = fit
    fit.assess = assess
    monkeypatch.setitem(sys.modules, "vvl.system", pkg)
    monkeypatch.setitem(sys.modules, "vvl.system.fit", fit)
    import vvl

    monkeypatch.setattr(vvl, "system", pkg, raising=False)
    return calls


def test_models_shape(client, auth, no_system):
    response = client.get("/v1/models", headers=auth)
    assert response.status_code == 200
    body = response.get_json()
    assert list(body) == ["local", "api", "demo", "selection", "selectable"]

    families = body["local"]["families"]
    qwen = families[0]
    assert qwen["id"] == "qwen3-vl" and qwen["status"] == "verified"
    build = next(b for b in qwen["builds"] if b["id"] == "qwen3-vl-8b-q8")
    assert set(build) == {"id", "model_id", "label", "params", "quant", "tier", "hf_repo", "total_size_bytes",
                          "files", "install_state", "installed_bytes", "download_job_id", "recommended", "fit",
                          "runnable", "unavailable_reason"}
    assert build["model_id"] == "local:qwen3-vl-8b-q8"
    assert build["total_size_bytes"] == 9461809184
    assert build["files"][0] == {"role": "model", "path": "Qwen3VL-8B-Instruct-Q8_0.gguf", "size_bytes": 8709519456}
    assert build["install_state"] == "not_installed"
    assert build["installed_bytes"] == 0
    assert build["download_job_id"] is None
    assert build["recommended"] is False  # fit is unknown without a system report
    assert build["runnable"] is False
    assert build["unavailable_reason"] == (
        "Download this model and install the llama.cpp engine first (Models and Engine tabs).")
    assert build["fit"]["verdict"] == "unknown"
    assert build["fit"]["label"] == "Not checked"
    assert build["fit"]["required_bytes"] is None
    assert any(f["status"] == "planned" and f["builds"] == [] for f in families)

    api_ids = [p["id"] for p in body["api"]["providers"]]
    assert api_ids == ["gemini", "anthropic", "openai", "openrouter"]  # model providers only
    gemini = body["api"]["providers"][0]
    assert gemini["configured"] is False
    assert gemini["models_state"] == "no_key"
    assert gemini["models"] == []
    assert gemini["unavailable_reason"] == "This provider is not available in this version yet. Use OpenRouter for hosted models."

    assert body["demo"] == [{"model_id": "demo:mock", "label": "Demo engine (sample output, no model)",
                             "roles": ["ocr", "parse"], "runnable": True}]
    assert body["selection"] == {"ocr": ["demo:mock"], "parse": "demo:mock"}

    for role in ("ocr", "parse"):
        picks = body["selectable"][role]
        assert picks[0]["model_id"] == "demo:mock"
        assert picks[0]["group"] == "Demo"
        assert picks[0]["runnable"] is True
        assert picks[0]["capabilities"] == {"vision": True, "thinking_levels": [], "structured_output": True}
        local = [p for p in picks if p["group"] == "Local"]
        verified = [b for f in body["local"]["families"] if f["status"] == "verified" for b in f["builds"]]
        assert len(local) == len(verified) and len(local) >= 7
        assert all(p["runnable"] is False and p["unavailable_reason"] for p in local)


def test_models_uses_system_fit(client, auth, fake_system):
    body = client.get("/v1/models", headers=auth).get_json()
    build = body["local"]["families"][0]["builds"][0]
    assert build["fit"] == {"verdict": "fits_gpu", "label": "Fits on GPU",
                            "required_bytes": build["total_size_bytes"], "detail": "fake"}
    assert fake_system["report"] == 1
    builds = [b for f in body["local"]["families"] for b in f["builds"]]
    assert len(fake_system["assess"]) == len(builds), "one fit check per build"


def test_models_survives_system_errors(client, auth, monkeypatch):
    pkg = types.ModuleType("vvl.system")

    def get_report(refresh=False):
        raise RuntimeError("probe exploded")

    pkg.get_report = get_report
    monkeypatch.setitem(sys.modules, "vvl.system", pkg)
    import vvl

    monkeypatch.setattr(vvl, "system", pkg, raising=False)
    body = client.get("/v1/models", headers=auth).get_json()
    assert body["local"]["families"][0]["builds"][0]["fit"]["verdict"] == "unknown"
    report = client.get("/v1/system", headers=auth).get_json()
    assert report["schema"] == 1
    assert report["warnings"][0]["code"] == "SYSTEM_CHECK_UNAVAILABLE"


def test_system_route(client, auth, fake_system):
    assert client.get("/v1/system", headers=auth).get_json()["capability"]["summary"] == "fake box"
    assert client.get("/v1/system?refresh=1", headers=auth).status_code == 200


def test_system_fallback_shape(client, auth, no_system):
    report = client.get("/v1/system", headers=auth).get_json()
    for key in ("schema", "generated_at", "duration_ms", "os", "cpu", "memory", "gpus", "accelerators",
                "disk", "capability", "warnings", "probes"):
        assert key in report
    assert report["gpus"] == []
    assert report["probes"][0]["ok"] is False


def test_app_route(client, auth, vvl_env):
    body = client.get("/v1/app", headers=auth).get_json()
    assert body["name"] == "VoucherVisionLocal"
    assert body["phase"] == 2
    assert body["from_source"] is True
    assert body["python"] == platform.python_version()
    assert body["packaged"] is False
    assert body["paths"]["private_file"] == str(vvl_env["VVL_PRIVATE_PATH"])
    assert body["paths"]["settings_file"] == str(vvl_env["VVL_SETTINGS_PATH"])
    assert body["paths"]["prompts_user"] == str(vvl_env["VVL_PROMPTS_USER_DIR"])
    assert body["features"] == {
        "demo_engine": True, "local_inference": True, "api_inference": False, "engine_install": True,
        "model_download": True, "enrichment_elevation": True, "enrichment_wfo": False,
    }
    assert set(body["feature_notes"]) == {"api_inference", "enrichment_wfo"}
    assert not vvl_env["VVL_PRIVATE_PATH"].exists()


def test_app_route_packaged(client, auth, monkeypatch):
    monkeypatch.setenv("VVL_PACKAGED", "1")
    assert client.get("/v1/app", headers=auth).get_json()["packaged"] is True
    monkeypatch.setenv("VVL_PACKAGED", "yes")
    assert client.get("/v1/app", headers=auth).get_json()["packaged"] is False


def test_engine_routes(client, auth, fake_system):
    """Phase 2 shape of GET /v1/engine (the full engine tests are in test_engine_routes.py)."""
    body = client.get("/v1/engine", headers=auth).get_json()
    assert set(body) == {"available", "unavailable_reason", "models_dir", "managed", "recommended_variant",
                         "variants", "install_job_id", "settings", "plan"}
    assert body["plan"] == {"mode": "none", "staged": False, "summary": "", "concurrency": None, "fits": True}
    assert body["managed"]["installed"] is False
    assert body["managed"]["state"] == "not_installed"
    assert {"pinned_version", "installed", "version", "variant", "state", "running", "pid", "port", "backend",
            "loaded_model", "gpu_devices", "context_size", "parallel", "started_at", "last_error",
            "idle_stop_at"} <= set(body["managed"])
    assert body["managed"]["pinned_version"] == "b12359"
    assert body["settings"] == {"variant": "auto", "context_size": "auto", "parallel": "auto", "gpu_devices": "auto",
                                "idle_stop_minutes": 15, "staged_batches": False}
    response = client.post("/v1/engine/start", headers=auth, json={})
    assert response.status_code == 400
    response = client.post("/v1/engine/start", headers=auth, json={"model_id": "local:qwen3-vl-2b-q8"})
    assert response.status_code == 409
    assert response.get_json()["error_code"] == "ENGINE_NOT_INSTALLED"
    assert client.post("/v1/engine/stop", headers=auth).get_json()["managed"]["state"] == "not_installed"


def test_shutdown_route(make_app, auth):
    import threading

    exited = threading.Event()
    codes = []

    def fake_exit(code):
        codes.append(code)
        exited.set()

    client = make_app(VVL_EXIT_FUNC=fake_exit).test_client()
    response = client.post("/v1/shutdown", headers=auth)
    assert response.get_json() == {"ok": True}
    assert exited.wait(5)
    assert codes == [0]


def test_api_costs(client, auth):
    body = client.get("/api-costs", headers=auth).get_json()
    assert isinstance(body, dict)
    assert "GEMINI_2_5_FLASH" in body
