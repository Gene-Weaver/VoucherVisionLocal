"""The engine pool (vvl/llama/pool.py) against the fake llama-server: one model on
every card, two models side by side, taking turns when they do not fit, and the
two-pass /process route that goes with it."""

from __future__ import annotations

import copy
import io
import os
import signal
import threading
import time

import pytest
import yaml

from conftest import free_port
from conftest import FIXTURE_IMAGE
from p2util import FakeModels, fake_engine_install, load_report, make_runtime, wait_until

pytestmark = [pytest.mark.usefixtures("no_network"),
              pytest.mark.skipif(os.name == "nt", reason="the fake engine wrapper is a POSIX shell script")]

GiB = 1024 ** 3


@pytest.fixture
def fixture_image():
    return FIXTURE_IMAGE.read_bytes()


class PoolEnv:
    def __init__(self, tmp_path, vvl_env):
        self.models_dir = tmp_path / "models_dir"
        self.state = tmp_path / "fake_state"
        self.state.mkdir()
        self.fake = FakeModels(tmp_path)
        self.fake.install_on_disk(self.models_dir, "test-a")
        self.fake.install_on_disk(self.models_dir, "test-b")
        fake_engine_install(self.models_dir, state_dir=self.state)
        self.settings_path = vvl_env["VVL_SETTINGS_PATH"]
        self.report = load_report()
        self.runtimes = []
        self.write()

    def write(self, selection=None, **engine):
        data = {"models_dir": str(self.models_dir)}
        if selection:
            data["selection"] = selection
        if engine:
            data["engine"] = engine
        self.settings_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    def free(self, *gib):
        """Set the free graphics memory the pool sees, per card."""
        for gpu, value in zip(self.report["gpus"], gib):
            gpu["vram_free_bytes"] = int(value * GiB)

    def runtime(self):
        base = free_port()
        rt = make_runtime(self.settings_path, self.fake.registry_path,
                          VVL_SYSTEM_REPORT=lambda refresh=False: copy.deepcopy(self.report),
                          VVL_ENGINE_LOAD_TIMEOUT=15, VVL_ENGINE_PORT_RANGE=(base, base + 40),
                          VVL_ENGINE_STOP_TIMEOUT=3)
        self.runtimes.append(rt)
        return rt

    def starts(self):
        import json

        path = self.state / "starts.txt"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture
def env(tmp_path, vvl_env, monkeypatch):
    monkeypatch.delenv("FAKE_LLAMA_MODE", raising=False)
    e = PoolEnv(tmp_path, vvl_env)
    yield e
    for rt in e.runtimes:
        rt.shutdown(timeout=5)
    for start in e.starts():
        try:
            os.kill(start["pid"], 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            pass
        try:
            import psutil

            if psutil.Process(start["pid"]).status() == psutil.STATUS_ZOMBIE:
                continue
        except Exception:  # noqa: BLE001
            continue
        os.kill(start["pid"], signal.SIGKILL)
        pytest.fail(f"fake engine {start['pid']} was left running")


def ready(rt, count):
    def check():
        status = rt.supervisor.status()
        engines = status["engines"]
        return len(engines) == count and all(e["state"] == "ready" for e in engines) and status
    return wait_until(check, 20)


def test_one_model_runs_on_every_card(env):
    env.write({"ocr": ["local:test-a"], "parse": "local:test-a"})
    rt = env.runtime()
    rt.supervisor.start("local:test-a")
    status = ready(rt, 2)
    assert sorted(e["gpu_devices"] for e in status["engines"]) == [[0], [1]]
    assert status["state"] == "ready" and status["loaded_models"] == ["local:test-a"]
    assert status["layout"] == {"mode": "single", "waiting": []}
    assert status["gpu_devices"] == [0, 1] and status["parallel_total"] == sum(e["parallel"] for e in status["engines"])
    assert sorted(s["cuda_visible"] for s in env.starts()) == ["0", "1"]
    assert rt.supervisor.loaded_model() == "local:test-a" and rt.supervisor.is_running()
    assert len(rt.supervisor.engine_pids()) == 2
    # Log lines say which engine they came from once there are several.
    text = "\n".join(line["text"] for line in rt.supervisor.logs.read(None, 500)["lines"])
    assert "[engine 1] " in text and "[engine 2] " in text
    stopped = rt.supervisor.stop()
    assert stopped["engines"] == [] and stopped["state"] == "stopped" and not rt.supervisor.is_running()


def test_requests_are_shared_between_engines(env):
    env.write({"ocr": ["local:test-a"], "parse": "local:test-a"}, parallel=1)
    rt = env.runtime()
    held = threading.Event()
    release = threading.Event()
    used = []

    def request():
        with rt.supervisor.lease("local:test-a") as handle:
            used.append(handle.base_url)
            if len(used) == 2:
                held.set()
            release.wait(10)

    threads = [threading.Thread(target=request) for _ in range(2)]
    for thread in threads:
        thread.start()
    assert held.wait(20), "both requests should be running at once"
    assert len(set(used)) == 2, "one request per engine, not two on the same engine"
    assert rt.supervisor.status()["active_requests"] == 2
    release.set()
    for thread in threads:
        thread.join(10)
    assert rt.supervisor.status()["active_requests"] == 0
    assert len(env.starts()) == 2


def test_two_models_that_fit_stay_loaded_side_by_side(env):
    env.write({"ocr": ["local:test-a"], "parse": "local:test-b"})
    rt = env.runtime()
    with rt.supervisor.lease("local:test-a"):
        pass
    with rt.supervisor.lease("local:test-b"):
        pass
    status = wait_until(lambda: (lambda s: all(e["state"] == "ready" for e in s["engines"]) and s)(rt.supervisor.status()))
    assert status["layout"]["mode"] == "shared" and sorted(status["loaded_models"]) == ["local:test-a", "local:test-b"]
    starts_before = len(env.starts())
    # Going back and forth needs no restart: both models are there.
    for model in ("local:test-a", "local:test-b", "local:test-a"):
        with rt.supervisor.lease(model):
            pass
    assert len(env.starts()) == starts_before
    text = "\n".join(line["text"] for line in rt.supervisor.logs.read(None, 1000)["lines"])
    assert "stopping llama-server" not in text


def test_two_models_that_do_not_fit_take_turns(env):
    env.write({"ocr": ["local:test-a"], "parse": "local:test-b"}, gpu_devices=[0], parallel=1)
    rt = env.runtime()
    need_a = rt.supervisor._need("local:test-a", ("ocr",), rt.engine_settings()).unit_bytes(1)
    need_b = rt.supervisor._need("local:test-b", ("parse",), rt.engine_settings()).unit_bytes(1)
    from vvl.llama.pool import CARD_RESERVE_BYTES

    room = (max(need_a, need_b) + CARD_RESERVE_BYTES) / GiB + 0.01
    env.free(room, 0.1)
    order = []
    in_a = threading.Event()
    finish_a = threading.Event()

    def use_a():
        with rt.supervisor.lease("local:test-a"):
            order.append("a-start")
            in_a.set()
            finish_a.wait(10)
            order.append("a-end")

    def use_b():
        with rt.supervisor.lease("local:test-b"):
            order.append("b-start")

    thread_a = threading.Thread(target=use_a)
    thread_a.start()
    assert in_a.wait(20)
    env.free(room - need_a / GiB, 0.1)      # what a measurement shows once model A is loaded
    status = rt.supervisor.status()
    assert status["layout"] == {"mode": "exclusive", "waiting": ["local:test-b"]}
    thread_b = threading.Thread(target=use_b)
    thread_b.start()
    time.sleep(0.6)
    assert order == ["a-start"], "the other model waits for the running request"
    finish_a.set()
    thread_a.join(10)
    thread_b.join(20)
    env.free(room - need_b / GiB, 0.1)
    assert order == ["a-start", "a-end", "b-start"]
    status = rt.supervisor.status()
    assert status["loaded_models"] == ["local:test-b"] and status["layout"]["waiting"] == ["local:test-a"]
    aliases = [s["argv"][s["argv"].index("--alias") + 1] for s in env.starts()]
    assert aliases == ["test-a", "test-b"], "one swap, not one per request"


def test_engine_that_is_already_right_is_kept_when_memory_shifts(env):
    env.write({"ocr": ["local:test-a"], "parse": "local:test-a"})
    rt = env.runtime()
    rt.supervisor.start("local:test-a")
    ready(rt, 2)
    env.free(30, 31)          # other programs took memory; the plan would now come out differently
    with rt.supervisor.lease("local:test-a"):
        pass
    rt.supervisor.start("local:test-a")
    assert len(env.starts()) == 2, "working engines are not restarted"


def test_unknown_or_missing_model_is_refused(env):
    from vvl.llama.supervisor import EngineNotReady, SupervisorError

    rt = env.runtime()
    with pytest.raises(SupervisorError):
        rt.supervisor.start("local:nope")
    with pytest.raises(EngineNotReady):
        with rt.supervisor.lease("local:nope", timeout=3):
            pass
    assert env.starts() == []


# ------------------------------------------------------- two-pass route --

def test_two_pass_request_matches_one_pass(client, auth, fixture_image):
    """`stage=ocr` then /v1/process/finish gives what a single /process gives."""
    def upload(**fields):
        data = {"file": (io.BytesIO(fixture_image), "sheet.jpg"), "engines": "demo:mock", "llm_model": "demo:mock",
                "prompt": "SLTPvM_default.yaml", "include_cop90": "false"}
        data.update(fields)
        return client.post("/process", headers=auth, data=data, content_type="multipart/form-data")

    whole = upload().get_json()
    first = upload(stage="ocr")
    assert first.status_code == 200
    staged = first.get_json()
    assert staged["staged"] is True and staged["stage"] == "ocr" and staged["filename"] == "sheet.jpg"
    assert staged["ocr"] and "formatted_json" not in staged
    token = staged["stage_token"]

    second = client.post("/v1/process/finish", headers=auth,
                         json={"stage_token": token, "llm_model": "demo:mock", "prompt": "SLTPvM_default.yaml",
                               "include_cop90": False})
    assert second.status_code == 200
    done = second.get_json()
    assert list(done) == list(whole), "same keys in the same order as a one-pass result"
    for key in ("filename", "prompt", "ocr", "formatted_json", "parsing_info", "success", "engine_notice"):
        assert done[key] == whole[key], key
    assert done["collage_info"]["base64image_text_collage"] == whole["collage_info"]["base64image_text_collage"]

    # A finished image's saved result is gone; so is anything unknown or malformed.
    again = client.post("/v1/process/finish", headers=auth, json={"stage_token": token})
    assert again.status_code == 404 and again.get_json()["error_code"] == "STAGE_NOT_FOUND"
    for bad in ("../../etc/passwd", "x", ""):
        response = client.post("/v1/process/finish", headers=auth, json={"stage_token": bad})
        assert response.status_code in (400, 404)
    assert client.post("/v1/process/finish", json={"stage_token": token}).status_code == 401
    assert upload(stage="parse").status_code == 400


def test_two_pass_ocr_only_request_is_finished_in_one_pass(client, auth, fixture_image):
    response = client.post("/process", headers=auth, content_type="multipart/form-data", data={
        "file": (io.BytesIO(fixture_image), "sheet.jpg"), "engines": "demo:mock", "ocr_only": "true", "stage": "ocr"})
    body = response.get_json()
    assert response.status_code == 200 and "staged" not in body and body["success"]["llm"] == "False"


# ------------------------------------------------------------ plan summary --

def test_plan_summary_tells_batches_what_to_do(env):
    from vvl.api.engine import plan_summary
    from vvl.llama.pool import CARD_RESERVE_BYTES

    env.write({"ocr": ["demo:mock"], "parse": "demo:mock"})
    rt = env.runtime()
    assert plan_summary(rt) == {"mode": "none", "staged": False, "summary": "", "concurrency": None, "fits": True}

    env.write({"ocr": ["local:test-a"], "parse": "local:test-a"})
    one = plan_summary(rt)
    assert one["mode"] == "single" and one["staged"] is False
    assert one["concurrency"] == 8 and "2 copies, 8 images at a time" in one["summary"]

    # The user's setting turns two passes on even when everything fits.
    env.write({"ocr": ["local:test-a"], "parse": "local:test-b"}, staged_batches=True)
    both = plan_summary(rt)
    assert both["mode"] == "shared" and both["staged"] is True

    # Models that cannot be loaded together always use two passes.
    env.write({"ocr": ["local:test-a"], "parse": "local:test-b"}, gpu_devices=[0], parallel=1)
    need = max(rt.supervisor._need(m, ("ocr",), rt.engine_settings()).unit_bytes(1)
               for m in ("local:test-a", "local:test-b"))
    env.free((need + CARD_RESERVE_BYTES) / GiB + 0.01, 40)
    turns = plan_summary(rt)
    assert turns["mode"] == "exclusive" and turns["staged"] is True and turns["concurrency"] == 1
    assert "one at a time" in turns["summary"]
