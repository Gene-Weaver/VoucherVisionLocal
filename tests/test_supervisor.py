"""The engine supervisor against tests/fakes/fake_llama_server.py: readiness, slow
load, crashes, port retry, stop escalation, stale-engine reaper, idle stop,
concurrency, model switching, and dying with the parent."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from conftest import free_port
from p2util import FAKE_LLAMA, REPO, FakeModels, fake_engine_install, make_runtime, wait_until, write_settings

pytestmark = [pytest.mark.usefixtures("no_network"),
              pytest.mark.skipif(os.name == "nt", reason="the fake engine wrapper is a POSIX shell script")]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        import psutil

        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:  # noqa: BLE001
        return False


class Env:
    def __init__(self, tmp_path, vvl_env, monkeypatch):
        self.tmp = tmp_path
        self.models_dir = tmp_path / "models_dir"
        self.state = tmp_path / "fake_state"
        self.state.mkdir()
        self.fake = FakeModels(tmp_path)
        self.fake.install_on_disk(self.models_dir, "test-a")
        self.fake.install_on_disk(self.models_dir, "test-b")
        self.server = fake_engine_install(self.models_dir, state_dir=self.state)
        self.settings_path = vvl_env["VVL_SETTINGS_PATH"]
        self.monkeypatch = monkeypatch
        self.engine_settings = {}
        self.runtimes = []
        write_settings(self.settings_path, self.models_dir)
        base = free_port()
        self.port_range = (base, base + 30)

    def mode(self, mode: str, **env):
        self.monkeypatch.setenv("FAKE_LLAMA_MODE", mode)
        for key, value in env.items():
            self.monkeypatch.setenv(key, str(value))

    def settings(self, **engine):
        self.engine_settings.update(engine)
        write_settings(self.settings_path, self.models_dir, **self.engine_settings)

    def runtime(self, **extra):
        config = {"VVL_ENGINE_LOAD_TIMEOUT": 15, "VVL_ENGINE_PORT_RANGE": self.port_range,
                  "VVL_ENGINE_STOP_TIMEOUT": 3}
        config.update(extra)
        rt = make_runtime(self.settings_path, self.fake.registry_path, **config)
        # These tests exercise ONE engine: the supervisor unit the engine pool is
        # built from (the pool itself is covered in test_pool.py).
        rt.pool = rt.supervisor
        rt.supervisor = rt.pool._unit(0)
        self.runtimes.append(rt)
        return rt

    def starts(self) -> list[dict]:
        path = self.state / "starts.txt"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def requests(self) -> list[dict]:
        path = self.state / "requests.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture
def env(tmp_path, vvl_env, monkeypatch):
    monkeypatch.delenv("FAKE_LLAMA_MODE", raising=False)
    e = Env(tmp_path, vvl_env, monkeypatch)
    yield e
    for rt in e.runtimes:
        rt.shutdown(timeout=5)
        rt.pool.shutdown(timeout=5)
    for start in e.starts():
        pid = start["pid"]
        if _alive(pid):
            os.kill(pid, signal.SIGKILL)
            pytest.fail(f"fake engine {pid} was left running")


def wait_state(rt, *states, timeout=15):
    return wait_until(lambda: rt.supervisor.status()["state"] in states and rt.supervisor.status(), timeout)


# ------------------------------------------------------------- lifecycle --

def test_start_ready_and_stop(env, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_HOST", "0.0.0.0")      # must never reach the engine
    monkeypatch.setenv("LLAMA_API_KEY", "inherited-key")  # replaced by VVL's own key
    rt = env.runtime()
    status = rt.supervisor.start("local:test-a")
    assert status["state"] in ("starting", "loading_model", "ready")
    status = wait_state(rt, "ready")
    assert status["running"] is True
    assert status["loaded_model"] == "local:test-a"
    assert env.port_range[0] <= status["port"] < env.port_range[1]
    assert status["gpu_devices"] == [1]          # GPU 1 has the most free memory in the fixture report
    assert status["context_size"] == 8192        # the smallest context VVL uses
    assert status["parallel"] == 2
    assert status["variant"]["id"] == "linux-x64-cuda12"
    assert status["backend"] == "cuda"
    assert status["started_at"] and status["last_error"] is None
    start = env.starts()[0]
    argv = start["argv"]
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--ctx-size") + 1] == "16384"     # 8192 per slot x 2 slots
    assert argv[argv.index("--parallel") + 1] == "2"
    assert "--no-kv-unified" in argv and "--no-webui" in argv
    assert argv[argv.index("--image-max-tokens") + 1] == "1024"
    assert argv[argv.index("--mmproj") + 1].endswith("mmproj-shared.gguf")
    assert start["api_key_set"] is True
    assert start["cuda_visible"] == "1"
    assert start["llama_env"] == ["LLAMA_API_KEY"]
    assert "inherited-key" not in " ".join(argv)
    pidfile = json.loads((Path(os.environ["VVL_HOME"]) / "run" / "engine.pid.json").read_text())
    assert pidfile["pid"] == status["pid"] and pidfile["create_time"] and pidfile["cmdline"]
    logs = rt.supervisor.logs.read(None, 500)
    text = "\n".join(line["text"] for line in logs["lines"])
    assert "[vvl] starting local:test-a" in text and "server is listening" in text
    pid = status["pid"]
    stopped = rt.supervisor.stop()
    assert stopped["state"] == "stopped" and stopped["running"] is False and stopped["pid"] is None
    assert not _alive(pid)
    assert not (Path(os.environ["VVL_HOME"]) / "run" / "engine.pid.json").exists()


def test_api_key_never_logged_or_reported(env):
    rt = env.runtime()
    rt.supervisor.start("local:test-a")
    wait_state(rt, "ready")
    proc = rt.supervisor._proc
    key = proc.api_key
    assert key and key not in json.dumps(rt.supervisor.status())
    assert key not in json.dumps(rt.supervisor.logs.read(None, 1000))
    assert key not in (Path(os.environ["VVL_HOME"]) / "run" / "engine.pid.json").read_text()
    rt.supervisor.logs.append(f"pretend the engine printed {key}")
    assert key not in json.dumps(rt.supervisor.logs.read(None, 1000))


def test_slow_load_goes_through_loading_model(env):
    env.mode("ready", FAKE_LLAMA_LOAD_SECONDS=1.5)
    rt = env.runtime()
    rt.supervisor.start("local:test-a")
    seen = set()
    end = time.monotonic() + 15
    while time.monotonic() < end:
        state = rt.supervisor.status()["state"]
        seen.add(state)
        if state == "ready":
            break
        time.sleep(0.05)
    assert "loading_model" in seen and "ready" in seen


def test_crash_on_start(env):
    env.mode("crash_on_start")
    rt = env.runtime()
    rt.supervisor.start("local:test-a")
    status = wait_state(rt, "error")
    err = status["last_error"]
    assert err["code"] == "ENGINE_FAILED"
    assert "exit code 3" in err["message"]
    assert any("GGML_ASSERT" in line for line in err["log_tail"])
    assert status["running"] is False and status["loaded_model"] is None


def test_out_of_memory_is_explained(env):
    env.mode("oom_on_start")
    rt = env.runtime()
    rt.supervisor.start("local:test-a")
    err = wait_state(rt, "error")["last_error"]
    assert "ran out of memory" in err["message"]
    assert any("cudaMalloc failed" in line for line in err["log_tail"])


def test_crash_later(env):
    env.mode("crash_later", FAKE_LLAMA_CRASH_AFTER=0.5)
    rt = env.runtime()
    rt.supervisor.start("local:test-a")
    wait_state(rt, "ready")
    status = wait_state(rt, "error")
    assert "stopped unexpectedly (exit code 7)" in status["last_error"]["message"]
    assert any("illegal memory access" in line for line in status["last_error"]["log_tail"])
    assert not (Path(os.environ["VVL_HOME"]) / "run" / "engine.pid.json").exists()


def test_port_in_use_retries_another_port(env):
    env.mode("bind_fail_once")
    rt = env.runtime()
    rt.supervisor.start("local:test-a")
    status = wait_state(rt, "ready", "error")
    assert status["state"] == "ready", status["last_error"]
    starts = env.starts()
    assert len(starts) == 2
    ports = [int(s["argv"][s["argv"].index("--port") + 1]) for s in starts]
    assert ports[0] != ports[1]


def test_busy_ports_are_skipped(env):
    held = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    held.bind(("127.0.0.1", env.port_range[0]))
    held.listen(1)
    try:
        rt = env.runtime()
        rt.supervisor.start("local:test-a")
        status = wait_state(rt, "ready")
        assert status["port"] != env.port_range[0]
    finally:
        held.close()


def test_stop_escalates_to_sigkill(env):
    env.mode("ignore_sigterm")
    rt = env.runtime(VVL_ENGINE_STOP_TIMEOUT=0.5)
    rt.supervisor.start("local:test-a")
    pid = wait_state(rt, "ready")["pid"]
    t0 = time.monotonic()
    rt.supervisor.stop()
    assert time.monotonic() - t0 < 8
    assert not _alive(pid)
    text = "\n".join(line["text"] for line in rt.supervisor.logs.read(None, 500)["lines"])
    assert "ignoring SIGTERM" in text and "llama-server killed" in text


def test_load_timeout(env):
    env.mode("ready", FAKE_LLAMA_LOAD_SECONDS=30)
    rt = env.runtime(VVL_ENGINE_LOAD_TIMEOUT=1)
    rt.supervisor.start("local:test-a")
    status = wait_state(rt, "error", timeout=15)
    assert status["last_error"]["code"] == "ENGINE_TIMEOUT"
    assert not any(_alive(s["pid"]) for s in env.starts())


def test_idle_stop(env):
    env.settings(idle_stop_minutes=1)
    rt = env.runtime()
    rt.supervisor.start("local:test-a")
    status = wait_state(rt, "ready")
    assert status["idle_stop_at"] is not None
    assert rt.supervisor.check_idle() is False
    rt.supervisor._last_used -= 61
    assert rt.supervisor.check_idle() is True
    assert rt.supervisor.status()["state"] == "stopped"
    assert not _alive(status["pid"])


def test_idle_stop_disabled(env):
    env.settings(idle_stop_minutes=0)
    rt = env.runtime()
    rt.supervisor.start("local:test-a")
    status = wait_state(rt, "ready")
    assert status["idle_stop_at"] is None
    rt.supervisor._last_used -= 10 ** 6
    assert rt.supervisor.check_idle() is False


def test_settings_are_used(env):
    env.settings(context_size=12288, parallel=3, gpu_devices=[0, 1])
    rt = env.runtime()
    rt.supervisor.start("local:test-a")
    status = wait_state(rt, "ready")
    assert status["context_size"] == 12288 and status["parallel"] == 3 and status["gpu_devices"] == [0, 1]
    argv = env.starts()[0]["argv"]
    assert argv[argv.index("--ctx-size") + 1] == str(12288 * 3)
    assert env.starts()[0]["cuda_visible"] == "0,1"


def test_start_errors(env):
    from vvl.llama.supervisor import SupervisorError

    rt = env.runtime()
    with pytest.raises(SupervisorError) as excinfo:
        rt.supervisor.start("local:nope")
    assert excinfo.value.status == 400
    with pytest.raises(SupervisorError) as excinfo:
        rt.supervisor.start("demo:mock")
    assert excinfo.value.status == 400
    rt.models.delete("test-b")
    with pytest.raises(SupervisorError) as excinfo:
        rt.supervisor.start("local:test-b")
    assert excinfo.value.code == "MODEL_NOT_AVAILABLE"


# ------------------------------------------------------------ leases --

def test_concurrent_requests_start_one_engine(env):
    rt = env.runtime()
    handles, errors = [], []

    def use():
        try:
            with rt.supervisor.lease("local:test-a", timeout=15) as handle:
                handles.append(handle)
                time.sleep(0.2)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=use) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not errors
    assert len(handles) == 6
    assert len(env.starts()) == 1
    assert len({h.base_url for h in handles}) == 1
    assert rt.supervisor.status()["active_requests"] == 0


def test_requests_run_up_to_parallel_slots(env):
    from vvl.engines.base import ModelRef, OCRRequest
    from vvl.engines.local import LocalEngine

    env.mode("ready", FAKE_LLAMA_REPLY_DELAY=0.4)
    env.settings(parallel=2)
    rt = env.runtime()
    family, build = rt.local_registry().find_build("test-a")
    image = REPO / "tests" / "fixtures" / "images" / "MICH_16205594_Poaceae_Jouvea_pilosa.jpg"
    results, errors = [], []

    def ocr():
        try:
            engine = LocalEngine(ModelRef("local:test-a", "local", "test-a"), family, build, rt)
            results.append(engine.ocr(OCRRequest(image_path=str(image), prompt_text="Read it.", prompt_key="x")))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=ocr) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors, errors
    assert len(results) == 5
    assert int((env.state / "concurrency.txt").read_text()) == 2
    assert all(r.text.startswith("FAKE OCR TEXT") and r.tokens_in == 1234 for r in results)


def test_switching_models_waits_for_running_requests(env):
    rt = env.runtime()
    order = []
    release_a = threading.Event()

    def use_a():
        with rt.supervisor.lease("local:test-a", timeout=15):
            order.append("a-start")
            release_a.wait(10)
            order.append("a-end")

    def use_b():
        with rt.supervisor.lease("local:test-b", timeout=15) as handle:
            order.append("b-start")
            assert handle.model_name == "test-b"

    ta = threading.Thread(target=use_a)
    ta.start()
    wait_until(lambda: "a-start" in order)
    tb = threading.Thread(target=use_b)
    tb.start()
    time.sleep(0.5)
    assert "b-start" not in order            # B waits while A's request runs
    assert rt.supervisor.status()["loaded_model"] == "local:test-a"
    release_a.set()
    ta.join(10)
    tb.join(20)
    assert order == ["a-start", "a-end", "b-start"]
    aliases = [s["argv"][s["argv"].index("--alias") + 1] for s in env.starts()]
    assert aliases == ["test-a", "test-b"]
    assert not _alive(env.starts()[0]["pid"])


def test_lease_fails_cleanly_when_engine_cannot_start(env):
    from vvl.llama.supervisor import EngineNotReady

    env.mode("crash_on_start")
    rt = env.runtime()
    with pytest.raises(EngineNotReady) as excinfo:
        with rt.supervisor.lease("local:test-a", timeout=15):
            pass
    assert "exit code 3" in excinfo.value.message


def test_lease_without_engine_install(env, tmp_path):
    from vvl.llama.supervisor import EngineNotReady

    rt = env.runtime()
    rt.installer.uninstall()
    with pytest.raises(EngineNotReady):
        with rt.supervisor.lease("local:test-a", timeout=2):
            pass


def test_shutdown_stops_engine(env):
    rt = env.runtime()
    rt.supervisor.start("local:test-a")
    pid = wait_state(rt, "ready")["pid"]
    rt.shutdown()
    assert not _alive(pid)
    from vvl.llama.supervisor import EngineNotReady

    with pytest.raises(EngineNotReady):
        with rt.supervisor.lease("local:test-a", timeout=1):
            pass


# ------------------------------------------------------------- reaper --

def _spawn_stale_engine(port: int, env: Env) -> subprocess.Popen:
    return subprocess.Popen([str(env.server), "--port", str(port), "--alias", "stale"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
                            env={**os.environ, "FAKE_LLAMA_STATE_DIR": str(env.tmp)})


def _write_pidfile(record: dict) -> Path:
    path = Path(os.environ["VVL_HOME"]) / "run" / "engine.pid.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return path


def test_reaper_kills_stale_engine(env):
    import psutil

    proc = _spawn_stale_engine(free_port(), env)
    try:
        wait_until(lambda: psutil.Process(proc.pid).cmdline()[-1:] != [])
        time.sleep(0.3)  # let the wrapper exec into python
        live = psutil.Process(proc.pid)
        path = _write_pidfile({"pid": proc.pid, "create_time": live.create_time(), "cmdline": live.cmdline(),
                               "vvl_pid": 999999999, "vvl_create_time": 1.0})
        rt = env.runtime()
        assert rt.supervisor.reap_stale() == "killed"
        proc.wait(10)
        assert not path.exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_reaper_never_kills_a_reused_pid(env):
    import psutil

    proc = _spawn_stale_engine(free_port(), env)
    try:
        time.sleep(0.3)
        live = psutil.Process(proc.pid)
        # Same pid, but a different start time: some unrelated process now has it.
        path = _write_pidfile({"pid": proc.pid, "create_time": live.create_time() - 1000,
                               "cmdline": live.cmdline()})
        rt = env.runtime()
        assert rt.supervisor.reap_stale() == "reused"
        assert proc.poll() is None
        assert not path.exists()
        # Same pid and start time, different command: also left alone.
        _write_pidfile({"pid": proc.pid, "create_time": live.create_time(), "cmdline": ["something", "else"]})
        assert rt.supervisor.reap_stale() == "reused"
        assert proc.poll() is None
    finally:
        proc.kill()
        proc.wait()


def test_reaper_leaves_engine_of_a_running_vvl(env):
    import psutil

    proc = _spawn_stale_engine(free_port(), env)
    try:
        time.sleep(0.3)
        live = psutil.Process(proc.pid)
        parent = psutil.Process(os.getppid())
        _write_pidfile({"pid": proc.pid, "create_time": live.create_time(), "cmdline": live.cmdline(),
                        "vvl_pid": parent.pid, "vvl_create_time": parent.create_time()})
        assert env.runtime().supervisor.reap_stale() == "owned"
        assert proc.poll() is None
    finally:
        proc.kill()
        proc.wait()


def test_reaper_handles_missing_and_dead(env):
    rt = env.runtime()
    assert rt.supervisor.reap_stale() == "none"
    _write_pidfile({"pid": 2 ** 22 + 12345, "create_time": 1.0, "cmdline": ["x"]})
    assert rt.supervisor.reap_stale() == "gone"
    path = _write_pidfile({"pid": "nope"})
    assert rt.supervisor.reap_stale() == "invalid" and not path.exists()


# -------------------------------------------------- dies with the parent --

PARENT_SCRIPT = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, {repo!r})
    sys.path.insert(0, {tests!r})
    from p2util import make_runtime
    rt = make_runtime({settings!r}, {registry!r}, VVL_ENGINE_PORT_RANGE=({low}, {high}), VVL_ENGINE_LOAD_TIMEOUT=15)
    rt.supervisor.start("local:test-a")
    for _ in range(300):
        status = rt.supervisor.status()
        if status["state"] == "ready":
            print(status["pid"], flush=True)
            break
        time.sleep(0.05)
    time.sleep(120)
""")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-only")
def test_engine_dies_when_vvl_is_sigkilled(env):
    script = env.tmp / "parent.py"
    script.write_text(PARENT_SCRIPT.format(repo=str(REPO), tests=str(REPO / "tests"), settings=str(env.settings_path),
                                           registry=str(env.fake.registry_path), low=env.port_range[0],
                                           high=env.port_range[1]))
    parent = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True, env=dict(os.environ))
    try:
        line = parent.stdout.readline().strip()
        assert line.isdigit(), "parent did not report an engine pid"
        engine_pid = int(line)
        assert _alive(engine_pid)
        parent.send_signal(signal.SIGKILL)
        parent.wait(10)
        wait_until(lambda: not _alive(engine_pid), timeout=10)
    finally:
        if parent.poll() is None:
            parent.kill()


LIFELINE_SCRIPT = textwrap.dedent("""
    import os, sys, time
    sys.path.insert(0, {repo!r})
    from vvl.llama.procutil import Launcher
    proc = Launcher().spawn([{server!r}, "--port", "{port}", "--alias", "x"], env=dict(os.environ), cwd=None,
                            use_lifeline=True)
    print(proc.pid, flush=True)
    time.sleep(120)
""")


def test_lifeline_guardian_kills_engine_when_parent_dies(env):
    """The macOS path (no PDEATHSIG), exercised on any POSIX system."""
    import psutil

    script = env.tmp / "lifeline_parent.py"
    script.write_text(LIFELINE_SCRIPT.format(repo=str(REPO), server=str(env.server), port=free_port()))
    parent = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True,
                              env={**os.environ, "FAKE_LLAMA_STATE_DIR": str(env.state)})
    try:
        guardian_pid = int(parent.stdout.readline().strip())
        engine = wait_until(lambda: psutil.Process(guardian_pid).children())[0]
        engine_pid = engine.pid
        parent.send_signal(signal.SIGKILL)
        parent.wait(10)
        wait_until(lambda: not _alive(engine_pid) and not _alive(guardian_pid), timeout=10)
    finally:
        if parent.poll() is None:
            parent.kill()
