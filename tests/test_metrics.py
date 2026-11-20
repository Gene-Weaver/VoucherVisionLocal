"""GET /v1/metrics: live readings (vvl.system.live) and the memory plan (vvl.llama.plan)."""

from __future__ import annotations

import copy
import os
import threading
import types

import pytest

from conftest import free_port
from p2util import FIXTURES, FakeModels, fake_engine_install, load_report, make_runtime, wait_until, write_settings
from vvl.llama.plan import memory_plan
from vvl.llama.planner import Rules
from vvl.llama.pool import CARD_RESERVE_BYTES
from vvl.system import _proc, live

pytestmark = pytest.mark.usefixtures("no_network")

GiB = 1024 ** 3
MiB = 1024 ** 2
UUID0 = "GPU-1b2e5c9a-0000-4000-8000-000000000001"
UUID1 = "GPU-cefc9b14-0000-4000-8000-000000000002"

SMI_TWO_GPUS = (
    f"0, {UUID0}, 11, 49140, 2883, 45654, 48, NVIDIA RTX 6000 Ada Generation\n"
    f"1, {UUID1}, 97, 49140, 40026, 8514, 71, NVIDIA RTX 6000 Ada Generation\n"
)
SMI_APPS = (
    f"4242, 38000, {UUID1}\n"
    f"777, 1500, {UUID1}\n"
    f"778, 2000, {UUID0}\n"
)


# ---------------------------------------------------------------- parsers --

def test_parse_nvidia_gpus_multi():
    rows = live.parse_nvidia_gpus(SMI_TWO_GPUS)
    assert [r["nvidia_index"] for r in rows] == [0, 1]
    assert rows[0] == {"nvidia_index": 0, "uuid": UUID0, "name": "NVIDIA RTX 6000 Ada Generation",
                       "util_percent": 11.0, "vram_total_bytes": 49140 * MiB, "vram_used_bytes": 2883 * MiB,
                       "vram_free_bytes": 45654 * MiB, "temperature_c": 48.0}
    assert rows[1]["util_percent"] == 97.0 and rows[1]["temperature_c"] == 71.0


def test_parse_nvidia_gpus_placeholders_and_junk():
    text = ("0, GPU-aaa, [N/A], [N/A], [N/A], [N/A], [N/A], NVIDIA GB10\n"
            "\n"
            "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n"
            "x, GPU-bbb, 1, 2, 3, 4, 5, Broken index\n"
            "1, GPU-ccc, 5, 8192, 100, 8092, [Not Supported], Card, with a comma\n")
    rows = live.parse_nvidia_gpus(text)
    assert [r["nvidia_index"] for r in rows] == [0, 1]
    assert rows[0]["util_percent"] is None and rows[0]["vram_total_bytes"] is None
    assert rows[0]["vram_used_bytes"] is None and rows[0]["temperature_c"] is None
    assert rows[1]["name"] == "Card, with a comma" and rows[1]["temperature_c"] is None
    assert live.parse_nvidia_gpus("") == [] and live.parse_nvidia_gpus(None) == []


def test_parse_nvidia_apps():
    apps = live.parse_nvidia_apps(SMI_APPS)
    assert apps[0] == {"pid": 4242, "used_bytes": 38000 * MiB, "gpu_uuid": UUID1}
    assert len(apps) == 3
    assert live.parse_nvidia_apps("") == []
    assert live.parse_nvidia_apps(f"4242, [N/A], {UUID1}\n") == [{"pid": 4242, "used_bytes": None,
                                                                  "gpu_uuid": UUID1}]


# ---------------------------------------------------------------- sampler --

class FakePsutil:
    def __init__(self, percents=(0.0, 12.5, 30.0), fail_first=False):
        self.percents = list(percents)
        self.fail_first = fail_first
        self.calls = 0

    def cpu_percent(self, interval=None):
        assert interval is None, "the sampler must never block on a CPU measurement"
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RuntimeError("no /proc/stat")
        return self.percents.pop(0) if len(self.percents) > 1 else self.percents[0]

    def cpu_count(self, logical=True):
        return 64

    def virtual_memory(self):
        return types.SimpleNamespace(total=64 * GiB, available=48 * GiB, percent=25.0)


class FakeSmi:
    """Stands in for _proc.run; answers the two nvidia-smi queries."""

    def __init__(self, gpus=SMI_TWO_GPUS, apps=SMI_APPS, *, gpus_ok=True, apps_ok=True, timed_out=False):
        self.gpus, self.apps = gpus, apps
        self.gpus_ok, self.apps_ok, self.timed_out = gpus_ok, apps_ok, timed_out
        self.calls = []

    def __call__(self, argv, timeout=None):
        self.calls.append(argv[1].split("=")[0])
        assert timeout is not None and timeout <= 2.0
        if argv[1].startswith("--query-gpu="):
            if self.timed_out:
                return _proc.CmdResult(argv=argv, timed_out=True)
            return _proc.CmdResult(argv=argv, returncode=0 if self.gpus_ok else 9,
                                   stdout=self.gpus if self.gpus_ok else "", stderr="" if self.gpus_ok else "failed")
        return _proc.CmdResult(argv=argv, returncode=0 if self.apps_ok else 1, stdout=self.apps if self.apps_ok else "")


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def make_sampler(smi=None, *, ps=None, clock=None, system="Linux", sys_root="/nonexistent", found=True):
    return live.LiveSampler(run=smi or FakeSmi(), find_tool=lambda name, candidates=(): "/usr/bin/nvidia-smi" if found else None,
                            clock=clock or Clock(), system=system, sys_root=sys_root, psutil_module=ps or FakePsutil())


def test_sample_cpu_memory_and_nvidia():
    smi = FakeSmi()
    sample = make_sampler(smi).sample()
    assert sample["cpu"] == {"percent": 12.5, "logical_cores": 64}
    assert sample["memory"] == {"total_bytes": 64 * GiB, "used_bytes": 16 * GiB, "available_bytes": 48 * GiB,
                                "percent": 25.0}
    assert sample["nvidia"]["state"] == "ok" and len(sample["nvidia"]["gpus"]) == 2
    assert len(sample["nvidia"]["apps"]) == 3
    assert smi.calls == ["--query-gpu", "--query-compute-apps"]
    assert sample["ts"].endswith("Z")


def test_first_cpu_reading_is_not_reported_as_zero():
    # The priming call failed, so the first request makes the first real call: unknown, not 0.0.
    sampler = make_sampler(ps=FakePsutil(percents=(0.0, 7.0), fail_first=True))
    assert sampler.sample()["cpu"]["percent"] is None
    sampler._clock.now += 5
    assert sampler.sample()["cpu"]["percent"] == 7.0


def test_samples_are_cached_between_calls():
    smi, clock = FakeSmi(), Clock()
    sampler = make_sampler(smi, clock=clock)
    first = sampler.sample()
    clock.now += live.MIN_INTERVAL_S - 0.1
    assert sampler.sample() is first
    assert smi.calls.count("--query-gpu") == 1
    clock.now += 0.2
    assert sampler.sample() is not first
    assert smi.calls.count("--query-gpu") == 2
    assert live.MIN_INTERVAL_S >= 1.5


def test_many_clients_share_one_reading():
    smi = FakeSmi()
    sampler = make_sampler(smi)
    results = []
    threads = [threading.Thread(target=lambda: results.append(sampler.sample())) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 12 and all(r is results[0] for r in results)
    assert smi.calls.count("--query-gpu") == 1


@pytest.mark.parametrize("smi", [FakeSmi(gpus_ok=False), FakeSmi(timed_out=True), FakeSmi(gpus="garbage\n")])
def test_nvidia_failure_gives_nulls_and_backs_off(smi):
    clock = Clock()
    sampler = make_sampler(smi, clock=clock)
    sample = sampler.sample()
    assert sample["nvidia"]["gpus"] == [] and sample["nvidia"]["apps"] is None
    assert sample["nvidia"]["state"] in ("error", "timeout")
    assert sample["cpu"]["percent"] is not None and sample["memory"]["total_bytes"] == 64 * GiB
    clock.now += 5
    sampler.sample()
    assert smi.calls.count("--query-gpu") == 1, "a failing nvidia-smi is not called again right away"
    clock.now += live.NVIDIA_RETRY_S
    sampler.sample()
    assert smi.calls.count("--query-gpu") == 2
    # Cards from the system report are still listed, with unknown readings.
    gpus = live.build_gpus(load_report(), sample)
    assert [g["index"] for g in gpus] == [0, 1]
    assert all(g["util_percent"] is None and g["vram_used_bytes"] is None and g["other_bytes"] is None for g in gpus)
    assert gpus[0]["vram_total_bytes"] == load_report()["gpus"][0]["vram_total_bytes"]


def test_nvidia_smi_missing_and_macos_run_nothing():
    smi = FakeSmi()
    assert make_sampler(smi, found=False).sample()["nvidia"]["state"] == "not_found"
    assert make_sampler(smi, system="Darwin").sample()["nvidia"]["state"] == "not_applicable"
    assert smi.calls == []


def test_compute_apps_failure_leaves_engine_share_unknown():
    sample = make_sampler(FakeSmi(apps_ok=False)).sample()
    assert sample["nvidia"]["apps"] is None
    gpus = live.build_gpus(load_report(), sample, engine_pids=[4242], engine_gpu_devices=[1])
    assert gpus[1]["engine_bytes"] is None and gpus[1]["other_bytes"] is None
    assert gpus[1]["vram_used_bytes"] == 40026 * MiB
    assert gpus[0]["engine_bytes"] == 0 and gpus[0]["other_bytes"] == 2883 * MiB


def test_sampler_never_raises():
    def explode(argv, timeout=None):
        raise RuntimeError("boom")

    sample = make_sampler(explode).sample()
    assert sample["nvidia"]["gpus"] == [] and sample["memory"]["total_bytes"] == 64 * GiB


# ---------------------------------------------------------------- build_gpus --

def test_build_gpus_attributes_engine_memory():
    sample = make_sampler().sample()
    gpus = live.build_gpus(load_report(), sample, engine_pids=[4242], engine_gpu_devices=[1])
    assert [(g["index"], g["vendor"], g["kind"]) for g in gpus] == [(0, "nvidia", "discrete"), (1, "nvidia", "discrete")]
    g0, g1 = gpus
    assert g0["engine_bytes"] == 0 and g0["other_bytes"] == 2883 * MiB and g0["util_percent"] == 11.0
    assert g1["engine_bytes"] == 38000 * MiB and g1["other_bytes"] == (40026 - 38000) * MiB
    assert g1["vram_total_bytes"] == 49140 * MiB and g1["vram_free_bytes"] == (49140 - 40026) * MiB
    assert g1["temperature_c"] == 71.0 and g1["memory_source"] == "nvidia-smi"


def test_build_gpus_no_engine():
    gpus = live.build_gpus(load_report(), make_sampler().sample())
    assert all(g["engine_bytes"] == 0 for g in gpus)
    assert gpus[1]["other_bytes"] == gpus[1]["vram_used_bytes"] == 40026 * MiB


def test_build_gpus_without_a_report_uses_the_vendor_list():
    gpus = live.build_gpus(None, make_sampler().sample())
    assert [g["index"] for g in gpus] == [0, 1] and gpus[0]["name"] == "NVIDIA RTX 6000 Ada Generation"
    assert live.build_gpus(None, make_sampler(FakeSmi(gpus_ok=False)).sample()) == []


def test_build_gpus_no_gpu_machine():
    report = load_report(FIXTURES / "system_reports" / "office_pc_cpu_only_8gb_no_avx2.json")
    sample = make_sampler(found=False).sample()
    assert [g for g in live.build_gpus(report, sample) if g["kind"] == "discrete"] == []


def test_build_gpus_apple_silicon_uses_system_memory_and_no_load():
    report = load_report(FIXTURES / "system_reports" / "macbook_air_m2_16gb.json")
    sample = make_sampler(system="Darwin").sample()
    gpus = live.build_gpus(report, sample)
    assert len(gpus) == 1
    g = gpus[0]
    assert g["vendor"] == "apple" and g["util_percent"] is None and g["temperature_c"] is None
    assert g["vram_total_bytes"] == 64 * GiB and g["vram_used_bytes"] == 16 * GiB
    assert g["memory_source"] == "system_ram" and g["engine_bytes"] == 0
    running = live.build_gpus(report, sample, engine_pids=[99])
    assert running[0]["engine_bytes"] is None and running[0]["other_bytes"] is None


def test_build_gpus_amd_sysfs(tmp_path):
    report = load_report(FIXTURES / "system_reports" / "linux_amd_rx7900xtx.json")
    amd = next(g for g in report["gpus"] if g["vendor"] == "amd")
    folder = tmp_path / "sys" / "bus" / "pci" / "devices" / amd["details"]["pci_bus_id"]
    (folder / "hwmon" / "hwmon3").mkdir(parents=True)
    (folder / "gpu_busy_percent").write_text("37\n")
    (folder / "mem_info_vram_total").write_text(str(24 * GiB) + "\n")
    (folder / "mem_info_vram_used").write_text(str(6 * GiB) + "\n")
    (folder / "hwmon" / "hwmon3" / "temp1_input").write_text("56000\n")
    other = tmp_path / "sys" / "bus" / "pci" / "devices" / "0000:00:1f.0"
    other.mkdir()
    assert list(live.read_amd_sysfs(str(tmp_path))) == [amd["details"]["pci_bus_id"]]
    sample = make_sampler(found=False, sys_root=str(tmp_path)).sample()
    g = next(g for g in live.build_gpus(report, sample) if g["vendor"] == "amd")
    assert g["util_percent"] == 37.0 and g["temperature_c"] == 56.0
    assert g["vram_total_bytes"] == 24 * GiB and g["vram_used_bytes"] == 6 * GiB and g["vram_free_bytes"] == 18 * GiB
    assert g["memory_source"] == "sysfs" and g["engine_bytes"] == 0 and g["other_bytes"] == 6 * GiB


def test_build_gpus_unknown_vendor_stays_unknown():
    report = {"gpus": [{"index": 0, "vendor": "intel", "name": "Intel UHD", "kind": "integrated",
                        "vram_total_bytes": None, "details": {}}]}
    g = live.build_gpus(report, make_sampler(found=False).sample())[0]
    assert g["util_percent"] is None and g["vram_total_bytes"] is None and g["vram_used_bytes"] is None
    assert g["vram_free_bytes"] is None and g["other_bytes"] is None and g["memory_source"] is None


# ---------------------------------------------------------------- memory plan --

posix_only = pytest.mark.skipif(os.name == "nt", reason="the fake engine wrapper is a POSIX shell script")


class PlanEnv:
    def __init__(self, tmp_path, vvl_env):
        self.settings_path = vvl_env["VVL_SETTINGS_PATH"]
        self.models_dir = tmp_path / "models_dir"
        self.fake = FakeModels(tmp_path)
        self.fake.install_on_disk(self.models_dir, "test-a")
        self.fake.install_on_disk(self.models_dir, "test-b")
        fake_engine_install(self.models_dir, state_dir=tmp_path)
        self.report = load_report()
        base = free_port()
        self.write()
        self.rt = make_runtime(self.settings_path, self.fake.registry_path, report=self.report,
                               VVL_ENGINE_PORT_RANGE=(base, base + 30), VVL_ENGINE_LOAD_TIMEOUT=15)

    def write(self, selection=None, **engine):
        import yaml

        data = {"models_dir": str(self.models_dir)}
        if selection:
            data["selection"] = selection
        if engine:
            data["engine"] = engine
        self.settings_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    def gpus(self, free=(45 * GiB, 47 * GiB), engine=(0, 0)):
        total = 48 * GiB
        return [{"index": i, "name": "NVIDIA RTX 6000 Ada Generation", "vendor": "nvidia", "kind": "discrete",
                 "vram_total_bytes": total, "vram_used_bytes": total - free[i], "vram_free_bytes": free[i],
                 "engine_bytes": engine[i], "other_bytes": total - free[i] - engine[i]} for i in range(2)]

    def plan(self, gpus=None):
        return memory_plan(self.rt.supervisor, self.rt.settings()["selection"], self.rt.local_registry(),
                           self.report, gpus if gpus is not None else self.gpus())

    def need(self, build_id, slots):
        """Bytes one engine of `build_id` needs with `slots` request slots (the pool's own figure)."""
        model = self.rt.supervisor._need(f"local:{build_id}", ("ocr", "parse"), self.rt.engine_settings())
        return model.unit_bytes(slots)


@pytest.fixture
def env(tmp_path, vvl_env, monkeypatch):
    monkeypatch.delenv("FAKE_LLAMA_MODE", raising=False)
    e = PlanEnv(tmp_path, vvl_env)
    yield e
    e.rt.shutdown(timeout=5)


@posix_only
def test_plan_no_local_model_selected(env):
    for selection in (None, {"ocr": ["demo:mock"], "parse": "demo:mock"},
                      {"ocr": ["api:openai:gpt-4o"], "parse": "demo:mock"}):
        env.write(selection)
        plan = env.plan()
        assert plan["mode"] == "none" and plan["models"] == []
        assert plan["gpus"] == {0: {"planned_bytes": 0, "planned_for": [], "planned_fits": True},
                                1: {"planned_bytes": 0, "planned_for": [], "planned_fits": True}}
        assert "No local model" in plan["note"]


@posix_only
def test_plan_one_model_runs_on_every_card(env):
    env.write({"ocr": ["local:test-a"], "parse": "local:test-a"})
    plan = env.plan()
    assert plan["mode"] == "single"
    model = plan["models"][0]
    assert (model["model_id"], model["label"], model["roles"], model["loaded"]) == \
        ("local:test-a", "Test A", ["ocr", "parse"], False)
    # One engine per card, each grown to the slot limit: spare memory is used.
    slots = Rules().max_slots
    per_card = env.need("test-a", slots)
    assert model["gpu_indices"] == [0, 1] and model["required_bytes"] == 2 * per_card
    for index in (0, 1):
        assert plan["gpus"][index] == {"planned_bytes": per_card, "planned_fits": True,
                                       "planned_for": [{"model_id": "local:test-a", "label": "Test A",
                                                        "bytes": per_card}]}


@posix_only
def test_plan_two_models_that_fit_stay_loaded_together(env):
    env.write({"ocr": ["local:test-a"], "parse": "local:test-b"})
    plan = env.plan()
    assert plan["mode"] == "shared" and "side by side" in plan["note"]
    assert [(m["model_id"], m["roles"]) for m in plan["models"]] == [("local:test-a", ["ocr"]),
                                                                     ("local:test-b", ["parse"])]
    assert all(m["gpu_indices"] for m in plan["models"])
    for card in plan["gpus"].values():
        assert card["planned_bytes"] == sum(p["bytes"] for p in card["planned_for"]) and card["planned_fits"] is True
    total = sum(card["planned_bytes"] for card in plan["gpus"].values())
    assert total == sum(m["required_bytes"] for m in plan["models"])


@posix_only
def test_plan_two_models_that_do_not_fit_take_turns(env):
    env.write({"ocr": ["local:test-a"], "parse": "local:test-b"})
    one_a, one_b = env.need("test-a", 1), env.need("test-b", 1)
    # Each card has room for one of the two models, but only one card is usable.
    room = max(one_a, one_b) + CARD_RESERVE_BYTES + 1
    env.write({"ocr": ["local:test-a"], "parse": "local:test-b"}, gpu_devices=[1])
    plan = env.plan(env.gpus(free=(45 * GiB, room)))
    assert plan["mode"] == "exclusive" and "one is loaded at a time" in plan["note"]
    card = plan["gpus"][1]
    assert [p["model_id"] for p in card["planned_for"]] == ["local:test-a", "local:test-b"]
    assert card["planned_bytes"] == max(p["bytes"] for p in card["planned_for"]), "the larger one, not the sum"
    assert card["planned_bytes"] < sum(p["bytes"] for p in card["planned_for"])
    assert plan["gpus"][0]["planned_bytes"] == 0


@posix_only
def test_plan_several_ocr_models(env):
    env.write({"ocr": ["local:test-b", "demo:mock", "local:test-a"], "parse": "local:test-b"})
    plan = env.plan()
    assert plan["mode"] == "shared"
    assert [(m["model_id"], m["roles"]) for m in plan["models"]] == [("local:test-b", ["ocr", "parse"]),
                                                                     ("local:test-a", ["ocr"])]


@posix_only
def test_plan_model_already_loaded(env):
    env.write({"ocr": ["local:test-a"], "parse": "local:test-a"}, gpu_devices=[1], parallel=2)
    env.rt.supervisor.start("local:test-a")
    wait_until(lambda: env.rt.supervisor.state() == "ready")
    status = env.rt.supervisor.status()
    assert status["gpu_devices"] == [1] and status["loaded_models"] == ["local:test-a"]
    engine_bytes = 3 * GiB
    plan = env.plan(env.gpus(free=(45 * GiB, 44 * GiB), engine=(0, engine_bytes)))
    assert plan["models"][0]["loaded"] is True and plan["models"][0]["gpu_indices"] == [1]
    assert plan["gpus"][1] == {"planned_bytes": 0, "planned_for": [], "planned_fits": True}


@posix_only
def test_plan_explicit_gpu_devices(env):
    env.write({"ocr": ["local:test-a"], "parse": "local:test-a"}, gpu_devices=[0])
    plan = env.plan()
    assert plan["models"][0]["gpu_indices"] == [0]
    assert plan["gpus"][0]["planned_bytes"] == env.need("test-a", Rules().max_slots)
    assert plan["gpus"][1]["planned_bytes"] == 0


@posix_only
def test_plan_follows_engine_settings(env):
    env.write({"ocr": ["local:test-a"], "parse": "local:test-a"}, context_size=8192, parallel=3, gpu_devices=[0])
    assert env.plan()["models"][0]["required_bytes"] == env.need("test-a", 3)
    env.write({"ocr": ["local:test-a"], "parse": "local:test-a"}, context_size=4096, parallel=1, gpu_devices=[0])
    small = env.plan()["models"][0]["required_bytes"]
    assert small == env.need("test-a", 1) and small < 3 * GiB


@posix_only
def test_plan_model_larger_than_free_memory(env):
    env.write({"ocr": ["local:test-a"], "parse": "local:test-a"}, gpu_devices=[1], parallel=1)
    need = env.need("test-a", 1)
    plan = env.plan(env.gpus(free=(45 * GiB, need - 1)))
    assert plan["gpus"][1]["planned_bytes"] == need and plan["gpus"][1]["planned_fits"] is False
    assert plan["gpus"][0]["planned_fits"] is True

    # Too little room on every card: the model is spread over all of them.
    env.write({"ocr": ["local:test-a"], "parse": "local:test-a"}, parallel=1)
    plan = env.plan(env.gpus(free=(need // 2, need // 4)))
    assert plan["models"][0]["gpu_indices"] == [0, 1]
    assert [plan["gpus"][i]["planned_fits"] for i in (0, 1)] == [False, False]


@posix_only
def test_plan_model_not_installed_or_unknown(env):
    env.write({"ocr": ["local:nope"], "parse": "local:nope"})
    plan = env.plan()
    assert plan["models"] == [{"model_id": "local:nope", "label": "local:nope", "roles": ["ocr", "parse"],
                               "required_bytes": None, "gpu_indices": [], "loaded": False}]
    assert all(c["planned_bytes"] == 0 for c in plan["gpus"].values())


def test_plan_cpu_engine_build_plans_no_graphics_memory(tmp_path, vvl_env):
    models_dir = tmp_path / "models_dir"
    fake = FakeModels(tmp_path)
    fake.install_on_disk(models_dir, "test-a")
    fake_engine_install(models_dir, variant="linux-x64-cpu", backend="cpu")
    write_settings(vvl_env["VVL_SETTINGS_PATH"], models_dir)
    rt = make_runtime(vvl_env["VVL_SETTINGS_PATH"], fake.registry_path)
    try:
        gpus = [{"index": 0, "vendor": "nvidia", "kind": "discrete", "vram_free_bytes": 40 * GiB, "engine_bytes": 0}]
        plan = memory_plan(rt.supervisor, {"ocr": ["local:test-a"], "parse": "local:test-a"}, rt.local_registry(),
                           load_report(), gpus)
    finally:
        rt.shutdown(timeout=5)
    assert plan["models"][0]["gpu_indices"] == [] and plan["gpus"][0]["planned_bytes"] == 0
    assert "processor" in plan["note"]


# ---------------------------------------------------------------- route --

@pytest.fixture
def metrics_app(tmp_path, vvl_env, make_app, monkeypatch):
    monkeypatch.delenv("FAKE_LLAMA_MODE", raising=False)
    e = PlanEnv(tmp_path, vvl_env)
    e.rt.shutdown(timeout=5)
    base = free_port()
    report = load_report()
    smi = FakeSmi()
    app = make_app(VVL_LOCAL_MODELS_PATH=str(e.fake.registry_path), VVL_ENGINE_PORT_RANGE=(base, base + 30),
                   VVL_ENGINE_LOAD_TIMEOUT=15, VVL_SYSTEM_REPORT=lambda refresh=False: copy.deepcopy(report),
                   VVL_LIVE_SAMPLER=make_sampler(smi))
    rt = app.extensions["vvl"]["runtime"]
    yield types.SimpleNamespace(client=app.test_client(), rt=rt, env=e, smi=smi, app=app)
    rt.shutdown(timeout=5)


def test_metrics_requires_the_token(metrics_app):
    assert metrics_app.client.get("/v1/metrics").status_code == 401


@posix_only
def test_metrics_route_shape(metrics_app, auth):
    metrics_app.env.write({"ocr": ["local:test-a"], "parse": "local:test-b"})
    res = metrics_app.client.get("/v1/metrics", headers=auth)
    assert res.status_code == 200
    body = res.get_json()
    assert set(body) == {"ts", "cpu", "memory", "gpus", "engine", "plan"}
    assert body["cpu"] == {"percent": 12.5, "logical_cores": 64}
    assert body["memory"]["total_bytes"] == 64 * GiB and body["memory"]["used_bytes"] == 16 * GiB
    assert body["engine"] == {"state": "stopped", "loaded_model": None, "gpu_devices": None}
    assert set(body["plan"]) == {"mode", "note", "models"} and body["plan"]["mode"] == "shared"
    g0, g1 = body["gpus"]
    assert set(g1) == {"index", "name", "vendor", "kind", "util_percent", "vram_total_bytes", "vram_used_bytes",
                       "vram_free_bytes", "engine_bytes", "other_bytes", "planned_bytes", "planned_for",
                       "planned_fits", "temperature_c", "memory_source"}
    assert g1["engine_bytes"] == 0 and g1["other_bytes"] == 40026 * MiB
    # Both models fit together; GPU 0 has the most free memory in this sample.
    assert g0["planned_bytes"] > 0 and g0["planned_fits"] is True
    assert {p["label"] for g in (g0, g1) for p in g["planned_for"]} == {"Test A", "Test B"}
    assert all(m["gpu_indices"] for m in body["plan"]["models"])


@posix_only
def test_metrics_route_with_a_loaded_model(metrics_app, auth):
    metrics_app.env.write({"ocr": ["local:test-a"], "parse": "local:test-a"}, gpu_devices=[1])
    metrics_app.rt.supervisor.start("local:test-a")
    wait_until(lambda: metrics_app.rt.supervisor.state() == "ready")
    pid = metrics_app.rt.supervisor.status()["pid"]
    metrics_app.smi.apps = f"{pid}, 38000, {UUID1}\n777, 1500, {UUID1}\n"
    metrics_app.app.config["VVL_LIVE_SAMPLER"] = make_sampler(metrics_app.smi)
    body = metrics_app.client.get("/v1/metrics", headers=auth).get_json()
    assert body["engine"] == {"state": "ready", "loaded_model": "local:test-a", "gpu_devices": [1]}
    g1 = body["gpus"][1]
    assert g1["engine_bytes"] == 38000 * MiB and g1["other_bytes"] == (40026 - 38000) * MiB
    assert g1["planned_bytes"] == 0 and g1["planned_for"] == []
    assert body["plan"]["models"][0]["loaded"] is True and body["plan"]["mode"] == "single"


def test_metrics_route_survives_missing_hardware_data(metrics_app, auth):
    metrics_app.app.config["VVL_LIVE_SAMPLER"] = make_sampler(FakeSmi(timed_out=True))
    metrics_app.app.config["VVL_SYSTEM_REPORT"] = lambda refresh=False: (_ for _ in ()).throw(RuntimeError("no report"))
    res = metrics_app.client.get("/v1/metrics", headers=auth)
    assert res.status_code == 200
    body = res.get_json()
    assert body["cpu"]["percent"] == 12.5 and body["plan"]["mode"] in ("none", "single", "alternating")
    for g in body["gpus"]:
        assert g["util_percent"] is None and g["vram_used_bytes"] is None
