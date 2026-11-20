"""Model downloads (docs/API.md section 9): resume, checksums, cancel, gated models,
disk space, shared files, restart recovery, and the download routes."""

from __future__ import annotations

import json

import pytest

from conftest import TEST_TOKEN, FileRoute
from p2util import FakeModels, make_runtime, sha, wait_job, wait_until, write_settings

pytestmark = pytest.mark.usefixtures("no_network")


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    monkeypatch.setattr("vvl.downloads.RETRY_BACKOFF_SECONDS", (0.01,) * 6)
    monkeypatch.setattr("vvl.downloads.CHUNK_BYTES", 16 * 1024)
    # The test machine's /tmp may be nearly full; disk space is tested explicitly below.
    monkeypatch.setattr("vvl.downloads.free_bytes", lambda path: 10 ** 15)


@pytest.fixture
def setup(tmp_path, fake_server, vvl_env):
    models_dir = tmp_path / "models_dir"
    fake = FakeModels(tmp_path, fake_server)
    write_settings(vvl_env["VVL_SETTINGS_PATH"], models_dir)
    rt = make_runtime(vvl_env["VVL_SETTINGS_PATH"], fake.registry_path, hf_base_url=fake_server.base_url,
                      VVL_PRIVATE_PATH=str(vvl_env["VVL_PRIVATE_PATH"]))
    return {"rt": rt, "fake": fake, "models_dir": models_dir, "server": fake_server}


def _build(rt, build_id):
    return rt.local_registry().find_build(build_id)[1]


def _file(models_dir, name):
    return models_dir / "models" / "TestOrg" / "Test-VL-GGUF" / name


def test_full_download(setup):
    rt, fake, models_dir = setup["rt"], setup["fake"], setup["models_dir"]
    build = _build(rt, "test-a")
    assert rt.models.status(build) == {"install_state": "not_installed", "installed_bytes": 0, "download_job_id": None}
    job = rt.models.start_download("test-a", fake.total("test-a"))
    done = wait_job(job)
    assert done["state"] == "done", done
    assert done["bytes_done"] == done["bytes_total"] == fake.total("test-a")
    assert [f["state"] for f in done["files"]] == ["done", "done"]
    assert _file(models_dir, "model-a.gguf").read_bytes() == fake.data["model-a.gguf"]
    assert not _file(models_dir, "model-a.gguf.partial").exists()
    assert rt.models.status(build) == {"install_state": "installed", "installed_bytes": fake.total("test-a"),
                                       "download_job_id": None}
    index = json.loads((models_dir / "models" / ".vvl-verified.json").read_text())
    assert index["files"]["TestOrg/Test-VL-GGUF/model-a.gguf"]["sha256"] == sha(fake.data["model-a.gguf"])


def test_resume_after_interruption(setup):
    rt, fake, server = setup["rt"], setup["fake"], setup["server"]
    route = server.files[fake.url_path("model-a.gguf")]
    route.cut_after = [100_000, 180_000]  # two dropped connections, then the rest
    job = rt.models.start_download("test-a", fake.total("test-a"))
    done = wait_job(job)
    assert done["state"] == "done", done
    ranges = [r["headers"].get("range") for r in server.requests if r["path"] == fake.url_path("model-a.gguf")]
    assert ranges[0] is None
    # Bytes up to the last whole chunk before each cut are kept and not fetched again.
    first = int(ranges[1][6:-1])
    second = int(ranges[2][6:-1])
    assert 100_000 - 16 * 1024 < first <= 100_000
    assert first < second <= first + 180_000
    assert _file(setup["models_dir"], "model-a.gguf").read_bytes() == fake.data["model-a.gguf"]


def test_restart_recovery_resumes_partial(setup, vvl_env):
    rt, fake, models_dir, server = setup["rt"], setup["fake"], setup["models_dir"], setup["server"]
    partial = _file(models_dir, "model-a.gguf.partial")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(fake.data["model-a.gguf"][:123_456])
    # A fresh runtime (as after a server restart) sees the partial on disk.
    rt2 = make_runtime(vvl_env["VVL_SETTINGS_PATH"], fake.registry_path, hf_base_url=server.base_url)
    status = rt2.models.status(_build(rt2, "test-a"))
    assert status["install_state"] == "partial"
    assert status["installed_bytes"] == 123_456
    done = wait_job(rt2.models.start_download("test-a", fake.total("test-a")))
    assert done["state"] == "done"
    ranges = [r["headers"].get("range") for r in server.requests if r["path"] == fake.url_path("model-a.gguf")]
    assert ranges == ["bytes=123456-"]
    assert _file(models_dir, "model-a.gguf").read_bytes() == fake.data["model-a.gguf"]


def test_complete_but_unverified_file_is_hashed_not_downloaded(setup):
    rt, fake, models_dir, server = setup["rt"], setup["fake"], setup["models_dir"], setup["server"]
    path = _file(models_dir, "model-a.gguf")
    path.parent.mkdir(parents=True)
    path.write_bytes(fake.data["model-a.gguf"])  # e.g. the server stopped right after the rename
    assert rt.models.status(_build(rt, "test-a"))["install_state"] == "partial"
    done = wait_job(rt.models.start_download("test-a", fake.total("test-a")))
    assert done["files"][0]["state"] == "skipped"
    assert not [r for r in server.requests if r["path"] == fake.url_path("model-a.gguf")]


def test_checksum_mismatch_deletes_partial(setup):
    rt, fake, server, models_dir = setup["rt"], setup["fake"], setup["server"], setup["models_dir"]
    data = bytearray(fake.data["model-a.gguf"])
    data[500] ^= 0xFF
    server.files[fake.url_path("model-a.gguf")] = FileRoute(bytes(data))
    done = wait_job(rt.models.start_download("test-a", fake.total("test-a")))
    assert done["state"] == "error"
    assert done["error"]["code"] == "CHECKSUM_MISMATCH"
    assert done["files"][0]["state"] == "error"
    assert not _file(models_dir, "model-a.gguf").exists()
    assert not _file(models_dir, "model-a.gguf.partial").exists()
    status = rt.models.status(_build(rt, "test-a"))
    assert status["install_state"] == "error" and status["download_job_id"] == done["id"]


def test_size_mismatch_from_server(setup):
    rt, fake, server = setup["rt"], setup["fake"], setup["server"]
    server.files[fake.url_path("model-a.gguf")] = FileRoute(fake.data["model-a.gguf"] + b"extra")
    done = wait_job(rt.models.start_download("test-a", fake.total("test-a")))
    assert done["error"]["code"] == "SIZE_MISMATCH"


def test_cancel_then_resume(setup):
    rt, fake, server, models_dir = setup["rt"], setup["fake"], setup["server"], setup["models_dir"]
    server.files[fake.url_path("model-a.gguf")].chunk_delay = 0.02
    job = rt.models.start_download("test-a", fake.total("test-a"))
    wait_until(lambda: job.to_dict()["files"][0]["bytes_done"] > 64 * 1024)
    rt.models.cancel_download("test-a")
    done = wait_job(job)
    assert done["state"] == "canceled"
    partial = _file(models_dir, "model-a.gguf.partial")
    assert partial.exists() and 0 < partial.stat().st_size < len(fake.data["model-a.gguf"])
    kept = partial.stat().st_size
    assert rt.models.status(_build(rt, "test-a"))["install_state"] == "partial"
    server.files[fake.url_path("model-a.gguf")].chunk_delay = 0
    done = wait_job(rt.models.start_download("test-a", fake.total("test-a")))
    assert done["state"] == "done"
    ranges = [r["headers"].get("range") for r in server.requests if r["path"] == fake.url_path("model-a.gguf")]
    assert ranges[-1] == f"bytes={kept}-"
    assert _file(models_dir, "model-a.gguf").read_bytes() == fake.data["model-a.gguf"]


@pytest.mark.parametrize("status", [401, 403])
def test_gated_model(setup, status):
    rt, fake, server = setup["rt"], setup["fake"], setup["server"]
    server.files[fake.url_path("model-a.gguf")] = FileRoute(b"", status=status)
    done = wait_job(rt.models.start_download("test-a", fake.total("test-a")))
    assert done["state"] == "error"
    assert done["error"]["code"] == "GATED_MODEL"
    assert f"HTTP {status}" in done["error"]["message"]
    assert "token" in done["error"]["message"].lower()


def test_hf_token_header_sent_but_never_shown(setup, vvl_env, caplog):
    from vvl.secrets.store import SecretStore

    token = "hf_FAKE_token_value_0123456789"
    SecretStore(vvl_env["VVL_PRIVATE_PATH"]).set_fields("huggingface", {"token": token}, ["token"])
    rt, fake, server = setup["rt"], setup["fake"], setup["server"]
    done = wait_job(rt.models.start_download("test-a", fake.total("test-a")))
    assert done["state"] == "done"
    auth = [r["headers"].get("authorization") for r in server.requests if r["path"].startswith("/TestOrg")]
    assert auth and all(a == f"Bearer {token}" for a in auth)
    assert token not in json.dumps(done)
    assert token not in caplog.text


def test_token_not_forwarded_across_hosts(setup, vvl_env):
    from conftest import FakeServer
    from vvl.secrets.store import SecretStore

    token = "hf_FAKE_token_value_9876543210"
    SecretStore(vvl_env["VVL_PRIVATE_PATH"]).set_fields("huggingface", {"token": token}, ["token"])
    cdn = FakeServer().start()
    try:
        rt, fake, server = setup["rt"], setup["fake"], setup["server"]
        cdn.files["/blob/model-a"] = FileRoute(fake.data["model-a.gguf"])
        server.files[fake.url_path("model-a.gguf")] = FileRoute(b"", redirect_to=cdn.base_url + "/blob/model-a")
        done = wait_job(rt.models.start_download("test-a", fake.total("test-a")))
        assert done["state"] == "done"
        assert cdn.requests and all("authorization" not in r["headers"] for r in cdn.requests)
    finally:
        cdn.stop()


def test_network_error_keeps_partial(setup, monkeypatch):
    rt, fake, server, models_dir = setup["rt"], setup["fake"], setup["server"], setup["models_dir"]
    monkeypatch.setattr("vvl.downloads.MAX_ATTEMPTS_WITHOUT_PROGRESS", 2)
    route = server.files[fake.url_path("model-a.gguf")]
    route.cut_after = [100_000]
    route.statuses = [200]
    route.status = 500          # every request after the first (cut) one fails
    done = wait_job(rt.models.start_download("test-a", fake.total("test-a")))
    assert done["state"] == "error" and done["error"]["code"] == "NETWORK_ERROR"
    partial = _file(models_dir, "model-a.gguf.partial")
    assert partial.exists() and 0 < partial.stat().st_size <= 100_000
    assert partial.read_bytes() == fake.data["model-a.gguf"][:partial.stat().st_size]
    # The next attempt resumes from the partial.
    route.status = 200
    done = wait_job(rt.models.start_download("test-a", fake.total("test-a")))
    assert done["state"] == "done"


def test_shared_file_between_builds(setup):
    rt, fake, server, models_dir = setup["rt"], setup["fake"], setup["server"], setup["models_dir"]
    assert wait_job(rt.models.start_download("test-a", fake.total("test-a")))["state"] == "done"
    status_b = rt.models.status(_build(rt, "test-b"))
    assert status_b["install_state"] == "not_installed"
    assert status_b["installed_bytes"] == len(fake.data["mmproj-shared.gguf"])
    assert rt.models.remaining_bytes(_build(rt, "test-b")) == len(fake.data["model-b.gguf"])
    mmproj_hits = server.files[fake.url_path("mmproj-shared.gguf")].hits
    done = wait_job(rt.models.start_download("test-b", fake.total("test-b")))
    assert [f["state"] for f in done["files"]] == ["done", "skipped"]
    assert server.files[fake.url_path("mmproj-shared.gguf")].hits == mmproj_hits
    # Deleting A keeps the projector B still needs.
    freed = rt.models.delete("test-a")
    assert freed == len(fake.data["model-a.gguf"])
    assert _file(models_dir, "mmproj-shared.gguf").exists()
    assert rt.models.status(_build(rt, "test-b"))["install_state"] == "installed"
    assert rt.models.status(_build(rt, "test-a"))["install_state"] == "not_installed"
    freed = rt.models.delete("test-b")
    assert freed == fake.total("test-b")
    assert not (models_dir / "models" / "TestOrg").exists()


def test_delete_removes_partials(setup):
    rt, fake, models_dir = setup["rt"], setup["fake"], setup["models_dir"]
    partial = _file(models_dir, "model-a.gguf.partial")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"x" * 1000)
    assert rt.models.delete("test-a") == 1000
    assert not partial.exists()


# -------------------------------------------------------------- routes --

@pytest.fixture
def client_setup(make_app, setup, vvl_env):
    flask_app = make_app(VVL_LOCAL_MODELS_PATH=str(setup["fake"].registry_path),
                         VVL_HF_BASE_URL=setup["server"].base_url)
    rt = flask_app.extensions["vvl"]["runtime"]
    return flask_app.test_client(), rt, setup


def test_download_routes(client_setup, auth):
    client, rt, setup = client_setup
    fake = setup["fake"]
    assert client.post("/v1/models/local/test-a/download", headers=auth, json={}).status_code == 400
    response = client.post("/v1/models/local/test-a/download", headers=auth, json={"confirm_size_bytes": 1})
    assert response.status_code == 400
    assert response.get_json()["details"]["total_size_bytes"] == fake.total("test-a")
    assert client.post("/v1/models/local/nope/download", headers=auth,
                       json={"confirm_size_bytes": 1}).status_code == 404
    response = client.post("/v1/models/local/test-a/download", headers=auth,
                           json={"confirm_size_bytes": fake.total("test-a")})
    assert response.status_code == 202
    job = response.get_json()["job"]
    assert job["kind"] == "model_download" and job["target"] == "test-a" and job["label"] == "Test A"
    assert set(job) == {"id", "kind", "target", "label", "state", "bytes_done", "bytes_total", "speed_bps",
                        "eta_seconds", "current_file", "files", "queue_position", "error", "started_at",
                        "finished_at"}
    wait_job(rt.jobs.get(job["id"]))
    body = client.get(f"/v1/jobs/{job['id']}", headers=auth).get_json()
    assert body["state"] == "done"
    assert [j["id"] for j in client.get("/v1/jobs", headers=auth).get_json()["jobs"]] == [job["id"]]
    assert client.get("/v1/jobs/j_nope", headers=auth).status_code == 404
    # Already installed -> 409.
    response = client.post("/v1/models/local/test-a/download", headers=auth,
                           json={"confirm_size_bytes": fake.total("test-a")})
    assert response.status_code == 409 and response.get_json()["error_code"] == "CONFLICT"
    models = client.get("/v1/models", headers=auth).get_json()
    build = models["local"]["families"][0]["builds"][0]
    assert build["install_state"] == "installed"
    assert build["installed_bytes"] == fake.total("test-a")
    assert build["download_job_id"] is None
    assert build["runnable"] is False  # no engine installed
    assert build["unavailable_reason"] == "Install the llama.cpp engine first (Engine tab)."
    response = client.delete("/v1/models/local/test-a", headers=auth)
    assert response.get_json() == {"ok": True, "freed_bytes": fake.total("test-a")}


def test_downloads_wait_in_line(client_setup, auth):
    """A second download waits until the first one ends; then it starts by itself."""
    client, rt, setup = client_setup
    fake, server = setup["fake"], setup["server"]
    server.files[fake.url_path("model-a.gguf")].chunk_delay = 0.05

    def start(build_id):
        return client.post(f"/v1/models/local/{build_id}/download", headers=auth,
                           json={"confirm_size_bytes": fake.total(build_id)})

    def build_state(build_id):
        models = client.get("/v1/models", headers=auth).get_json()
        return next(b for f in models["local"]["families"] for b in f["builds"] if b["id"] == build_id)

    first = start("test-a").get_json()["job"]
    assert first["state"] in ("queued", "running") and first["queue_position"] is None
    again = start("test-a")
    assert again.status_code == 409 and again.get_json()["details"]["active_job_id"] == first["id"]

    second = start("test-b")
    assert second.status_code == 202
    queued = second.get_json()["job"]
    assert queued["state"] == "queued" and queued["queue_position"] == 1
    assert start("test-b").status_code == 409, "the same model cannot be queued twice"
    assert build_state("test-a")["install_state"] == "downloading"
    b = build_state("test-b")
    assert b["install_state"] == "queued" and b["download_job_id"] == queued["id"]
    assert b["unavailable_reason"] == "This model is waiting to download."
    assert client.delete("/v1/models/local/test-b", headers=auth).status_code == 409

    # Canceling a queued download takes it out of line at once.
    canceled = client.post("/v1/models/local/test-b/download/cancel", headers=auth).get_json()["job"]
    assert canceled["id"] == queued["id"] and canceled["state"] == "canceled"
    assert build_state("test-b")["install_state"] == "not_installed"
    assert rt.jobs.get(first["id"]).active, "the running download is not affected"

    # Queue it again, then cancel the running one: the queued one starts by itself.
    requeued = start("test-b").get_json()["job"]
    assert requeued["state"] == "queued"
    client.post("/v1/models/local/test-a/download/cancel", headers=auth)
    assert wait_job(rt.jobs.get(first["id"]))["state"] == "canceled"
    assert wait_job(rt.jobs.get(requeued["id"]))["state"] == "done"
    assert build_state("test-b")["install_state"] == "installed"
    assert build_state("test-a")["install_state"] == "partial", "the canceled download can resume"


def test_queue_runs_in_order(client_setup, auth):
    client, rt, setup = client_setup
    fake, server = setup["fake"], setup["server"]
    server.files[fake.url_path("model-a.gguf")].chunk_delay = 0.01
    order = []
    jobs = []
    for build_id in ("test-a", "test-b"):
        jobs.append(client.post(f"/v1/models/local/{build_id}/download", headers=auth,
                                json={"confirm_size_bytes": fake.total(build_id)}).get_json()["job"])
    for job in jobs:
        done = wait_job(rt.jobs.get(job["id"]))
        assert done["state"] == "done"
        order.append(done["finished_at"])
    assert order == sorted(order), "downloads finish in the order they were requested"


def test_queued_downloads_count_against_disk_space(client_setup, auth, monkeypatch):
    client, rt, setup = client_setup
    fake, server = setup["fake"], setup["server"]
    server.files[fake.url_path("model-a.gguf")].chunk_delay = 0.05
    first = client.post("/v1/models/local/test-a/download", headers=auth,
                        json={"confirm_size_bytes": fake.total("test-a")}).get_json()["job"]
    need_b = fake.total("test-b") + 2 * 1024 ** 3
    monkeypatch.setattr("vvl.downloads.free_bytes", lambda path: need_b + 10)
    response = client.post("/v1/models/local/test-b/download", headers=auth,
                           json={"confirm_size_bytes": fake.total("test-b")})
    assert response.status_code == 507
    assert response.get_json()["details"]["queued_bytes"] > 0
    client.post("/v1/models/local/test-a/download/cancel", headers=auth)
    wait_job(rt.jobs.get(first["id"]))


def test_insufficient_disk(client_setup, auth, monkeypatch):
    client, rt, setup = client_setup
    monkeypatch.setattr("vvl.downloads.free_bytes", lambda path: 1000)
    response = client.post("/v1/models/local/test-a/download", headers=auth,
                           json={"confirm_size_bytes": setup["fake"].total("test-a")})
    assert response.status_code == 507
    body = response.get_json()
    assert body["error_code"] == "INSUFFICIENT_DISK_SPACE"
    assert body["details"] == {"required_bytes": setup["fake"].total("test-a") + 2 * 1024 ** 3, "free_bytes": 1000,
                               "queued_bytes": 0}


def test_cancel_without_job_is_404(client_setup, auth):
    client, _rt, _setup = client_setup
    assert client.post("/v1/models/local/test-a/download/cancel", headers=auth).status_code == 404


def test_download_routes_need_token(client_setup):
    client, _rt, setup = client_setup
    assert client.post("/v1/models/local/test-a/download",
                       json={"confirm_size_bytes": setup["fake"].total("test-a")}).status_code == 401
    assert client.get("/v1/jobs").status_code == 401
    assert client.delete("/v1/models/local/test-a").status_code == 401
    assert TEST_TOKEN  # imported for the security pattern used elsewhere


# ------------------------------------------------------------- job manager --

def test_job_manager_queue_and_busy():
    import threading as _threading

    from vvl.jobs import JobBusy, JobManager

    manager = JobManager()
    release = _threading.Event()
    started = []

    def slow(job):
        started.append(job.target)
        release.wait(10)

    first = manager.submit("model_download", "a", "A", [], slow, queue=True)
    second = manager.submit("model_download", "b", "B", [], slow, queue=True)
    third = manager.submit("model_download", "c", "C", [], slow, queue=True)
    assert [second.queue_position, third.queue_position] == [1, 2]
    assert [j.target for j in manager.active_jobs("model_download")] == ["a", "b", "c"]
    # Engine installs keep refusing a second job.
    install = manager.submit("engine_install", "x", "X", [], lambda job: release.wait(10))
    try:
        manager.submit("engine_install", "y", "Y", [], lambda job: None)
        raise AssertionError("expected JobBusy")
    except JobBusy as busy:
        assert busy.job is install
    manager.cancel(second)
    assert second.state == "canceled" and third.queue_position == 1
    release.set()
    for job in (first, third, install):
        manager.wait(job, 10)
    assert started == ["a", "c"], "the canceled job never ran"
    assert first.state == third.state == "done"


def test_two_processes_cannot_write_the_same_file(setup, tmp_path):
    """A second downloader (another app window) is refused while the first holds the file."""
    from vvl import downloads
    from vvl.jobs import JobError

    dest = tmp_path / "shared" / "model.gguf"
    dest.parent.mkdir()
    holder = downloads._FileLock(downloads.partial_path(dest), "model.gguf")
    with holder:
        try:
            downloads.download_file("http://127.0.0.1:1/model.gguf", dest, size_bytes=10, sha256="0" * 64)
            raise AssertionError("expected the second download to be refused")
        except JobError as exc:
            assert exc.code == "DOWNLOAD_FAILED" and "another VoucherVisionLocal" in exc.message
    assert not holder.path.exists(), "the lock file is removed when the download ends"
