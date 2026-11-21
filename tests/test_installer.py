"""Engine installer: variant selection, safe extraction, self-test, uninstall."""

from __future__ import annotations

import copy
import io
import json
import os
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

from conftest import FileRoute
from p2util import FAKE_LLAMA, FIXTURES, load_report, make_runtime, sha, wait_job, write_settings

pytestmark = pytest.mark.usefixtures("no_network")

REPORTS = FIXTURES / "system_reports"


# ------------------------------------------------------- variant selection --
#
# Every call passes an explicit glibc, so the results are the same on any test
# machine. MODERN passes the lock's `glibc_min: '2.38'` on the Linux CUDA builds.

MODERN_GLIBC = "2.39"
UBUNTU_2204_GLIBC = "2.35"


def _variants(report, *, glibc, **kwargs):
    from vvl.llama import variants
    from vvl.llama.lock import load_lock

    assessed = variants.assess(load_lock(), report, glibc=glibc, **kwargs)
    return assessed, variants.recommend(assessed)


def _synth(base: str, **changes) -> dict:
    report = copy.deepcopy(load_report(REPORTS / base))
    for path, value in changes.items():
        target = report
        keys = path.split(".")
        for key in keys[:-1]:
            target = target[int(key)] if key.isdigit() else target[key]
        last = keys[-1]
        if last.isdigit():
            target[int(last)] = value
        else:
            target[last] = value
    from vvl.system import capability

    capability.apply(report)
    return report


@pytest.mark.parametrize("fixture,expected,listed", [
    ("linux_workstation_2x_rtx6000ada.json", "linux-x64-cuda12",
     ["linux-x64-cuda13", "linux-x64-cuda12", "linux-x64-vulkan", "linux-x64-cpu"]),
    ("wsl2_ubuntu_rtx3090.json", "linux-x64-cuda12", None),
    ("windows_laptop_rtx4060_8gb.json", "windows-x64-cuda12",
     ["windows-x64-cuda13", "windows-x64-cuda12", "windows-x64-vulkan", "windows-x64-cpu"]),
    ("linux_amd_rx7900xtx.json", "linux-x64-vulkan", None),
    ("office_pc_cpu_only_8gb_no_avx2.json", "windows-x64-cpu", None),
    ("macbook_air_m2_16gb.json", "macos-arm64-metal", ["macos-arm64-metal"]),
    ("macbook_pro_m3_max_64gb_rosetta_dropbox.json", "macos-arm64-metal", ["macos-arm64-metal"]),
    ("intel_mac_2019_radeon_pro.json", "macos-x64-cpu", ["macos-x64-cpu"]),
])
def test_variant_for_fixture_reports(fixture, expected, listed):
    assessed, rec = _variants(load_report(REPORTS / fixture), glibc=MODERN_GLIBC)
    assert rec["id"] == expected
    if listed:
        assert [v["id"] for v in assessed] == listed
    assert rec["reason"]


def test_cuda13_needs_new_driver():
    assessed, rec = _variants(load_report(REPORTS / "linux_workstation_2x_rtx6000ada.json"), glibc=MODERN_GLIBC)
    cuda13 = assessed[0]
    assert cuda13["compatible"] is False
    assert cuda13["reason"] == "Needs NVIDIA driver 580.0 or newer; this computer has 560.35.05."
    assert "CUDA 13" in rec["reason"]


def test_new_driver_gets_cuda13():
    report = _synth("linux_workstation_2x_rtx6000ada.json", **{"accelerators.cuda.driver_version": "580.82.07",
                                                               "gpus.0.driver_version": "580.82.07"})
    assert _variants(report, glibc=MODERN_GLIBC)[1]["id"] == "linux-x64-cuda13"


def test_old_driver_falls_back_to_vulkan():
    report = _synth("linux_workstation_2x_rtx6000ada.json", **{"accelerators.cuda.driver_version": "470.239.06"})
    assessed, rec = _variants(report, glibc=MODERN_GLIBC)
    assert rec["id"] == "linux-x64-vulkan"
    assert "525.60" in assessed[1]["reason"]


def test_unknown_driver_version_still_allows_cuda():
    report = _synth("linux_workstation_2x_rtx6000ada.json", **{"accelerators.cuda.driver_version": None})
    for gpu in report["gpus"]:
        gpu["driver_version"] = None
    assessed, rec = _variants(report, glibc=MODERN_GLIBC)
    assert rec["id"] == "linux-x64-cuda13"   # cannot rule it out; the self-test decides
    assert "could not be read" in rec["reason"]


def test_linux_arm64_spark_like():
    report = _synth("linux_workstation_2x_rtx6000ada.json", **{"os.arch": "aarch64",
                                                               "accelerators.cuda.driver_version": "580.95.05"})
    assessed, rec = _variants(report, glibc=MODERN_GLIBC)
    assert [v["id"] for v in assessed] == ["linux-arm64-cuda13", "linux-arm64-vulkan", "linux-arm64-cpu"]
    assert rec["id"] == "linux-arm64-cuda13"
    report = _synth("linux_workstation_2x_rtx6000ada.json", **{"os.arch": "aarch64"})
    assert _variants(report, glibc=MODERN_GLIBC)[1]["id"] == "linux-arm64-vulkan"   # 560 driver: no CUDA 13


def test_windows_arm64():
    report = _synth("office_pc_cpu_only_8gb_no_avx2.json", **{"os.arch": "ARM64"})
    assessed, rec = _variants(report, glibc=MODERN_GLIBC)
    assert [v["id"] for v in assessed] == ["windows-arm64-vulkan", "windows-arm64-cpu"]
    assert rec["id"] == "windows-arm64-cpu"


def test_unsupported_arch_lists_nothing():
    report = _synth("office_pc_cpu_only_8gb_no_avx2.json", **{"os.system": "Linux", "os.arch": "armv7l"})
    assessed, rec = _variants(report, glibc=MODERN_GLIBC)
    assert assessed == [] and rec is None


def test_broken_nvidia_driver():
    report = _synth("linux_workstation_2x_rtx6000ada.json")
    for gpu in report["gpus"]:
        gpu["source"] = "sysfs"
    report["accelerators"]["cuda"]["state"] = "driver_not_loaded"
    from vvl.system import capability

    capability.apply(report)
    assessed, rec = _variants(report, glibc=MODERN_GLIBC)
    assert assessed[1]["reason"] == "The NVIDIA driver is not working, so CUDA cannot be used."
    assert rec["id"] == "linux-x64-cpu"


def test_selftest_failure_moves_recommendation_until_driver_changes():
    # A glibc that passes the lock's pre-check, so the recorded failure is what moves the choice.
    report = load_report(REPORTS / "linux_workstation_2x_rtx6000ada.json")
    assert _variants(report, glibc=MODERN_GLIBC)[1]["id"] == "linux-x64-cuda12"
    failures = {"linux-x64-cuda12": {"message": "no kernel image for this GPU", "driver_version": "560.35.05"}}
    assessed, rec = _variants(report, glibc=MODERN_GLIBC, failures=failures)
    assert rec["id"] == "linux-x64-vulkan"
    assert assessed[1]["compatible"] is False
    assert assessed[1]["reason"] == "Failed its self-test on this computer: no kernel image for this GPU"
    failures["linux-x64-cuda12"]["driver_version"] = "550.00"   # recorded under another driver
    assert _variants(report, glibc=MODERN_GLIBC, failures=failures)[1]["id"] == "linux-x64-cuda12"


# --------------------------------------------------------------- extraction --

def _tar(path: Path, members: list[tuple]) -> Path:
    """members: (name, kind, payload_or_target, mode)"""
    with tarfile.open(path, "w:gz") as tar:
        for name, kind, payload, mode in members:
            info = tarfile.TarInfo(name)
            info.mode = mode
            if kind == "file":
                data = payload if isinstance(payload, bytes) else payload.encode()
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            elif kind == "dir":
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = payload
                tar.addfile(info)
            elif kind == "hardlink":
                info.type = tarfile.LNKTYPE
                info.linkname = payload
                tar.addfile(info)
            elif kind == "fifo":
                info.type = tarfile.FIFOTYPE
                tar.addfile(info)
            elif kind == "chr":
                info.type = tarfile.CHRTYPE
                tar.addfile(info)
    return path


def _zip(path: Path, members: list[tuple]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, kind, payload, mode in members:
            info = zipfile.ZipInfo(name)
            if kind == "file":
                info.external_attr = (stat.S_IFREG | mode) << 16
                zf.writestr(info, payload)
            elif kind == "symlink":
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                zf.writestr(info, payload)
            elif kind == "dir":
                info.external_attr = (stat.S_IFDIR | 0o755) << 16
                zf.writestr(info, b"")
    return path


def test_tar_extract_keeps_exec_bits_and_inner_links(tmp_path):
    from vvl.llama.extract import extract

    archive = _tar(tmp_path / "a.tar.gz", [
        ("pkg", "dir", None, 0o755),
        ("pkg/llama-server", "file", "#!/bin/sh\n", 0o755),
        ("pkg/libx.so.1.0", "file", "lib", 0o644),
        ("pkg/libx.so.1", "symlink", "libx.so.1.0", 0o777),
        ("pkg/hard", "hardlink", "pkg/libx.so.1.0", 0o644),
        ("pkg/setuid", "file", "x", 0o4777),
    ])
    out = tmp_path / "out"
    extract(archive, out)
    assert os.access(out / "pkg" / "llama-server", os.X_OK)
    assert os.readlink(out / "pkg" / "libx.so.1") == "libx.so.1.0"
    assert (out / "pkg" / "hard").read_text() == "lib"
    mode = (out / "pkg" / "setuid").stat().st_mode
    assert not mode & (stat.S_ISUID | stat.S_IWGRP | stat.S_IWOTH)


@pytest.mark.parametrize("members", [
    [("../evil", "file", "x", 0o644)],
    [("pkg/../../evil", "file", "x", 0o644)],
    [("/abs/evil", "file", "x", 0o644)],
    [("pkg/link", "symlink", "../../outside", 0o777)],
    [("pkg/link", "symlink", "/etc/passwd", 0o777)],
    [("pkg/hard", "hardlink", "../outside", 0o644)],
    [("pkg/fifo", "fifo", None, 0o644)],
    [("pkg/dev", "chr", None, 0o644)],
    [("pkg/d", "symlink", ".", 0o777), ("pkg/d/../../x", "file", "x", 0o644)],
])
def test_tar_rejects_unsafe_members(tmp_path, members):
    from vvl.llama.extract import UnsafeArchive, extract

    archive = _tar(tmp_path / "bad.tar.gz", members)
    out = tmp_path / "deep" / "out"
    with pytest.raises(UnsafeArchive):
        extract(archive, out)
    assert not (tmp_path / "evil").exists() and not (tmp_path / "deep" / "evil").exists()
    assert not (tmp_path / "deep" / "x").exists()


def test_tar_symlink_dir_then_write_through_it_is_rejected(tmp_path):
    from vvl.llama.extract import UnsafeArchive, extract

    outside = tmp_path / "outside"
    outside.mkdir()
    # A link that resolves inside is allowed, but a link chain that points out is not.
    archive = _tar(tmp_path / "chain.tar.gz", [
        ("a", "symlink", "b", 0o777),
        ("b", "symlink", "../outside", 0o777),
    ])
    with pytest.raises(UnsafeArchive):
        extract(archive, tmp_path / "out")
    assert list(outside.iterdir()) == []


def test_zip_extract_keeps_exec_bits(tmp_path):
    from vvl.llama.extract import extract

    archive = _zip(tmp_path / "a.zip", [
        ("pkg/", "dir", None, 0o755),
        ("pkg/llama-server.exe", "file", b"MZ", 0o755),
        ("pkg/readme.txt", "file", b"hi", 0o644),
    ])
    out = tmp_path / "out"
    extract(archive, out)
    assert (out / "pkg" / "llama-server.exe").read_bytes() == b"MZ"
    if os.name != "nt":
        assert os.access(out / "pkg" / "llama-server.exe", os.X_OK)
        assert not os.access(out / "pkg" / "readme.txt", os.X_OK)


@pytest.mark.parametrize("members", [
    [("../evil", "file", b"x", 0o644)],
    [("/abs/evil", "file", b"x", 0o644)],
    [("C:\\evil.txt", "file", b"x", 0o644)],
    [("pkg\\..\\..\\evil", "file", b"x", 0o644)],
    [("pkg/link", "symlink", "../../outside", 0o777)],
])
def test_zip_rejects_unsafe_members(tmp_path, members):
    from vvl.llama.extract import UnsafeArchive, extract

    archive = _zip(tmp_path / "bad.zip", members)
    with pytest.raises(UnsafeArchive):
        extract(archive, tmp_path / "deep" / "out")
    assert not (tmp_path / "evil").exists() and not (tmp_path / "deep" / "evil").exists()


# --------------------------------------------------------------- install job --

FAKE_SERVER_SCRIPT = (
    '#!/bin/sh\n: "${{FAKE_LLAMA_MODE:=ready}}"\nexport FAKE_LLAMA_MODE\n'
    'exec "{python}" -I "{script}" "$@"\n'
)


class FakeRelease:
    """A lock file with three variants whose archives the FakeServer serves."""

    def __init__(self, tmp_path, server):
        import sys

        script = FAKE_SERVER_SCRIPT.format(python=sys.executable, script=FAKE_LLAMA)
        self.engine_tar = self._bytes_of(_tar, tmp_path / "engine.tar.gz", [
            ("llama-b12359", "dir", None, 0o755),
            ("llama-b12359/llama-server", "file", script, 0o755),
            ("llama-b12359/libggml-base.so.1.0.0", "file", "lib", 0o755),
            ("llama-b12359/libggml-base.so.1", "symlink", "libggml-base.so.1.0.0", 0o777),
        ])
        self.cudart_tar = self._bytes_of(_tar, tmp_path / "cudart.tar.gz", [
            ("libcudart.so.12", "file", "cudart", 0o644),
            ("lib/libcublas.so.12", "file", "cublas", 0o644),
        ])
        self.cpu_zip = self._bytes_of(_zip, tmp_path / "cpu.zip", [
            ("llama-server", "file", script.encode(), 0o755),
        ])
        files = {"engine.tar.gz": self.engine_tar, "cudart.tar.gz": self.cudart_tar, "cpu.zip": self.cpu_zip}
        for name, data in files.items():
            server.files[f"/rel/{name}"] = FileRoute(data)

        def asset(kind, name):
            data = files[name]
            return {"kind": kind, "name": name, "url": f"{server.base_url}/rel/{name}", "size_bytes": len(data),
                    "sha256": sha(data)}

        lock = {
            "version": 1,
            "release": {"tag": "b12359", "repository": "https://example.invalid/llama.cpp"},
            "variants": [
                {"id": "linux-x64-cuda12", "os": "linux", "arch": "x86_64", "backend": "cuda",
                 "label": "Linux x64, NVIDIA CUDA 12", "requires": {"nvidia_driver_min": "525.60"},
                 "assets": [asset("engine", "engine.tar.gz"), asset("runtime", "cudart.tar.gz")]},
                {"id": "linux-x64-vulkan", "os": "linux", "arch": "x86_64", "backend": "vulkan",
                 "label": "Linux x64, Vulkan", "requires": {}, "assets": [asset("engine", "engine.tar.gz")]},
                {"id": "linux-x64-cpu", "os": "linux", "arch": "x86_64", "backend": "cpu",
                 "label": "Linux x64, CPU", "requires": {}, "assets": [asset("engine", "cpu.zip")]},
            ],
        }
        self.lock_path = tmp_path / "engine.lock.yaml"
        self.lock_path.write_text(yaml.safe_dump(lock), encoding="utf-8")

    @staticmethod
    def _bytes_of(builder, path, members):
        builder(path, members)
        return path.read_bytes()


@pytest.fixture
def release(tmp_path, fake_server, vvl_env, monkeypatch):
    monkeypatch.delenv("FAKE_LLAMA_MODE", raising=False)
    monkeypatch.setattr("vvl.downloads.free_bytes", lambda path: 10 ** 15)
    rel = FakeRelease(tmp_path, fake_server)
    models_dir = tmp_path / "models_dir"
    write_settings(vvl_env["VVL_SETTINGS_PATH"], models_dir)
    rt = make_runtime(vvl_env["VVL_SETTINGS_PATH"], FIXTURES.parent.parent / "vvl" / "registry" / "models_local.yaml",
                      VVL_ENGINE_LOCK_PATH=str(rel.lock_path))
    return {"rt": rt, "rel": rel, "models_dir": models_dir, "server": fake_server}


@pytest.mark.skipif(os.name == "nt", reason="the fake engine is a POSIX shell script")
def test_install_cuda_build_with_runtime_libs(release):
    rt, models_dir = release["rt"], release["models_dir"]
    assert rt.installer.recommended()["id"] == "linux-x64-cuda12"
    job = rt.installer.start_install(None)
    done = wait_job(job)
    assert done["state"] == "done", done["error"]
    assert [f["state"] for f in done["files"]] == ["done", "done"]
    folder = models_dir / "engine" / "b12359" / "linux-x64-cuda12"
    server = folder / "llama-b12359" / "llama-server"
    assert os.access(server, os.X_OK)
    assert (server.parent / "libcudart.so.12").read_text() == "cudart"
    assert (server.parent / "libcublas.so.12").read_text() == "cublas"     # flattened beside the engine
    manifest = json.loads((folder / "vvl-install.json").read_text())
    assert manifest["variant"] == "linux-x64-cuda12" and manifest["backend"] == "cuda"
    assert manifest["server"] == "llama-b12359/llama-server"
    assert manifest["version"] == "b12359 (abc1234)"
    assert [d["name"] for d in manifest["devices"]] == ["CUDA0", "CUDA1", "Vulkan0", "Vulkan1"]
    assert not (models_dir / "engine" / "b12359" / "downloads").exists()
    assert not [p for p in (models_dir / "engine" / "b12359").iterdir() if p.name.startswith(".staging")]
    assessed = rt.installer.assess()
    assert next(v for v in assessed if v["id"] == "linux-x64-cuda12")["installed"] is True
    assert rt.installer.active_install()["variant"] == "linux-x64-cuda12"
    assert rt.supervisor.status()["state"] == "stopped"


@pytest.mark.skipif(os.name == "nt", reason="the fake engine is a POSIX shell script")
def test_install_zip_build(release):
    rt, models_dir = release["rt"], release["models_dir"]
    done = wait_job(rt.installer.start_install("linux-x64-cpu"))
    assert done["state"] == "done", done["error"]
    assert os.access(models_dir / "engine" / "b12359" / "linux-x64-cpu" / "llama-server", os.X_OK)


@pytest.mark.skipif(os.name == "nt", reason="the fake engine is a POSIX shell script")
def test_self_test_failure_names_next_variant(release, monkeypatch):
    rt, models_dir = release["rt"], release["models_dir"]
    monkeypatch.setenv("FAKE_LLAMA_MODE", "selftest_glibc")
    done = wait_job(rt.installer.start_install(None))
    assert done["state"] == "error"
    err = done["error"]
    assert err["code"] == "ENGINE_SELF_TEST_FAILED"
    assert "glibc 2.38" in err["message"]
    assert "Try Linux x64, Vulkan (linux-x64-vulkan) instead." in err["message"]
    assert err["details"]["next_variant"] == "linux-x64-vulkan"
    assert any("GLIBC_2.38" in line for line in err["details"]["output_tail"])
    tag_dir = models_dir / "engine" / "b12359"
    assert not (tag_dir / "linux-x64-cuda12").exists()
    assert not [p for p in tag_dir.iterdir() if p.name.startswith(".staging")]
    assert not (tag_dir / "downloads").exists()    # verified but unusable archives are removed
    assert rt.installer.recommended()["id"] == "linux-x64-vulkan"
    failures = json.loads(rt.installer.failures_path().read_text())
    assert "linux-x64-cuda12" in failures["b12359"]
    # The next build installs; a success clears its own record only.
    monkeypatch.setenv("FAKE_LLAMA_MODE", "ready")
    assert wait_job(rt.installer.start_install(None))["state"] == "done"
    assert rt.installer.active_install()["variant"] == "linux-x64-vulkan"


@pytest.mark.skipif(os.name == "nt", reason="the fake engine is a POSIX shell script")
def test_self_test_without_gpu_device(release, monkeypatch):
    rt = release["rt"]
    monkeypatch.setenv("FAKE_LLAMA_MODE", "no_devices")
    done = wait_job(rt.installer.start_install("linux-x64-cuda12"))
    assert done["error"]["code"] == "ENGINE_SELF_TEST_FAILED"
    assert "found no CUDA device" in done["error"]["message"]
    # A CPU build needs no device.
    assert wait_job(rt.installer.start_install("linux-x64-cpu"))["state"] == "done"


def test_install_checksum_mismatch(release):
    rt, server, rel = release["rt"], release["server"], release["rel"]
    corrupt = bytearray(rel.engine_tar)
    corrupt[len(corrupt) // 2] ^= 0xFF
    server.files["/rel/engine.tar.gz"] = FileRoute(bytes(corrupt))
    done = wait_job(rt.installer.start_install("linux-x64-vulkan"))
    assert done["error"]["code"] == "CHECKSUM_MISMATCH"


def test_install_rejects_unsafe_archive(release, tmp_path):
    rt, server = release["rt"], release["server"]
    bad = _tar(tmp_path / "bad.tar.gz", [("../evil", "file", "x", 0o644)]).read_bytes()
    lock = yaml.safe_load(release["rel"].lock_path.read_text())
    lock["variants"][1]["assets"][0].update(size_bytes=len(bad), sha256=sha(bad), url=server.base_url + "/rel/bad")
    release["rel"].lock_path.write_text(yaml.safe_dump(lock))
    server.files["/rel/bad"] = FileRoute(bad)
    done = wait_job(rt.installer.start_install("linux-x64-vulkan"))
    assert done["error"]["code"] == "EXTRACT_FAILED"
    assert not (release["models_dir"] / "engine" / "evil").exists()


def test_install_unknown_variant_and_busy(release):
    from vvl.local_models import StoreError

    rt = release["rt"]
    with pytest.raises(StoreError) as excinfo:
        rt.installer.start_install("macos-arm64-metal")    # not for this computer
    assert excinfo.value.status == 400
    rt.installer._is_running = lambda: True
    with pytest.raises(StoreError) as excinfo:
        rt.installer.start_install(None)
    assert excinfo.value.status == 409


@pytest.mark.skipif(os.name == "nt", reason="the fake engine is a POSIX shell script")
def test_uninstall(release):
    from vvl.local_models import StoreError

    rt, models_dir = release["rt"], release["models_dir"]
    assert wait_job(rt.installer.start_install("linux-x64-cpu"))["state"] == "done"
    rt.installer._is_running = lambda: True
    with pytest.raises(StoreError):
        rt.installer.uninstall()
    rt.installer._is_running = lambda: False
    assert rt.installer.uninstall("linux-x64-cpu") == ["linux-x64-cpu"]
    assert rt.installer.installed_ids() == set()
    assert not (models_dir / "engine").exists()


def test_install_route_and_disk_check(make_app, release, auth, monkeypatch):
    # The workstation report (Linux), so linux-x64-cpu is a build this "computer" can install.
    report = load_report()
    app = make_app(VVL_ENGINE_LOCK_PATH=str(release["rel"].lock_path),
                   VVL_SYSTEM_REPORT=lambda refresh=False: report)
    client = app.test_client()
    monkeypatch.setattr("vvl.downloads.free_bytes", lambda path: 1000)
    response = client.post("/v1/engine/install", headers=auth, json={})
    assert response.status_code == 507
    assert response.get_json()["error_code"] == "INSUFFICIENT_DISK_SPACE"
    monkeypatch.setattr("vvl.downloads.free_bytes", lambda path: 10 ** 15)
    assert client.post("/v1/engine/install", headers=auth, json={"bogus": 1}).status_code == 400
    response = client.post("/v1/engine/install", headers=auth, json={"variant": "linux-x64-cpu"})
    assert response.status_code == 202
    job = response.get_json()["job"]
    assert job["kind"] == "engine_install" and job["target"] == "linux-x64-cpu"
    rt = app.extensions["vvl"]["runtime"]
    wait_job(rt.jobs.get(job["id"]))
    body = client.get("/v1/engine", headers=auth).get_json()
    assert body["managed"]["installed"] is True and body["managed"]["state"] == "stopped"
    assert body["managed"]["variant"]["id"] == "linux-x64-cpu"
    response = client.post("/v1/engine/uninstall", headers=auth, json={})
    assert response.status_code == 200 and response.get_json()["managed"]["installed"] is False


# ------------------------------------------------------------- small parts --

def test_real_lock_loads():
    from vvl.llama.lock import load_lock

    lock = load_lock()
    assert lock.release.tag == "b12359"
    assert lock.find("linux-x64-cuda12").download_size_bytes == 199223370 + 594371571
    assert {a.archive_type for v in lock.variants for a in v.assets} == {"tar.gz", "zip"}


def test_parse_selftest_output():
    from vvl.llama.installer import classify_incompatible, parse_devices, parse_version

    assert parse_version("version: 1.0.0-dev (build 12359, commit b1b770d90)\nbuilt with GNU") == "b12359 (b1b770d90)"
    assert parse_version("version: 6123 (abc1234)") == "b6123 (abc1234)"
    devices = parse_devices("Available devices:\n  Vulkan0: NVIDIA RTX 6000 Ada Generation (49386 MiB, 45979 MiB free)\n"
                            "  CUDA1: NVIDIA X (100 MiB, 50 MiB free)\n  Metal: Apple M2\n")
    assert devices[0] == {"name": "Vulkan0", "description": "NVIDIA RTX 6000 Ada Generation", "total_mib": 49386,
                          "free_mib": 45979}
    assert [d["name"] for d in devices] == ["Vulkan0", "CUDA1"]
    assert "glibc 2.38" in classify_incompatible("x: version `GLIBC_2.38' not found (required by y)")
    assert classify_incompatible("CUDA error: no kernel image is available for execution") is not None
    assert classify_incompatible("all good") is None


def test_job_speed_and_eta():
    from vvl.jobs import Job

    job = Job("model_download", "x", "X", [{"path": "a", "size_bytes": 1000}])
    job.set_state("running")
    job.set_file_state("a", "running", bytes_done=0)
    job.add_network_bytes(0)
    import time

    time.sleep(0.3)
    job.add_network_bytes(500)
    job.set_file_bytes("a", 500)
    snap = job.to_dict()
    assert snap["bytes_done"] == 500 and snap["bytes_total"] == 1000
    assert snap["speed_bps"] and snap["speed_bps"] > 0
    assert snap["eta_seconds"] is not None and snap["current_file"] == "a"


GLIBC_REASON = ("Needs glibc 2.38 or newer (Ubuntu 24.04 or a similarly recent Linux); "
                "this computer has glibc 2.35.")


def test_real_lock_declares_glibc_for_linux_cuda():
    from vvl.llama.lock import load_lock

    lock = load_lock()
    pinned = {v.id for v in lock.variants if v.requires.glibc_min}
    assert pinned == {"linux-x64-cuda13", "linux-x64-cuda12", "linux-arm64-cuda13"}
    assert all(v.requires.glibc_min == "2.38" for v in lock.variants if v.id in pinned)


@pytest.mark.parametrize("fixture", ["linux_workstation_2x_rtx6000ada.json", "wsl2_ubuntu_rtx3090.json"])
def test_old_glibc_with_good_driver_selects_vulkan(fixture):
    assessed, rec = _variants(load_report(REPORTS / fixture), glibc=UBUNTU_2204_GLIBC)
    assert rec["id"] == "linux-x64-vulkan"
    cuda12 = next(v for v in assessed if v["id"] == "linux-x64-cuda12")
    assert cuda12["compatible"] is False and cuda12["reason"] == GLIBC_REASON
    assert "Linux x64, NVIDIA CUDA 12: " + GLIBC_REASON in rec["reason"]


def test_old_glibc_with_cuda13_driver_selects_vulkan():
    report = _synth("linux_workstation_2x_rtx6000ada.json", **{"accelerators.cuda.driver_version": "580.82.07"})
    assessed, rec = _variants(report, glibc=UBUNTU_2204_GLIBC)
    assert rec["id"] == "linux-x64-vulkan"
    assert [v["reason"] for v in assessed[:2]] == [GLIBC_REASON, GLIBC_REASON]


def test_old_glibc_on_arm64_selects_vulkan():
    report = _synth("linux_workstation_2x_rtx6000ada.json", **{"os.arch": "aarch64",
                                                               "accelerators.cuda.driver_version": "580.95.05"})
    assessed, rec = _variants(report, glibc=UBUNTU_2204_GLIBC)
    assert rec["id"] == "linux-arm64-vulkan"
    assert assessed[0]["reason"] == GLIBC_REASON


def test_driver_problem_is_reported_before_glibc():
    # A build that already fails on the driver keeps the driver reason.
    assessed, _rec = _variants(load_report(REPORTS / "linux_workstation_2x_rtx6000ada.json"),
                               glibc=UBUNTU_2204_GLIBC)
    assert assessed[0]["reason"] == "Needs NVIDIA driver 580.0 or newer; this computer has 560.35.05."


@pytest.mark.parametrize("glibc,expected", [("2.38", "linux-x64-cuda12"), ("2.39", "linux-x64-cuda12"),
                                            ("2.40.1", "linux-x64-cuda12"), ("2.37", "linux-x64-vulkan"),
                                            ("2.35", "linux-x64-vulkan")])
def test_glibc_boundary(glibc, expected):
    assert _variants(load_report(REPORTS / "linux_workstation_2x_rtx6000ada.json"), glibc=glibc)[1]["id"] == expected


def test_unknown_glibc_does_not_block_cuda():
    assessed, rec = _variants(load_report(REPORTS / "linux_workstation_2x_rtx6000ada.json"), glibc=None)
    assert rec["id"] == "linux-x64-cuda12"
    assert assessed[1]["compatible"] is True and assessed[1]["reason"] is None


def test_glibc_does_not_affect_other_systems():
    # Windows and macOS builds have no glibc requirement, whatever value is passed.
    for fixture, expected in (("windows_laptop_rtx4060_8gb.json", "windows-x64-cuda12"),
                              ("macbook_air_m2_16gb.json", "macos-arm64-metal")):
        assert _variants(load_report(REPORTS / fixture), glibc=UBUNTU_2204_GLIBC)[1]["id"] == expected


def test_auto_glibc_uses_the_host_seam(monkeypatch):
    from vvl.llama import variants
    from vvl.llama.lock import load_lock

    report = load_report(REPORTS / "linux_workstation_2x_rtx6000ada.json")
    monkeypatch.setattr("vvl.llama.variants.host_glibc", lambda: UBUNTU_2204_GLIBC)
    assert variants.recommend(variants.assess(load_lock(), report))["id"] == "linux-x64-vulkan"
    monkeypatch.setattr("vvl.llama.variants.host_glibc", lambda: None)
    assert variants.recommend(variants.assess(load_lock(), report))["id"] == "linux-x64-cuda12"
    # A non-Linux report never consults the host's glibc.
    calls = []
    monkeypatch.setattr("vvl.llama.variants.host_glibc", lambda: calls.append(1) or UBUNTU_2204_GLIBC)
    variants.assess(load_lock(), load_report(REPORTS / "windows_laptop_rtx4060_8gb.json"))
    assert calls == []


def test_host_glibc_reads_only_on_linux(monkeypatch, pinned_host_glibc):
    real_host_glibc = pinned_host_glibc   # the unpinned seam
    monkeypatch.setattr("vvl.llama.variants.platform.system", lambda: "Darwin")
    monkeypatch.setattr("vvl.llama.variants.local_glibc", lambda: "2.31")
    assert real_host_glibc() is None
    monkeypatch.setattr("vvl.llama.variants.platform.system", lambda: "Linux")
    assert real_host_glibc() == "2.31"
