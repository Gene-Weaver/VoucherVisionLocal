"""Server pieces: owner-PID watchdog, token redaction, CLI, enrichment client."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time

import pytest

from conftest import free_port
from vvl import server
from vvl.enrichment import vvgo_client


def test_watchdog_exits_when_owner_dies():
    owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    exited = threading.Event()
    codes = []

    def fake_exit(code):
        codes.append(code)
        exited.set()

    try:
        server.start_owner_watchdog(owner.pid, exit_func=fake_exit, interval=0.1)
        time.sleep(0.4)
        assert not exited.is_set()
        owner.kill()
        owner.wait(5)
        assert exited.wait(5)
        assert codes == [0]
    finally:
        if owner.poll() is None:
            owner.kill()


def test_watchdog_exits_at_once_for_missing_owner():
    owner = subprocess.Popen([sys.executable, "-c", "pass"])
    owner.wait(5)
    codes = []
    server.start_owner_watchdog(owner.pid, exit_func=codes.append, interval=0.1)
    assert codes == [0]


def test_owner_pid_from_env(monkeypatch):
    monkeypatch.setenv("VVL_OWNER_PID", "1234")
    assert server.owner_pid_from_env() == 1234
    for bad in ("abc", "-5", "0", ""):
        monkeypatch.setenv("VVL_OWNER_PID", bad)
        assert server.owner_pid_from_env() is None


def test_redact_filter():
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "url is %s", ("http://h/?token=SECRET123",), None)
    assert server.RedactFilter("SECRET123").filter(record)
    assert record.getMessage() == "url is http://h/?token=[redacted]"


def test_cli_system_prints_json(vvl_env):
    out = subprocess.run([sys.executable, "-m", "vvl.cli", "system"], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    report = json.loads(out.stdout)
    assert report["schema"] == 1


def test_cli_doctor(vvl_env):
    out = subprocess.run([sys.executable, "-m", "vvl.cli", "doctor"], capture_output=True, text=True, timeout=120)
    assert "readiness" in out.stdout
    assert "demo engine" in out.stdout
    assert "[FAIL] textcollage" not in out.stdout
    assert not vvl_env["VVL_PRIVATE_PATH"].exists()


# ------------------------------------------------------------- enrichment --

def test_elevation_against_fake_server(fake_server):
    fake_server.routes["/elevation"] = (200, {"lat": 1.0, "lon": 2.0, "elevation_m": 1532.4})
    assert vvgo_client.elevation_m("38.9", "-81.7", base_url=fake_server.base_url) == 1532
    sent = fake_server.requests[-1]
    assert sent["path"] == "/elevation"
    assert "lat=38.900000" in sent["query"] and "lon=-81.700000" in sent["query"]
    assert "x-api-key" not in sent["headers"]

    fake_server.routes["/elevation"] = (200, {"elevation_m": None})
    assert vvgo_client.elevation_m(1, 2, base_url=fake_server.base_url) is None
    fake_server.routes["/elevation"] = (500, {"error": "x"})
    assert vvgo_client.elevation_m(1, 2, base_url=fake_server.base_url) is None


def test_elevation_failures_are_quiet(fake_server):
    before = len(fake_server.requests)
    for lat, lon in (("", "2"), ("abc", "2"), (91, 0), (0, 181), (float("nan"), 0), (True, 1)):
        assert vvgo_client.elevation_m(lat, lon, base_url=fake_server.base_url) is None
    assert len(fake_server.requests) == before
    assert vvgo_client.elevation_m(1, 2, base_url=f"http://127.0.0.1:{free_port()}", timeout=1) is None
    assert vvgo_client.elevation_for_record({"decimalLatitude": "", "decimalLongitude": "1"}) is None
    assert vvgo_client.elevation_for_record("not a record") is None


def test_wfo_stub():
    result = vvgo_client.wfo_lookup({"scientificName": "Galax aphylla"}, "vvgo")
    assert result["available"] is False
    assert result["reason"]


def test_vvgo_base_url_from_registry():
    assert vvgo_client.vvgo_base_url().startswith("https://")


@pytest.mark.parametrize("host,expected", [("127.0.0.1", "127.0.0.1"), ("0.0.0.0", "127.0.0.1"), ("::1", "[::1]")])
def test_url_host(host, expected):
    assert server._url_host(host) == expected
