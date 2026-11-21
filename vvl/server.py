"""`vvl serve`: run app.app under waitress on 127.0.0.1.

  * The session token comes from $VVL_TOKEN or is generated here. It is
    printed once, inside the clickable URL, and never logged.
  * $VVL_OWNER_PID: when set, the server exits as soon as that process is gone
    (the desktop shell sets it so a crashed shell never strands the server).
  * The managed llama.cpp engine is stopped on every way out: Ctrl+C, SIGTERM,
    the owner watchdog, POST /v1/shutdown, and normal interpreter exit. A
    stale engine left by a crashed earlier run is reaped at startup.
"""

from __future__ import annotations

import atexit
import logging
import logging.handlers
import os
import secrets
import signal
import sys
import threading
import time

import vvl
from vvl import paths

logger = logging.getLogger("vvl.server")

DEFAULT_PORT = 8642
DEFAULT_HOST = "127.0.0.1"
WATCHDOG_INTERVAL_SECONDS = 1.0
LOOPBACK_BIND_HOSTS = {"127.0.0.1", "localhost", "::1"}


class RedactFilter(logging.Filter):
    """Defense in depth: replace the session token if it ever reaches a log line."""

    def __init__(self, secret: str):
        super().__init__()
        self._secret = secret

    def filter(self, record: logging.LogRecord) -> bool:
        if self._secret:
            try:
                message = record.getMessage()
            except Exception:
                return True
            if self._secret in message:
                record.msg = message.replace(self._secret, "[redacted]")
                record.args = ()
        return True


def configure_logging(token: str | None = None, *, log_to_file: bool = True) -> None:
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_to_file:
        try:
            log_dir = paths.ensure_dir(paths.logs_dir())
            handlers.append(logging.handlers.RotatingFileHandler(
                log_dir / "vvl.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"))
        except OSError:
            pass
    for handler in handlers:
        handler.setFormatter(formatter)
        if token:
            handler.addFilter(RedactFilter(token))
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    for noisy in ("urllib3", "requests", "httpx", "httpcore", "PIL", "pillow_heif", "werkzeug"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ------------------------------------------------------------ watchdog --

def _process_identity(pid: int):
    """(alive, create_time) for a pid; create_time guards against pid reuse."""
    import psutil

    try:
        proc = psutil.Process(pid)
        if proc.status() == psutil.STATUS_ZOMBIE:
            return False, None
        return True, proc.create_time()
    except psutil.NoSuchProcess:
        return False, None
    except psutil.AccessDenied:
        return True, None


def start_owner_watchdog(owner_pid: int, *, exit_func=os._exit, interval: float = WATCHDOG_INTERVAL_SECONDS):
    """Exit the process when `owner_pid` dies. Returns the watchdog thread."""
    alive, created = _process_identity(owner_pid)
    if not alive:
        logger.warning("Owner process %s is not running; exiting.", owner_pid)
        exit_func(0)
        return None

    def _watch():
        while True:
            time.sleep(interval)
            now_alive, now_created = _process_identity(owner_pid)
            if not now_alive or (created is not None and now_created is not None and now_created != created):
                logger.info("Owner process %s exited; shutting down.", owner_pid)
                exit_func(0)
                return

    thread = threading.Thread(target=_watch, name="vvl-owner-watchdog", daemon=True)
    thread.start()
    return thread


def owner_pid_from_env() -> int | None:
    raw = os.environ.get("VVL_OWNER_PID", "").strip()
    if not raw:
        return None
    try:
        pid = int(raw)
    except ValueError:
        logger.warning("Ignoring VVL_OWNER_PID=%r: not a process id.", raw)
        return None
    if pid <= 0:
        logger.warning("Ignoring VVL_OWNER_PID=%r: not a process id.", raw)
        return None
    return pid


# --------------------------------------------------------------- serve --

def _url_host(host: str) -> str:
    if host in ("0.0.0.0", "", "::"):
        return "127.0.0.1"
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _prepare_detector(runtime, processor) -> None:
    """Load the label detector, or start its download at first start
    (VVL_DETECTOR_AUTO_DOWNLOAD=0 turns the download off)."""
    try:
        if runtime.detector.is_installed():
            processor.warm_up()
        elif os.environ.get("VVL_DETECTOR_AUTO_DOWNLOAD", "1") != "0":
            logger.info("Downloading the label detector (first start).")
            runtime.detector.install()
    except Exception:  # noqa: BLE001
        logger.exception("Could not prepare the label detector.")


def serve(*, host: str | None = None, port: int | None = None, threads: int = 8) -> int:
    host_explicit = host is not None
    host = host or DEFAULT_HOST
    if port is None:
        try:
            port = int(os.environ.get("VVL_PORT", DEFAULT_PORT))
        except ValueError:
            print("VVL_PORT must be a number.", file=sys.stderr)
            return 2

    token = os.environ.get("VVL_TOKEN", "").strip() or secrets.token_urlsafe(32)
    configure_logging(token)

    import app as app_module  # the Flask app object lives in app.py

    application = app_module.app
    application.config["VVL_TOKEN"] = token
    if host_explicit and host not in LOOPBACK_BIND_HOSTS and host not in ("0.0.0.0", "::"):
        application.config["VVL_EXTRA_ALLOWED_HOSTS"] = [host]
    if host_explicit and host not in LOOPBACK_BIND_HOSTS:
        logger.warning("Listening on %s: other computers on the network may reach this server.", host)

    runtime = application.extensions["vvl"]["runtime"]
    try:
        runtime.startup()
    except Exception:  # noqa: BLE001 - never block the UI over the reaper
        logger.exception("Engine startup cleanup failed.")
    atexit.register(runtime.shutdown)

    def _stop_and_exit(code: int = 0) -> None:
        try:
            runtime.shutdown(timeout=15.0)
        finally:
            os._exit(code)

    def _on_sigterm(signum, _frame):
        logger.info("Received signal %s; stopping.", signum)
        threading.Thread(target=_stop_and_exit, args=(0,), name="vvl-sigterm", daemon=True).start()

    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, _on_sigterm)
    except ValueError:
        pass  # not the main thread (embedded use)

    owner_pid = owner_pid_from_env()
    if owner_pid is not None:
        start_owner_watchdog(owner_pid, exit_func=_stop_and_exit)

    processor = application.extensions["vvl"]["processor"]
    threading.Thread(target=_prepare_detector, args=(runtime, processor), name="vvl-warmup", daemon=True).start()

    try:
        from waitress import create_server

        server = create_server(application, host=host, port=port, threads=threads, ident=vvl.APP_NAME)
    except OSError as exc:
        print(f"Could not listen on {host}:{port} ({exc.strerror or exc}). Is another copy running?", file=sys.stderr)
        return 1

    url = f"http://{_url_host(host)}:{port}/?token={token}"
    # The only place the token is shown: one line on stdout for the user to click.
    print(f"VoucherVisionLocal {vvl.__version__} is running. Open: {url}", flush=True)
    logger.info("Listening on %s:%s (pid %s).", _url_host(host), port, os.getpid())
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Stopped.")
    finally:
        try:
            runtime.shutdown(timeout=15.0)
        except Exception:  # noqa: BLE001
            logger.exception("Engine shutdown failed.")
        try:
            server.close()
        except Exception:
            pass
    return 0
