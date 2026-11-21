"""Per-app runtime objects for Phase 2: jobs, model store, engine installer, supervisor.

`create_app()` builds one `Runtime` and keeps it in `app.extensions["vvl"]["runtime"]`.
Everything reads settings, the registries, and keys at the moment it needs
them, so a settings change (models_dir, engine settings) applies without a
restart. Test and scratch runs can override, through app config:

    VVL_SETTINGS_PATH       settings file
    VVL_LOCAL_MODELS_PATH   models_local.yaml
    VVL_ENGINE_LOCK_PATH    engine.lock.yaml
    VVL_HF_BASE_URL         Hugging Face base URL (default https://huggingface.co)
    VVL_SYSTEM_REPORT       callable(refresh: bool) -> report dict
    VVL_PRIVATE_PATH        credentials file (for the Hugging Face token)
    VVL_DETECTOR_LOCK_PATH  archival_detector.lock.yaml
    VVL_DETECTOR_GPU_LOCK_PATH  detector_gpu.lock.yaml
    VVL_DETECTOR_GPU_BASE_URL   tests: serve the GPU runtime wheels from here
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

from vvl import paths, settings
from vvl.jobs import JobManager
from vvl.detector_gpu import GpuRuntimeStore
from vvl.detector_gpu import load_lock as load_detector_gpu_lock
from vvl.label_detector import DetectorStore
from vvl.label_detector import load_lock as load_detector_lock
from vvl.llama.installer import EngineInstaller
from vvl.llama.lock import load_lock
from vvl.local_models import LocalModelStore
from vvl.registry.loader import load_local_models
from vvl.secrets.store import SecretStore

logger = logging.getLogger(__name__)

HF_PROVIDER_ID = "huggingface"
HF_TOKEN_FIELD = "token"


def _default_report(refresh: bool = False) -> dict | None:
    from vvl.api.system import get_system_report

    return get_system_report(refresh=refresh)


class Runtime:
    def __init__(self, config: dict | None = None):
        self.config = config if config is not None else {}
        self.jobs = JobManager()
        self.models = LocalModelStore(
            self.jobs,
            registry=self.local_registry,
            models_dir=self.models_dir,
            hf_token=self.hf_token,
            hf_base_url=self.config.get("VVL_HF_BASE_URL"),
            loaded_model=lambda: self.supervisor.loaded_models() if self.supervisor else None,
        )
        self.installer = EngineInstaller(
            self.jobs,
            lock=self.engine_lock,
            models_dir=self.models_dir,
            report=lambda: self.system_report(False),
            is_running=lambda: bool(self.supervisor and self.supervisor.is_running()),
        )
        # The TextCollage label detector's files (downloaded at first start).
        self.detector = DetectorStore(
            self.jobs,
            lock=lambda: load_detector_lock(self.config.get("VVL_DETECTOR_LOCK_PATH")),
            hf_base_url=self.config.get("VVL_HF_BASE_URL"),
        )
        # Its NVIDIA runtime (onnxruntime-gpu + CUDA/cuDNN wheels), downloaded when a usable GPU is found.
        self.detector_gpu = GpuRuntimeStore(
            self.jobs,
            lock=lambda: load_detector_gpu_lock(self.config.get("VVL_DETECTOR_GPU_LOCK_PATH")),
            report=lambda: self.system_report(False),
            base_url=self.config.get("VVL_DETECTOR_GPU_BASE_URL"),
        )
        from vvl.llama.pool import EnginePool

        # `supervisor` is the engine pool: it has the single supervisor's interface
        # and runs one or more llama-server engines behind it.
        self.supervisor = EnginePool(self)
        self._shutdown_lock = threading.Lock()
        self._shut_down = False

    # -- live configuration ------------------------------------------------

    def settings_file(self) -> Path:
        override = self.config.get("VVL_SETTINGS_PATH")
        return Path(override) if override else paths.settings_path()

    def settings(self) -> dict:
        return settings.load(self.settings_file())

    def models_dir(self) -> str:
        return self.settings()["models_dir"]

    def engine_settings(self) -> dict:
        return dict(self.settings()["engine"])

    def local_registry(self):
        return load_local_models(self.config.get("VVL_LOCAL_MODELS_PATH"))

    def engine_lock(self):
        return load_lock(self.config.get("VVL_ENGINE_LOCK_PATH"))

    def system_report(self, refresh: bool = False) -> dict | None:
        func: Callable | None = self.config.get("VVL_SYSTEM_REPORT")
        try:
            return (func or _default_report)(refresh)
        except Exception as exc:  # noqa: BLE001
            logger.info("System report unavailable (%s).", type(exc).__name__)
            return None

    def secret_store(self) -> SecretStore:
        override = self.config.get("VVL_PRIVATE_PATH")
        return SecretStore(Path(override) if override else None)

    def hf_token(self) -> str | None:
        try:
            return self.secret_store().get_value(HF_PROVIDER_ID, HF_TOKEN_FIELD)
        except Exception:  # noqa: BLE001 - an unreadable PRIVATE.yaml means "no token"
            return None

    # -- lifecycle -----------------------------------------------------------

    def startup(self) -> None:
        """Called once by `vvl serve`: clean up an engine left by a crashed VVL."""
        self.supervisor.reap_stale()
        self.supervisor.start_background()

    def shutdown(self, timeout: float = 15.0) -> None:
        """Stop the engine and the jobs. Safe to call more than once."""
        with self._shutdown_lock:
            if self._shut_down:
                return
            self._shut_down = True
        try:
            self.supervisor.shutdown(timeout=timeout)
        except Exception:  # noqa: BLE001
            logger.exception("Engine shutdown failed.")
        try:
            self.jobs.cancel_all(timeout=5.0)
        except Exception:  # noqa: BLE001
            logger.exception("Job shutdown failed.")


_live: list[Runtime] = []
_live_lock = threading.Lock()


def register(runtime: Runtime) -> None:
    with _live_lock:
        _live.append(runtime)


def shutdown_all(timeout: float = 15.0) -> None:
    with _live_lock:
        runtimes = list(_live)
    for runtime in runtimes:
        runtime.shutdown(timeout=timeout)
