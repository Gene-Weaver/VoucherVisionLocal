"""The label detector on an NVIDIA GPU: onnxruntime-gpu in a folder of its own.

On Linux (x86_64) and Windows (x64) with a usable NVIDIA GPU, `vvl serve`
downloads the NVIDIA runtime at first start (job kind `detector_gpu_download`):
onnxruntime-gpu built for CUDA 12 and the NVIDIA wheels it loads (cuBLAS, cuDNN
9, ...), pinned with their sha256 in `vvl/registry/detector_gpu.lock.yaml`
(generated from uv.lock by tools/lock_detector_gpu.py). The wheels are
installed with uv into `<VVL_HOME>/runtime/detector-gpu/<version>-<platform>/`,
never into the app's environment: onnxruntime-gpu and the app's onnxruntime
(which pymupdf needs) are both the `onnxruntime` package.

The detector then runs in a worker process (textcollage/cuda_worker.py) that has
this folder first on its path. Until the runtime is installed, and whenever the
GPU fails, the detector runs on the CPU as before. macOS uses Core ML instead.

Only the NVIDIA driver is needed from the system. `VVL_DETECTOR_GPU=0` turns
all of this off.
"""

from __future__ import annotations

import json
import logging
import os
import platform as _platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import yaml

from vvl import downloads, paths
from vvl.jobs import Job, JobError, JobManager

logger = logging.getLogger(__name__)

JOB_KIND = "detector_gpu_download"
JOB_LABEL = "GPU runtime"
MARKER = "installed.json"
# Unpacked, the wheels take about 1.6x their download size; they are deleted after the install.
INSTALLED_FACTOR = 1.7
DISK_MARGIN_BYTES = 1024 ** 3
INSTALL_TIMEOUT_SECONDS = 1800


def lock_path() -> Path:
    return paths.registry_dir() / "detector_gpu.lock.yaml"


def load_lock(path: str | os.PathLike | None = None) -> dict:
    with open(path or lock_path(), encoding="utf-8") as handle:
        lock = yaml.safe_load(handle)
    for key in ("onnxruntime_gpu", "cuda", "platforms"):
        if key not in lock:
            raise ValueError(f"detector_gpu.lock.yaml has no {key!r}")
    return lock


def disabled() -> bool:
    return os.environ.get("VVL_DETECTOR_GPU", "").strip() == "0"


def host_platform() -> str | None:
    """'linux-x86_64' | 'windows-x86_64' for this process, else None (no NVIDIA build here)."""
    machine = _platform.machine().lower()
    if sys.platform.startswith("linux") and machine in ("x86_64", "amd64"):
        return "linux-x86_64"
    if sys.platform == "win32" and machine in ("amd64", "x86_64"):
        return "windows-x86_64"
    return None


def host_glibc() -> str | None:
    from vvl.llama import variants

    return variants.host_glibc()


def _at_least(have, need) -> bool | None:
    from vvl.llama.variants import driver_at_least

    return driver_at_least(have, need)


def nvidia_gpus(report: dict | None) -> list[dict]:
    return [g for g in (report or {}).get("gpus") or [] if isinstance(g, dict) and g.get("vendor") == "nvidia"]


def eligibility(lock: dict, report: dict | None, *, platform: str | None = None,
                glibc: str | None = None) -> dict:
    """Can this computer run the detector on an NVIDIA GPU?

    -> {"usable": bool, "reason": code | None, "message": str | None, "device": {...} | None}
    `device` is the usable GPU with the most free memory (its nvidia-smi index).
    """
    def no(reason, message):
        return {"usable": False, "reason": reason, "message": message, "device": None}

    if platform is None:
        platform = host_platform()
    entry = (lock.get("platforms") or {}).get(platform) if platform else None
    if entry is None:
        return no("not_supported", "There is no NVIDIA runtime for the label detector on this system.")
    gpus = nvidia_gpus(report)
    if not gpus:
        return no("no_nvidia_gpu", "No NVIDIA GPU was found.")
    if platform.startswith("linux") and entry.get("min_glibc") and glibc is not None:
        if _at_least(glibc, entry["min_glibc"]) is False:
            return no("glibc_too_old", f"This Linux is too old for the GPU runtime (glibc {glibc}; "
                                       f"{entry['min_glibc']} or newer is needed).")
    driver_problem = None
    usable = []
    for gpu in gpus:
        driver = gpu.get("driver_version")
        if _at_least(driver, entry.get("min_driver")) is False:
            driver_problem = (f"The NVIDIA driver is too old for the GPU runtime ({driver}; "
                              f"{entry['min_driver']} or newer is needed).")
            continue
        capability = (gpu.get("details") or {}).get("compute_capability")
        if capability and _at_least(str(capability), str(lock.get("min_compute_capability") or "0")) is False:
            continue
        usable.append(gpu)
    if not usable:
        if driver_problem:
            return no("driver_too_old", driver_problem)
        return no("gpu_too_old", "The NVIDIA GPU is too old for the GPU runtime.")
    best = max(usable, key=lambda g: (g.get("vram_free_bytes") or g.get("vram_total_bytes") or 0))
    index = (best.get("details") or {}).get("nvidia_index", best.get("index", 0))
    device = {"index": int(index or 0), "name": best.get("name") or "NVIDIA GPU",
              "vram_free_bytes": best.get("vram_free_bytes"), "vram_total_bytes": best.get("vram_total_bytes")}
    return {"usable": True, "reason": None, "message": None, "device": device}


def wheels_for(lock: dict, platform: str) -> list[dict]:
    try:
        return list(lock["platforms"][platform]["wheels"])
    except KeyError:
        raise ValueError(f"the GPU runtime lock has no {platform!r} platform") from None


def size_bytes(lock: dict, platform: str) -> int:
    return sum(int(w["size_bytes"]) for w in wheels_for(lock, platform))


def runtime_root() -> Path:
    return paths.home() / "runtime" / "detector-gpu"


def install_dir(lock: dict, platform: str) -> Path:
    return runtime_root() / f"onnxruntime-gpu-{lock['onnxruntime_gpu']}-{platform}"


def _marker_for(lock: dict, platform: str) -> dict:
    return {"onnxruntime_gpu": lock["onnxruntime_gpu"], "platform": platform,
            "wheels": sorted(w["sha256"] for w in wheels_for(lock, platform))}


def is_installed(lock: dict, platform: str | None) -> bool:
    if not platform:
        return False
    root = install_dir(lock, platform)
    try:
        marker = json.loads((root / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return marker == _marker_for(lock, platform) and (root / "onnxruntime" / "__init__.py").is_file()


def uv_binary() -> str | None:
    """The uv that installs the wheels: the app's bundled one (VVL_UV), else uv on PATH."""
    bundled = os.environ.get("VVL_UV", "").strip()
    if bundled and Path(bundled).is_file():
        return bundled
    return shutil.which("uv")


def worker_env(runtime_dir: Path, base: dict | None = None) -> dict:
    """Environment for textcollage/cuda_worker.py: the GPU runtime first on the path, then the
    backend's own folder (for textcollage), then whatever PYTHONPATH already had."""
    env = dict(os.environ if base is None else base)
    backend = str(Path(__file__).resolve().parent.parent)
    parts = [str(runtime_dir), backend] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    seen, ordered = set(), []
    for part in parts:
        if part not in seen:
            seen.add(part)
            ordered.append(part)
    env["PYTHONPATH"] = os.pathsep.join(ordered)
    env["PYTHONNOUSERSITE"] = "1"
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"     # CUDA device ids then match nvidia-smi's indexes
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _system_report() -> dict | None:
    try:
        from vvl.api.system import get_system_report

        return get_system_report(refresh=False)
    except Exception:  # noqa: BLE001
        return None


def worker_config(lock: dict | None = None, report: dict | None = None) -> dict | None:
    """How the detector reaches the GPU (runtime folder, device, environment), or None when it
    should run on the CPU: turned off, not installed, or no usable GPU."""
    if disabled():
        return None
    platform = host_platform()
    try:
        lock = lock or load_lock()
    except (OSError, ValueError):
        return None
    if not is_installed(lock, platform):
        return None
    report = report if report is not None else _system_report()
    elig = eligibility(lock, report, platform=platform, glibc=host_glibc())
    if not elig["usable"]:
        return None
    runtime_dir = install_dir(lock, platform)
    return {"runtime_dir": str(runtime_dir), "device_index": elig["device"]["index"],
            "device_name": elig["device"]["name"], "env": worker_env(runtime_dir), "python": sys.executable}


class GpuRuntimeStore:
    """Install state and the download job, for the API, the Setup page and `vvl serve`."""

    def __init__(self, jobs: JobManager, *, lock: Callable[[], dict] = load_lock,
                 report: Callable[[], dict | None] = _system_report, base_url: str | None = None,
                 on_installed: Callable[[], None] | None = None):
        self.jobs = jobs
        self._lock_loader = lock
        self._report = report
        self.base_url = base_url          # tests: serve the wheels from here instead of their URLs
        self.on_installed = on_installed
        self._submit_lock = threading.Lock()

    def lock(self) -> dict:
        return self._lock_loader()

    def platform(self) -> str | None:
        return host_platform()

    def eligibility(self) -> dict:
        return eligibility(self.lock(), self._report(), platform=self.platform(), glibc=host_glibc())

    def is_installed(self) -> bool:
        return is_installed(self.lock(), self.platform())

    def state(self) -> dict:
        lock, platform = self.lock(), self.platform()
        base = {"onnxruntime_gpu": lock["onnxruntime_gpu"], "cuda": lock["cuda"], "device": None,
                "size_bytes": None, "path": None, "job": None, "error": None, "reason": None, "message": None}
        if disabled():
            return {**base, "state": "disabled", "reason": "disabled", "message": "Turned off (VVL_DETECTOR_GPU=0)."}
        elig = self.eligibility()
        if not elig["usable"]:
            return {**base, "state": "not_needed", "reason": elig["reason"], "message": elig["message"]}
        job = self.jobs.latest_for(JOB_KIND, platform)
        installed = is_installed(lock, platform)
        if installed:
            state = "installed"
        elif job is not None and job.active:
            state = "downloading"
        elif job is not None and job.state == "error":
            state = "error"
        else:
            state = "not_installed"
        return {**base, "state": state, "device": elig["device"], "size_bytes": size_bytes(lock, platform),
                "path": str(install_dir(lock, platform)), "job": job.to_dict() if job is not None else None,
                "error": job.error if job is not None and job.state == "error" and not installed else None}

    def should_auto_install(self) -> bool:
        """At first start: a usable GPU, not installed, and no download running or failed already."""
        if disabled():
            return False
        if not self.eligibility()["usable"] or self.is_installed():
            return False
        job = self.jobs.latest_for(JOB_KIND, self.platform())
        return job is None or job.state not in ("queued", "running", "verifying", "error")

    def maybe_auto_install(self) -> Job | None:
        try:
            if self.should_auto_install():
                logger.info("Downloading the label detector's GPU runtime (NVIDIA GPU found).")
                return self.install()
        except Exception:  # noqa: BLE001
            logger.exception("Could not start the GPU runtime download.")
        return None

    def install(self) -> Job:
        """Start the download (or return the one already running)."""
        with self._submit_lock:
            lock, platform = self.lock(), self.platform()
            if not platform or platform not in (lock.get("platforms") or {}):
                raise JobError("CONFLICT", "There is no NVIDIA runtime for the label detector on this system.")
            active = self.jobs.active_job(JOB_KIND)
            if active is not None:
                return active
            files = [{"path": w["filename"], "size_bytes": int(w["size_bytes"])} for w in wheels_for(lock, platform)]
            return self.jobs.submit(JOB_KIND, platform, JOB_LABEL, files, self._runner(lock, platform),
                                    on_finish=self._finished)

    def _finished(self, job: Job) -> None:
        if job.state == "done" and self.on_installed is not None:
            try:
                self.on_installed()
            except Exception:  # noqa: BLE001
                logger.exception("Loading the label detector on the GPU failed.")

    def _url(self, wheel: dict) -> str:
        if not self.base_url:
            return wheel["url"]
        return f"{self.base_url.rstrip('/')}/{wheel['filename']}"

    def _runner(self, lock: dict, platform: str):
        def run(job: Job):
            uv = uv_binary()
            if uv is None:
                raise JobError("INTERNAL_ERROR", "uv was not found, so the GPU runtime cannot be installed.")
            root = runtime_root()
            wheel_dir = root / "downloads"
            wheel_dir.mkdir(parents=True, exist_ok=True)
            wheels = wheels_for(lock, platform)
            total = sum(int(w["size_bytes"]) for w in wheels)
            remaining = sum(int(w["size_bytes"]) for w in wheels if not (wheel_dir / w["filename"]).is_file())
            need = remaining + int(total * INSTALLED_FACTOR) + DISK_MARGIN_BYTES
            free = downloads.free_bytes(root)
            if free is not None and free < need:
                raise JobError("INSUFFICIENT_DISK_SPACE", "Not enough free disk space for the GPU runtime.",
                               {"required_bytes": need, "free_bytes": free})
            for wheel in wheels:
                job.check_canceled()
                name, size, sha = wheel["filename"], int(wheel["size_bytes"]), wheel["sha256"]
                dest = wheel_dir / name
                if dest.is_file() and dest.stat().st_size == size:
                    job.set_file_state(name, "verifying", bytes_done=0)
                    digest = downloads.sha256_of_file(dest, cancel=job.cancel_event,
                                                      progress=lambda n, p=name: job.set_file_bytes(p, n))
                    if digest == sha:
                        job.set_file_state(name, "skipped", bytes_done=size)
                        continue
                    logger.warning("%s does not match its checksum; downloading it again.", name)
                    dest.unlink()
                elif dest.is_file():
                    dest.unlink()
                job.set_file_state(name, "running", bytes_done=0)

                def progress(done, net, p=name):
                    job.set_file_bytes(p, done)
                    if net:
                        job.add_network_bytes(net)

                downloads.download_file(self._url(wheel), dest, size_bytes=size, sha256=sha,
                                        cancel=job.cancel_event, progress=progress)
                job.set_file_state(name, "done", bytes_done=size)
            job.set_state("verifying")
            final = install_dir(lock, platform)
            staging = final.with_name(final.name + ".partial")
            shutil.rmtree(staging, ignore_errors=True)
            _uv_install(uv, staging, [wheel_dir / w["filename"] for w in wheels], root / "uv-cache", job)
            marker = _marker_for(lock, platform)
            (staging / MARKER).write_text(json.dumps(marker, indent=1) + "\n", encoding="utf-8")
            shutil.rmtree(final, ignore_errors=True)
            os.replace(staging, final)
            if not is_installed(lock, platform):
                raise JobError("EXTRACT_FAILED", "The GPU runtime could not be verified after its install.")
            # The wheels and uv's unpacked copies are no longer needed; older runtimes neither.
            shutil.rmtree(wheel_dir, ignore_errors=True)
            shutil.rmtree(root / "uv-cache", ignore_errors=True)
            for other in root.glob("onnxruntime-gpu-*"):
                if other != final and other.is_dir():
                    shutil.rmtree(other, ignore_errors=True)
            logger.info("GPU runtime (onnxruntime-gpu %s, %s) installed.", lock["onnxruntime_gpu"], platform)
            return {"path": str(final)}

        return run


def _uv_install(uv: str, target: Path, wheel_files: list[Path], cache: Path, job: Job) -> None:
    """`uv pip install --target` the downloaded wheels: no index, no dependencies, this Python."""
    cmd = [uv, "pip", "install", "--target", str(target), "--no-deps", "--no-index", "--offline",
           "--python", sys.executable, "--quiet", *[str(p) for p in wheel_files]]
    env = {k: v for k, v in os.environ.items() if not k.startswith("UV_") and k not in ("VIRTUAL_ENV", "CONDA_PREFIX")}
    env.update({"UV_CACHE_DIR": str(cache), "UV_NO_CONFIG": "1", "UV_NO_PROGRESS": "1",
                "UV_PYTHON_DOWNLOADS": "never"})
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, **kwargs)
    output: list[str] = []
    reader = threading.Thread(target=lambda: output.extend(
        line.decode("utf-8", "replace").rstrip() for line in proc.stdout), daemon=True)
    reader.start()
    deadline = time.monotonic() + INSTALL_TIMEOUT_SECONDS
    try:
        while proc.poll() is None:
            if job.cancel_event.is_set() or time.monotonic() > deadline:
                proc.kill()
                proc.wait()
                job.check_canceled()
                raise JobError("EXTRACT_FAILED", "Installing the GPU runtime took too long.")
            time.sleep(0.2)
    finally:
        reader.join(timeout=5)
    if proc.returncode != 0:
        tail = "\n".join(output[-12:])
        logger.error("uv pip install failed (%s):\n%s", proc.returncode, tail)
        raise JobError("EXTRACT_FAILED", "Installing the GPU runtime failed.", {"output": tail[-2000:]})
