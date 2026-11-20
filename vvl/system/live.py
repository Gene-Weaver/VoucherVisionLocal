"""Live resource use for the header monitor (GET /v1/metrics): processor,
memory, and each graphics card's load and memory.

Two parts:

- ``LiveSampler.sample()`` reads the hardware. It is cheap, cached for
  ``MIN_INTERVAL_S`` so several clients cannot stampede it, and never raises:
  whatever cannot be read is ``None``.
- ``build_gpus(report, sample, ...)`` is pure: it lines the readings up with the
  graphics cards of the system report (same ``index`` as ``GET /v1/system``).

Sources:

- CPU and RAM: psutil. The first ``cpu_percent`` call only starts the
  measurement, so it is reported as ``None``, never as 0.
- NVIDIA: one ``nvidia-smi --query-gpu`` call (load, memory, temperature) and
  one ``--query-compute-apps`` call (memory per process, to tell the engine's
  share from other programs').
- AMD on Linux: ``gpu_busy_percent`` and ``mem_info_vram_used/total`` in sysfs.
- Unified memory (Apple Silicon, NVIDIA GB10 class): memory figures are the
  system RAM figures; load is unknown unless the vendor tool reports it.
- Anything else: unknown (``None``). Numbers are never made up.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone

from . import _proc
from ._util import MiB, system_name, to_float, to_int
from .probes import gpu_nvidia

MIN_INTERVAL_S = 1.5
NVIDIA_TIMEOUT_S = 1.5
NVIDIA_RETRY_S = 15.0           # after a failed or timed-out call
NVIDIA_MISSING_RETRY_S = 120.0  # when nvidia-smi is not installed
BUSY_WAIT_S = 1.0               # how long a second caller waits for a sample in progress

GPU_FIELDS = ["index", "uuid", "utilization.gpu", "memory.total", "memory.used", "memory.free",
              "temperature.gpu", "name"]
APP_FIELDS = ["pid", "used_memory", "gpu_uuid"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mib(value):
    v = to_float(value)
    return None if v is None or v < 0 else int(v * MiB)


def _percent(value):
    v = to_float(value)
    return None if v is None or v < 0 else round(min(v, 100.0), 1)


# ------------------------------------------------------------ parsers ---

def parse_nvidia_gpus(text) -> list[dict]:
    """Rows of ``--query-gpu=<GPU_FIELDS> --format=csv,noheader,nounits``."""
    out = []
    for row in gpu_nvidia.parse_query_csv(text, GPU_FIELDS):
        index = to_int(row.get("index"))
        if index is None:
            continue
        out.append({
            "nvidia_index": index,
            "uuid": row.get("uuid"),
            "name": row.get("name"),
            "util_percent": _percent(row.get("utilization.gpu")),
            "vram_total_bytes": _mib(row.get("memory.total")),
            "vram_used_bytes": _mib(row.get("memory.used")),
            "vram_free_bytes": _mib(row.get("memory.free")),
            "temperature_c": to_float(row.get("temperature.gpu")),
        })
    return out


def parse_nvidia_apps(text) -> list[dict]:
    """Rows of ``--query-compute-apps=pid,used_memory,gpu_uuid``. An empty list means no process."""
    out = []
    for row in gpu_nvidia.parse_query_csv(text, APP_FIELDS):
        pid = to_int(row.get("pid"))
        if pid is None:
            continue
        out.append({"pid": pid, "used_bytes": _mib(row.get("used_memory")), "gpu_uuid": row.get("gpu_uuid")})
    return out


def _read_int(path):
    try:
        with open(path, "r", encoding="ascii", errors="replace") as fh:
            return to_int(fh.read(64).strip())
    except (OSError, ValueError):
        return None


def read_amd_sysfs(sys_root="/") -> dict:
    """{pci bus id: reading} for cards whose driver publishes load and memory in sysfs (amdgpu)."""
    base = os.path.join(sys_root, "sys/bus/pci/devices")
    out = {}
    try:
        entries = sorted(os.listdir(base))[:4096]
    except OSError:
        return out
    for addr in entries:
        folder = os.path.join(base, addr)
        busy_path = os.path.join(folder, "gpu_busy_percent")
        total_path = os.path.join(folder, "mem_info_vram_total")
        if not (os.path.exists(busy_path) or os.path.exists(total_path)):
            continue
        temperature = None
        try:
            for hw in sorted(os.listdir(os.path.join(folder, "hwmon")))[:8]:
                milli = _read_int(os.path.join(folder, "hwmon", hw, "temp1_input"))
                if milli is not None:
                    temperature = round(milli / 1000.0, 1)
                    break
        except OSError:
            pass
        out[addr.lower()] = {
            "util_percent": _percent(_read_int(busy_path)),
            "vram_total_bytes": _read_int(total_path),
            "vram_used_bytes": _read_int(os.path.join(folder, "mem_info_vram_used")),
            "temperature_c": temperature,
        }
    return out


# ------------------------------------------------------------ sampler ---

class LiveSampler:
    """Thread-safe, cached hardware readings. ``sample()`` never raises."""

    def __init__(self, *, run=None, find_tool=None, clock=time.monotonic, system=None, sys_root="/",
                 psutil_module=None, min_interval: float = MIN_INTERVAL_S):
        self._run = run or _proc.run
        self._find_tool = find_tool or _proc.find_tool
        self._clock = clock
        self._system = system or system_name()
        self._sys_root = sys_root
        self._psutil = psutil_module
        self.min_interval = float(min_interval)
        self._busy = threading.Lock()
        self._cached: dict | None = None
        self._cached_at: float | None = None
        self._cpu_started = False
        self._nvidia_retry_at = 0.0
        self._nvidia_state = "unknown"
        self._start_cpu()

    # -- psutil ----------------------------------------------------------

    def _ps(self):
        if self._psutil is None:
            import psutil

            self._psutil = psutil
        return self._psutil

    def _start_cpu(self) -> None:
        """The first cpu_percent call has nothing to compare with; make it now, not on a request."""
        try:
            self._ps().cpu_percent(interval=None)
            self._cpu_started = True
        except Exception:  # noqa: BLE001
            self._cpu_started = False

    def _cpu(self) -> dict:
        out = {"percent": None, "logical_cores": None}
        try:
            ps = self._ps()
            out["logical_cores"] = ps.cpu_count(logical=True)
            value = ps.cpu_percent(interval=None)
            if self._cpu_started:
                out["percent"] = round(float(value), 1)
            self._cpu_started = True
        except Exception:  # noqa: BLE001
            pass
        return out

    def _memory(self) -> dict:
        out = {"total_bytes": None, "used_bytes": None, "available_bytes": None, "percent": None}
        try:
            vm = self._ps().virtual_memory()
            total, available = int(vm.total), int(vm.available)
            out.update(total_bytes=total, available_bytes=available, used_bytes=max(0, total - available),
                       percent=round((total - available) * 100.0 / total, 1) if total > 0 else None)
        except Exception:  # noqa: BLE001
            pass
        return out

    # -- nvidia ------------------------------------------------------------

    def _nvidia(self) -> dict:
        out = {"state": self._nvidia_state, "gpus": [], "apps": None}
        if self._system == "Darwin":
            out["state"] = "not_applicable"
            return out
        now = self._clock()
        if now < self._nvidia_retry_at:
            return out
        exe = self._find_tool("nvidia-smi", gpu_nvidia.candidates(self._system, os.environ))
        if not exe:
            self._nvidia_state, self._nvidia_retry_at = "not_found", now + NVIDIA_MISSING_RETRY_S
            out["state"] = "not_found"
            return out
        res = self._run([exe, "--query-gpu=" + ",".join(GPU_FIELDS), "--format=csv,noheader,nounits"],
                        timeout=NVIDIA_TIMEOUT_S)
        rows = parse_nvidia_gpus(res.stdout) if res.ok else []
        if not rows:
            self._nvidia_state = "timeout" if res.timed_out else "error"
            self._nvidia_retry_at = self._clock() + NVIDIA_RETRY_S
            out["state"] = self._nvidia_state
            return out
        self._nvidia_state = out["state"] = "ok"
        out["gpus"] = rows
        apps = self._run([exe, "--query-compute-apps=" + ",".join(APP_FIELDS), "--format=csv,noheader,nounits"],
                         timeout=NVIDIA_TIMEOUT_S)
        if apps.ok:
            out["apps"] = parse_nvidia_apps(apps.stdout)
        return out

    # -- public -------------------------------------------------------------

    def _collect(self) -> dict:
        sample = {"ts": _now_iso(), "cpu": self._cpu(), "memory": self._memory(),
                  "nvidia": {"state": "unknown", "gpus": [], "apps": None}, "amd": {}}
        try:
            sample["nvidia"] = self._nvidia()
        except Exception:  # noqa: BLE001
            sample["nvidia"]["state"] = "error"
        if self._system == "Linux":
            try:
                sample["amd"] = read_amd_sysfs(self._sys_root)
            except Exception:  # noqa: BLE001
                sample["amd"] = {}
        return sample

    def _fresh(self) -> dict | None:
        if self._cached is not None and self._cached_at is not None \
                and self._clock() - self._cached_at < self.min_interval:
            return self._cached
        return None

    def sample(self) -> dict:
        """The latest readings, at most ``min_interval`` seconds old in normal use.

        While another thread is reading the hardware, a caller waits briefly
        and then takes the previous sample instead of starting a second read.
        """
        cached = self._fresh()
        if cached is not None:
            return cached
        if not self._busy.acquire(timeout=BUSY_WAIT_S):
            return self._cached or {"ts": _now_iso(), "cpu": {"percent": None, "logical_cores": None},
                                    "memory": self._memory(), "nvidia": {"state": "unknown", "gpus": [], "apps": None},
                                    "amd": {}}
        try:
            cached = self._fresh()
            if cached is not None:
                return cached
            sample = self._collect()
            self._cached, self._cached_at = sample, self._clock()
            return sample
        finally:
            self._busy.release()


_default: LiveSampler | None = None
_default_lock = threading.Lock()


def default_sampler() -> LiveSampler:
    global _default
    with _default_lock:
        if _default is None:
            _default = LiveSampler()
        return _default


# ------------------------------------------------------------ assembly ---

def _gpu_entry(index, name, vendor, kind):
    return {"index": index, "name": name, "vendor": vendor, "kind": kind, "util_percent": None,
            "vram_total_bytes": None, "vram_used_bytes": None, "vram_free_bytes": None,
            "engine_bytes": None, "other_bytes": None, "temperature_c": None, "memory_source": None}


def build_gpus(report: dict | None, sample: dict, *, engine_pids=(), engine_gpu_devices=None) -> list[dict]:
    """One entry per graphics card, indexed as in the system report.

    ``engine_pids``: process ids of the running engine (empty when it is not
    running). ``engine_gpu_devices``: the cards the engine was started on
    (``None`` = not known). ``engine_bytes`` is 0 when no engine runs, the
    measured figure when the vendor tool reports memory per process, and
    ``None`` when the engine runs but its share cannot be measured.
    """
    nvidia = sample.get("nvidia") or {}
    rows = nvidia.get("gpus") or []
    apps = nvidia.get("apps")
    amd = sample.get("amd") or {}
    memory = sample.get("memory") or {}
    pids = {int(p) for p in engine_pids if p}
    report_gpus = [g for g in (report or {}).get("gpus") or [] if isinstance(g, dict)]
    by_uuid = {r["uuid"]: r for r in rows if r.get("uuid")}
    by_nv_index = {r["nvidia_index"]: r for r in rows}

    if not report_gpus:
        # No system report yet: the vendor tool's own list, in its order.
        report_gpus = [{"index": i, "vendor": "nvidia", "name": r.get("name"), "kind": "discrete",
                        "vram_total_bytes": r.get("vram_total_bytes"),
                        "details": {"uuid": r.get("uuid"), "nvidia_index": r["nvidia_index"]}}
                       for i, r in enumerate(sorted(rows, key=lambda r: r["nvidia_index"]))]

    out = []
    for g in sorted(report_gpus, key=lambda g: g.get("index") if isinstance(g.get("index"), int) else 0):
        details = g.get("details") or {}
        index = g.get("index") if isinstance(g.get("index"), int) else len(out)
        entry = _gpu_entry(index, g.get("name"), g.get("vendor") or "unknown", g.get("kind"))
        total = g.get("vram_total_bytes") if isinstance(g.get("vram_total_bytes"), int) else None
        used = None
        engine_here = bool(pids) and (engine_gpu_devices is None or index in engine_gpu_devices)
        engine_bytes = None if engine_here else 0

        row = None
        if entry["vendor"] == "nvidia":
            row = by_uuid.get(details.get("uuid")) or by_nv_index.get(details.get("nvidia_index"))
        if row is not None:
            entry["util_percent"] = row["util_percent"]
            entry["temperature_c"] = row["temperature_c"]
            total = row["vram_total_bytes"] if row["vram_total_bytes"] is not None else total
            used = row["vram_used_bytes"]
            entry["memory_source"] = "nvidia-smi" if used is not None else None
            if pids and apps is not None:
                mine = [a for a in apps if a["pid"] in pids and a.get("gpu_uuid") == row.get("uuid")]
                if all(a["used_bytes"] is not None for a in mine):
                    engine_bytes = sum(a["used_bytes"] for a in mine)
        else:
            reading = amd.get(str(details.get("pci_bus_id") or "").lower())
            if reading:
                entry["util_percent"] = reading["util_percent"]
                entry["temperature_c"] = reading["temperature_c"]
                total = reading["vram_total_bytes"] if reading["vram_total_bytes"] is not None else total
                used = reading["vram_used_bytes"]
                entry["memory_source"] = "sysfs" if used is not None else None

        if used is None and (entry["kind"] == "unified" or entry["vendor"] == "apple"):
            # Unified memory: the card's memory is the computer's memory.
            if memory.get("total_bytes") is not None:
                total, used = memory["total_bytes"], memory.get("used_bytes")
                entry["memory_source"] = "system_ram"

        entry["vram_total_bytes"] = total
        entry["vram_used_bytes"] = used
        if total is not None and used is not None:
            entry["vram_free_bytes"] = max(0, total - used)
            if engine_bytes is not None:
                engine_bytes = min(engine_bytes, used)
                entry["other_bytes"] = used - engine_bytes
        entry["engine_bytes"] = engine_bytes
        out.append(entry)
    return out
