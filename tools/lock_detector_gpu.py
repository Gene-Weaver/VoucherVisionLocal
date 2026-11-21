#!/usr/bin/env python3
"""Write vvl/registry/detector_gpu.lock.yaml from uv.lock.

The label detector's NVIDIA runtime (onnxruntime-gpu with the CUDA 12 and
cuDNN 9 wheels) is resolved and hash-pinned by uv as the `detector-gpu`
dependency group in pyproject.toml, but it is never installed into the app's
environment: the app downloads these wheels into a folder of their own
(vvl/detector_gpu.py). This tool lists, per platform, the wheels that the
group needs beyond what the app's environment already has, with each wheel's
URL, sha256 and size from uv.lock.

    uv lock && uv run python tools/lock_detector_gpu.py

tests/test_detector_gpu.py fails when the YAML and uv.lock disagree.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import yaml
from packaging.markers import Marker
from packaging.utils import parse_wheel_filename

REPO = Path(__file__).resolve().parent.parent
PROJECT = "vouchervision-local"
GROUP = "detector-gpu"
ROOT_PACKAGE = "onnxruntime-gpu"
OUTPUT = REPO / "vvl" / "registry" / "detector_gpu.lock.yaml"

# The two platforms with NVIDIA builds of onnxruntime-gpu that VVL ships for.
PLATFORMS = {
    "linux-x86_64": {"sys_platform": "linux", "platform_system": "Linux", "platform_machine": "x86_64",
                     "os_name": "posix"},
    "windows-x86_64": {"sys_platform": "win32", "platform_system": "Windows", "platform_machine": "AMD64",
                       "os_name": "nt"},
}
PYTHON = "3.12"
# CUDA 12 minor-version compatibility: the oldest driver that runs any CUDA 12.x build.
MIN_DRIVER = {"linux-x86_64": "525.60.13", "windows-x86_64": "527.41"}
MIN_COMPUTE_CAPABILITY = "6.0"   # cuDNN 9 and onnxruntime's CUDA kernels; older GPUs stay on the CPU


def _env(platform: str) -> dict:
    env = {"python_version": PYTHON, "python_full_version": f"{PYTHON}.0", "implementation_name": "cpython",
           "platform_python_implementation": "CPython", "platform_release": "", "platform_version": "",
           "implementation_version": f"{PYTHON}.0", "extra": ""}
    env.update(PLATFORMS[platform])
    return env


def _applies(dep: dict, env: dict) -> bool:
    marker = dep.get("marker")
    return True if not marker else Marker(marker).evaluate(env)


def _closure(packages: dict, start: list[dict], env: dict) -> dict[str, dict]:
    """Every package (name -> lock entry) that `start` needs on this platform."""
    found: dict[str, dict] = {}
    todo = list(start)
    while todo:
        dep = todo.pop()
        if not _applies(dep, env):
            continue
        entries = packages[dep["name"]]
        if "version" in dep:
            entries = [e for e in entries if e.get("version") == dep["version"]]
        if len(entries) != 1:
            # Several versions without a version on the edge: the edge's marker picks one.
            entries = [e for e in entries if _applies({"marker": e.get("resolution-markers", [None])[0]}, env)] \
                if all(e.get("resolution-markers") for e in entries) else entries
        if len(entries) != 1:
            raise SystemExit(f"cannot choose a version of {dep['name']} for {env['sys_platform']}")
        entry = entries[0]
        name = entry["name"]
        extras = tuple(dep.get("extra") or ())
        key = (name, extras)
        if name in found and key in found[name]["_extras_seen"]:
            continue
        found.setdefault(name, {**entry, "_extras_seen": set()})["_extras_seen"].add(key)
        todo.extend(entry.get("dependencies") or [])
        for extra in extras:
            todo.extend((entry.get("optional-dependencies") or {}).get(extra) or [])
    return found


def _wheel_for(entry: dict, platform: str) -> dict:
    """The one wheel of `entry` for cp312 on `platform` (platform-specific before py3-none-any)."""
    best = None
    for wheel in entry.get("wheels") or []:
        filename = wheel["url"].rsplit("/", 1)[-1]
        _name, _version, _build, tags = parse_wheel_filename(filename)
        rank = None
        for tag in tags:
            if tag.interpreter not in ("cp312", "py3", "py312") or tag.abi not in ("cp312", "abi3", "none"):
                continue
            plat = tag.platform
            if platform == "linux-x86_64" and plat.endswith("_x86_64") and plat.startswith(("manylinux", "linux")):
                rank = 0
            elif platform == "windows-x86_64" and plat == "win_amd64":
                rank = 0
            elif plat == "any":
                rank = 1 if rank is None else rank
        if rank is not None and (best is None or rank < best[0]):
            best = (rank, filename, wheel)
    if best is None:
        raise SystemExit(f"{entry['name']} {entry['version']} has no wheel for {platform}")
    _rank, filename, wheel = best
    algo, _, digest = wheel["hash"].partition(":")
    if algo != "sha256":
        raise SystemExit(f"{filename}: expected a sha256 hash, got {algo}")
    return {"name": entry["name"], "version": entry["version"], "filename": filename, "url": wheel["url"],
            "sha256": digest, "size_bytes": int(wheel["size"])}


def min_glibc(filenames) -> str:
    """The newest glibc any of these wheels needs (each wheel needs the oldest of its manylinux tags)."""
    legacy = {"manylinux1": (2, 5), "manylinux2010": (2, 12), "manylinux2014": (2, 17)}
    need = (2, 5)
    for filename in filenames:
        lowest = None
        for tag in parse_wheel_filename(filename)[3]:
            plat = tag.platform
            if plat.startswith("manylinux_"):
                major, minor = plat.split("_")[1:3]
                version = (int(major), int(minor))
            elif plat.rsplit("_x86_64", 1)[0] in legacy:
                version = legacy[plat.rsplit("_x86_64", 1)[0]]
            else:
                continue
            lowest = version if lowest is None else min(lowest, version)
        if lowest is not None:
            need = max(need, lowest)
    return f"{need[0]}.{need[1]}"


def build(uv_lock: Path = REPO / "uv.lock") -> dict:
    lock = tomllib.loads(uv_lock.read_text(encoding="utf-8"))
    packages: dict[str, list[dict]] = {}
    for entry in lock["package"]:
        packages.setdefault(entry["name"], []).append(entry)
    project = packages[PROJECT][0]
    group = (project.get("dev-dependencies") or {}).get(GROUP)
    if not group:
        raise SystemExit(f"uv.lock has no {GROUP!r} group; run `uv lock` after editing pyproject.toml")
    root = packages[ROOT_PACKAGE][0]
    out = {
        "onnxruntime_gpu": root["version"],
        "cuda": "12",
        "min_compute_capability": MIN_COMPUTE_CAPABILITY,
        "platforms": {},
    }
    for platform in PLATFORMS:
        env = _env(platform)
        base = _closure(packages, list(project.get("dependencies") or []), env)
        gpu = _closure(packages, list(group), env)
        wheels = []
        for name in sorted(gpu):
            if name in base:
                if base[name]["version"] != gpu[name]["version"]:
                    raise SystemExit(f"{name}: the app has {base[name]['version']}, the GPU runtime needs "
                                     f"{gpu[name]['version']}; both cannot be on one path")
                continue
            wheels.append(_wheel_for(gpu[name], platform))
        if ROOT_PACKAGE not in {w["name"] for w in wheels}:
            raise SystemExit(f"{ROOT_PACKAGE} is not in the {platform} wheel list")
        entry = {"min_driver": MIN_DRIVER[platform]}
        if platform.startswith("linux"):
            entry["min_glibc"] = min_glibc(w["filename"] for w in wheels)
        entry["wheels"] = wheels
        out["platforms"][platform] = entry
    return out


HEADER = """\
# The label detector's NVIDIA runtime: onnxruntime-gpu (CUDA 12, cuDNN 9) and the
# NVIDIA library wheels it loads, per platform. GENERATED from uv.lock by
# tools/lock_detector_gpu.py (the `detector-gpu` group in pyproject.toml); do not
# edit by hand. Only wheels the app's own environment lacks are listed: the rest
# (numpy, protobuf, ...) come from the environment. vvl/detector_gpu.py downloads
# them, checks each sha256 and installs them into <VVL_HOME>/runtime/detector-gpu/.
"""


def render(data: dict) -> str:
    return HEADER + yaml.safe_dump(data, sort_keys=False, width=200)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="Exit 1 if the YAML is out of date; write nothing.")
    args = parser.parse_args()
    text = render(build())
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != text:
            print(f"{OUTPUT.relative_to(REPO)} is out of date; run tools/lock_detector_gpu.py", file=sys.stderr)
            return 1
        return 0
    OUTPUT.write_text(text, encoding="utf-8")
    data = yaml.safe_load(text)
    for platform, entry in data["platforms"].items():
        total = sum(w["size_bytes"] for w in entry["wheels"])
        print(f"{platform}: {len(entry['wheels'])} wheels, {total / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
