"""Command line: `vvl serve`, `vvl doctor`, `vvl system`."""

from __future__ import annotations

import argparse
import json
import sys

import vvl


def _cmd_serve(args) -> int:
    from vvl.server import serve

    return serve(host=args.host, port=args.port, threads=args.threads)


def _cmd_system(args) -> int:
    from vvl.api.system import get_system_report

    print(json.dumps(get_system_report(refresh=args.refresh), indent=2))
    return 0


def _cmd_doctor(args) -> int:
    from vvl.doctor import run_checks

    checks = run_checks()
    width = max(len(c["name"]) for c in checks)
    print(f"{vvl.APP_NAME} {vvl.__version__} (Phase {vvl.PHASE}) readiness")
    for check in checks:
        mark = {"ok": "ok  ", "warn": "warn", "fail": "FAIL"}[check["status"]]
        print(f"  [{mark}] {check['name'].ljust(width)}  {check['detail']}")
    failed = [c for c in checks if c["status"] == "fail"]
    print("Ready." if not failed else f"Not ready: {len(failed)} check(s) failed.")
    return 0 if not failed else 1


def _cmd_detector(args) -> int:
    """Show the label detector's state; --download fetches it (in the foreground)."""
    import os
    import time

    from vvl import label_detector
    from vvl.jobs import JobManager

    if args.format:
        os.environ["VVL_DETECTOR_FORMAT"] = args.format
    store = label_detector.DetectorStore(JobManager())
    state = store.state()
    print(f"Label detector: {state['repo']} @ {state['revision'][:12]}")
    print(f"Format: {state['format']} ({state['runtime']}), {state['size_bytes'] / 1e6:.0f} MB")
    print(f"Folder: {state['path']}")
    if state["state"] != "installed" and args.download:
        job = store.install()
        while job.active:
            d = job.to_dict()
            done, total = d.get("bytes_done") or 0, d.get("bytes_total") or 0
            print(f"\r  {d['state']}: {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
            time.sleep(0.5)
        print()
        if job.state != "done":
            print(f"Download failed: {job.error}")
            return 1
        state = store.state()
    print(f"State: {state['state']}")
    if state["state"] != "installed":
        return 0 if not args.download else 1
    t0 = time.perf_counter()
    detector = label_detector.load_detector(fmt=state["format"])
    print(f"Loads: {detector.label} ({time.perf_counter() - t0:.1f} s)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="vvl", description="VoucherVisionLocal backend")
    parser.add_argument("--version", action="version", version=f"{vvl.APP_NAME} {vvl.__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Run the backend and UI server (127.0.0.1 only by default).")
    p_serve.add_argument("--port", type=int, default=None, help="Listen port (default $VVL_PORT or 8642).")
    p_serve.add_argument("--host", default=None, help="Bind address (default 127.0.0.1).")
    p_serve.add_argument("--threads", type=int, default=8, help="Worker threads (default 8).")
    p_serve.set_defaults(func=_cmd_serve)

    p_doctor = sub.add_parser("doctor", help="Print a short readiness report.")
    p_doctor.set_defaults(func=_cmd_doctor)

    p_system = sub.add_parser("system", help="Print the system report as JSON.")
    p_system.add_argument("--refresh", action="store_true", help="Re-probe instead of using the cache.")
    p_system.set_defaults(func=_cmd_system)

    p_det = sub.add_parser("detector", help="Show (or --download) the TextCollage label detector.")
    p_det.add_argument("--download", action="store_true", help="Download and verify it if it is missing.")
    p_det.add_argument("--format", choices=["coreml_fp32", "onnx"], default=None,
                       help="Override the format (default: coreml_fp32 on macOS, onnx elsewhere).")
    p_det.set_defaults(func=_cmd_detector)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
