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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
