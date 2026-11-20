"""Guardian for platforms without PR_SET_PDEATHSIG (macOS).

    python -I .../vvl/llama/lifeline.py <vvl_pid> -- <engine argv...>

Runs the engine as its child (same process group, same stdout), forwards
SIGTERM/SIGINT to it, and kills it as soon as VVL (`vvl_pid`, our parent) is
gone, so a crashed or force-killed VVL never leaves an engine holding
graphics memory. Exits with the engine's exit code.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

POLL_SECONDS = 0.5


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] != "--":
        print("usage: lifeline <parent_pid> -- <command...>", file=sys.stderr)
        return 2
    parent = int(argv[0])
    command = argv[2:]
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL)

    def _forward(signum, _frame):
        try:
            child.send_signal(signum)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT, _forward)
    while True:
        code = child.poll()
        if code is not None:
            return code
        if os.getppid() != parent:
            child.terminate()
            try:
                child.wait(5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            return 1
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
