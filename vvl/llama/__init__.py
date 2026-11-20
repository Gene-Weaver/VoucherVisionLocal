"""The managed llama.cpp engine (docs/API.md section 9).

    engine.lock.yaml   the pinned upstream release (per-platform builds, sha256)
    lock.py            loads and validates the lock file
    variants.py        which build fits this computer (and why the others do not)
    installer.py       download, verify, extract, self-test, uninstall
    extract.py         safe .tar.gz / .zip extraction
    supervisor.py      start/stop/switch `llama-server`, health, logs, idle stop, reaper
    procutil.py        launch so the engine dies with VVL; stop with escalation
    lifeline.py        parent-death guardian for macOS (no PR_SET_PDEATHSIG there)
    inference.py       per-model engine and sampling settings from models_local.yaml
"""
