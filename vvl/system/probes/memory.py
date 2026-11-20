"""System memory and swap (psutil)."""

from __future__ import annotations


def probe(ctx, log):
    import psutil

    out = {"total_bytes": None, "available_bytes": None, "swap_total_bytes": None, "swap_free_bytes": None}
    try:
        vm = psutil.virtual_memory()
        out["total_bytes"] = int(vm.total)
        out["available_bytes"] = int(vm.available)
    except Exception as exc:
        log.fail(f"memory: {type(exc).__name__}: {exc}")
    try:
        sw = psutil.swap_memory()
        out["swap_total_bytes"] = int(sw.total)
        out["swap_free_bytes"] = int(sw.free)
    except Exception as exc:
        # Swap is informational; some sandboxes refuse to report it.
        log.fail(f"swap: {type(exc).__name__}: {exc}")
    return out
