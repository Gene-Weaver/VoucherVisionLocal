"""Will a model build run on this computer? ``assess(report, build)``.

Memory estimate (constants in ``policy.py``)::

    required = sum(file sizes: model + mmproj)
             + context_default * KV_BYTES_PER_TOKEN      # f16 KV cache
             + RUNTIME_MARGIN_BYTES                       # compute buffers, vision encoder

Verdicts compare ``required`` with the budgets from ``capability.plan``:

- ``fits_gpu``   fits on one GPU, or in the GPU share of unified/shared memory
- ``fits_split`` needs more than one GPU, or GPU + system RAM offload (slower)
- ``cpu_only``   no usable GPU, but fits in system RAM (slow)
- ``too_large``  does not fit even with offload
- ``unknown``    not enough information (missing sizes, failed probes)

Budgets are capacity (total minus a reserve), not "free right now", so the
verdict does not flip when a browser or another job briefly uses memory. When
current free memory is below the requirement, the detail says so.
"""

from __future__ import annotations

from . import capability, policy
from ._util import fmt_gb

VERDICTS = ("fits_gpu", "fits_split", "cpu_only", "too_large", "unknown")


def estimate_required_bytes(build):
    """Bytes needed to run ``build`` (a models_local.yaml builds[] entry), or None."""
    if not isinstance(build, dict):
        return None
    total = None
    files = build.get("files")
    if isinstance(files, list) and files:
        sizes = [f.get("size_bytes") if isinstance(f, dict) else None for f in files]
        if all(isinstance(s, int) and not isinstance(s, bool) and s > 0 for s in sizes):
            total = sum(sizes)
    if total is None:
        t = build.get("total_size_bytes")
        if isinstance(t, int) and not isinstance(t, bool) and t > 0:
            total = t
    if total is None:
        return None
    ctx = build.get("context_default")
    if not isinstance(ctx, int) or isinstance(ctx, bool) or ctx <= 0:
        ctx = policy.DEFAULT_CONTEXT_TOKENS
    return int(total + ctx * policy.KV_BYTES_PER_TOKEN + policy.RUNTIME_MARGIN_BYTES)


def _result(verdict, label, required, detail):
    return {"verdict": verdict, "label": label, "required_bytes": required, "detail": detail}


def _unknown(required, detail):
    return _result("unknown", "Unknown", required, detail)


def _distinct(a, b):
    """Format two sizes so they do not read the same ("10.8 GB" vs "10.7 GB", not "11 GB" vs "11 GB")."""
    ta, tb = fmt_gb(a), fmt_gb(b)
    if ta == tb:
        ta, tb = fmt_gb(a, decimals_below=1000), fmt_gb(b, decimals_below=1000)
    return ta, tb


def _free_note(free, required):
    if isinstance(free, int) and isinstance(required, int) and free < required:
        return f", but only {fmt_gb(free)} is free right now because other programs are using it"
    return ""


def _assess(report, build):
    required = estimate_required_bytes(build)
    if required is None:
        return _unknown(None, "This model's file sizes are not listed, so it could not be checked.")
    if not isinstance(report, dict) or not report:
        return _unknown(required, "No system report is available, so this model could not be checked.")
    need = f"Needs about {fmt_gb(required)}"
    p = capability.plan(report)
    cpu_b = p.cpu_budget

    if p.mode == "discrete":
        if not p.slots:
            return _unknown(required, f"{need}, but the GPU's memory size could not be read.")
        slots = sorted(p.slots, key=lambda s: s.usable, reverse=True)
        best = slots[0]
        if best.usable >= required:
            where = f"GPU {best.index}" if len(p.gpus) > 1 else "the GPU"
            return _result("fits_gpu", "Fits on GPU", required,
                           f"{need}; {where} has about {fmt_gb(best.usable)} of usable memory"
                           f"{_free_note(best.free, required)}.")
        gpu_sum = sum(s.usable for s in slots)
        if len(slots) > 1 and gpu_sum >= required:
            return _result("fits_split", "Split across GPUs", required,
                           f"{need}, more than any one GPU has; it can be split across your {len(slots)} GPUs "
                           f"({fmt_gb(gpu_sum)} together).")
        if isinstance(cpu_b, int) and gpu_sum + cpu_b >= required:
            return _result("fits_split", "Partly on GPU", required,
                           f"{need}; about {fmt_gb(gpu_sum)} fits on the GPU and the rest runs from system "
                           f"memory, which is much slower.")
        if cpu_b is None:
            return _unknown(required, f"{need}, more than the GPU holds, and system memory could not be read.")
        return _result("too_large", "Too large", required,
                       f"{need}, more than this computer's GPU and system memory can hold together "
                       f"(about {fmt_gb(gpu_sum + cpu_b)}).")

    if p.mode == "shared":
        gpu_b = p.gpu_budget
        if not isinstance(gpu_b, int):
            return _unknown(required, f"{need}, but this computer's memory size could not be read.")
        if gpu_b >= required:
            of_total = f" of the {fmt_gb(p.ram_total)} of memory" if isinstance(p.ram_total, int) else ""
            return _result("fits_gpu", "Fits on GPU", required,
                           f"{need}; {p.device_label} can use about {fmt_gb(gpu_b)}{of_total}"
                           f"{_free_note(p.ram_available, required)}.")
        if isinstance(cpu_b, int) and cpu_b >= required:
            need_txt, gpu_txt = _distinct(required, gpu_b)
            return _result("fits_split", "Partly on GPU", required,
                           f"Needs about {need_txt}, more than the {gpu_txt} {p.device_label} can use; part of it "
                           f"runs on the processor, which is slower.")
        limit = max(v for v in (gpu_b, cpu_b) if isinstance(v, int))
        return _result("too_large", "Too large", required,
                       f"{need}, more than this computer can hold (about {fmt_gb(limit)} usable).")

    # Processor only.
    if not isinstance(cpu_b, int):
        return _unknown(required, f"{need}, but this computer's memory size could not be read.")
    if cpu_b >= required:
        low = isinstance(p.ram_available, int) and p.ram_available < required
        note = f", and only {fmt_gb(p.ram_available)} of memory is free right now" if low else ""
        return _result("cpu_only", "CPU only (slow)", required,
                       f"{need}; no usable GPU was found, so it runs on the processor, which is slow{note}.")
    return _result("too_large", "Too large", required,
                   f"{need}, more than the about {fmt_gb(cpu_b)} of memory this computer can spare.")


def assess(report, build):
    """Fit verdict for one model build. Never raises."""
    try:
        return _assess(report, build)
    except Exception as exc:  # noqa: BLE001 - must never break /v1/models
        return _unknown(None, f"This model could not be checked ({type(exc).__name__}).")
