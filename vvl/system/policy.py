"""Every tunable number the system checker uses, in one place.

Change a value here and both the capability summary (``capability.py``) and
the model-fit verdicts (``fit.py``) follow. Memory sizes are bytes; "GB" in
user-facing text means binary gigabytes (GiB), as operating systems show it.
"""

from __future__ import annotations

from ._util import GiB, KiB, MiB

# ---------------------------------------------------------------------------
# Memory budgets: how much memory a model may use on each kind of device.
# ---------------------------------------------------------------------------

# Discrete GPU: usable = total - max(RESERVE_MIN, RESERVE_FRACTION * total).
# The reserve covers the CUDA/Vulkan context and whatever the desktop compositor
# keeps on the card. (8 GB card -> ~7.6 GB usable; 48 GB card -> ~46 GB.)
DISCRETE_RESERVE_MIN_BYTES = 384 * MiB
DISCRETE_RESERVE_FRACTION = 0.04

# Unified memory (Apple Silicon; NVIDIA GB10/Grace/Jetson class): the GPU may use
# this fraction of system RAM. These match macOS's default Metal
# ``recommendedMaxWorkingSetSize`` (about 2/3 of RAM up to 32 GB, about 3/4
# above), which llama.cpp's Metal backend respects by default. The rest is left
# for macOS and other apps. A user-set ``iogpu.wired_limit_mb`` overrides it.
UNIFIED_SMALL_RAM_LIMIT_BYTES = 32 * GiB  # at or below this, use the small fraction
UNIFIED_FRACTION_SMALL = 2 / 3
UNIFIED_FRACTION_LARGE = 0.75

# Integrated GPU on a PC (AMD APU, Intel iGPU) using Vulkan: the larger of its
# dedicated carve-out and this fraction of system RAM (Windows and the Linux
# amdgpu/i915 drivers let an iGPU map about half of RAM by default).
INTEGRATED_SHARED_FRACTION = 0.5

# System RAM a model may use when it runs on the CPU (or for layers offloaded
# from the GPU): total - max(CPU_RESERVE_MIN, CPU_RESERVE_FRACTION * total).
# The reserve is for the OS, the browser and VoucherVisionLocal itself.
CPU_RESERVE_MIN_BYTES = 3 * GiB
CPU_RESERVE_FRACTION = 0.25

# ---------------------------------------------------------------------------
# Tiers, from the GPU memory budget (usable bytes, after the reserves above).
# A machine whose recommended backend is "cpu" is always "cpu_only".
# Lower bounds, inclusive, checked from the top down.
# ---------------------------------------------------------------------------
TIER_THRESHOLDS = (
    ("workstation", 40 * GiB),  # 48 GB cards, multi-GPU, 64 GB+ Apple Silicon
    ("high", 14 * GiB),  # 16-32 GB cards, 24-36 GB Apple Silicon
    ("mid", 7 * GiB),  # 8-12 GB cards, 16 GB Apple Silicon
    ("entry", 0),  # 4-6 GB cards, 8 GB Apple Silicon, small iGPUs
)

# ---------------------------------------------------------------------------
# Model memory estimate (used by fit.estimate_required_bytes):
#
#   required = sum(file sizes: model + mmproj)
#            + context_default * KV_BYTES_PER_TOKEN
#            + RUNTIME_MARGIN_BYTES
#
# KV_BYTES_PER_TOKEN is an f16 K+V cache rounded up from Qwen3-VL-8B
# (36 layers x 8 KV heads x 128 dims x 2 (K and V) x 2 bytes = 144 KiB/token).
# It slightly overstates small/MoE models and understates 32B (256 KiB/token),
# which the margin absorbs. RUNTIME_MARGIN_BYTES covers llama.cpp compute
# buffers and the vision encoder's working memory for one image.
# ---------------------------------------------------------------------------
KV_BYTES_PER_TOKEN = 160 * KiB
RUNTIME_MARGIN_BYTES = 768 * MiB
DEFAULT_CONTEXT_TOKENS = 8192

# ---------------------------------------------------------------------------
# Warning thresholds.
# ---------------------------------------------------------------------------
LOW_RAM_BYTES = 7 * GiB  # "8 GB" machines show ~7.6-7.9 GB; warn below that
LOW_DISK_FREE_BYTES = 10 * GiB
MIN_CUDA_COMPUTE_CAPABILITY = 5.0  # llama.cpp's CUDA builds need Maxwell or newer

# Integrated GPUs that are worth using through Vulkan (shared memory, but much
# faster than the CPU). Others (older Intel UHD/HD, small Vega) are not
# recommended by default. Regular expressions, matched case-insensitively.
CAPABLE_IGPU_PATTERNS = (
    r"\bRadeon\b.*\b(?:680M|760M|780M|880M|890M|8040S|8050S|8060S)\b",
    r"\b(?:Phoenix|Hawk Point|Strix|Krackan|Rembrandt)\b",
    r"\bArc\b.*\b(?:Graphics|1[34]0[VT])\b",
    r"\bArc\(TM\)\s+(?:Graphics|1[34]0[VT])",
)
