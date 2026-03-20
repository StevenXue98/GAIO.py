"""
gaio/cuda/backends.py
=====================
Runtime backend detection and shared constants for Phase 3 acceleration.

Backends
--------
BACKEND_PYTHON  — pure NumPy fallback (always available, zero deps)
BACKEND_CPU     — Numba @njit(parallel=True) multi-threaded CPU
BACKEND_GPU     — Numba @cuda.jit explicit CUDA kernel

Detection order for 'auto':
    1. GPU  — if CUDA is available AND f_device was provided
    2. CPU  — if Numba is available AND f_jit was provided
    3. Python — unconditional fallback

Importing this module never triggers Numba JIT compilation; detection is
deferred to runtime checks so the module can be imported in environments
without Numba or CUDA.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Backend name constants — used as string literals throughout the package
# ---------------------------------------------------------------------------
BACKEND_PYTHON: str = "python"
BACKEND_CPU: str    = "cpu"
BACKEND_GPU: str    = "gpu"

# ---------------------------------------------------------------------------
# CUDA thread-block geometry constants
# ---------------------------------------------------------------------------
# THREADS_PER_BLOCK must be a multiple of 32 (warp size).
# 256 = 8 warps — a good default for compute-bound kernels on all NVIDIA
# architectures from Kepler onward.  Users may override per AcceleratedBoxMap.
THREADS_PER_BLOCK: int = 256

# Maximum block size (hardware limit for compute capability >= 2.0)
MAX_THREADS_PER_BLOCK: int = 1024

# ---------------------------------------------------------------------------
# Hardware-aware precision detection
# ---------------------------------------------------------------------------
# NVIDIA physically throttles FP64 throughput on consumer/gaming GPUs:
#
#   Consumer (GeForce, RTX gaming):  FP64 = 1/32 to 1/64 of FP32  → ratio 32–64
#   Professional workstation:        FP64 = 1/4  to 1/16 of FP32  → ratio 4–16
#   Datacenter (A100, H100, V100):   FP64 = 1/2  to 1/1  of FP32  → ratio 1–2
#
# Queried via CUDA attribute CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO
# (Numba enum value = 87).  If ratio >= _THROTTLE_THRESHOLD we default to float32
# so the GPU uses its full TFLOPS instead of 1/64th.
#
# Users can always override per AcceleratedBoxMap(dtype=np.float64).

_THROTTLE_THRESHOLD: int = 32   # ratio ≥ this → consumer GPU → use float32


def _query_fp64_ratio(device_id: int = 0) -> int:
    """
    Query ``CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO``
    from the CUDA driver via Numba's internal driver proxy.

    Uses ``numba.cuda.cudadrv.enums.CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO``
    (value 87 in all supported CUDA versions) so the attribute ID is never
    hard-coded — it is read from Numba's own enum table.

    Returns
    -------
    int
        Hardware FP64:FP32 ratio (e.g. 64 for RTX 4080, 2 for A100).
        Returns ``_THROTTLE_THRESHOLD`` (conservative, treats as consumer GPU)
        on any failure so auto-detection degrades safely.
    """
    try:
        import ctypes
        from numba.cuda.cudadrv.driver import driver as cuda_driver
        from numba.cuda.cudadrv import enums

        attr_id = enums.CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO
        ratio = ctypes.c_int(0)
        cuda_driver.cuDeviceGetAttribute(ctypes.byref(ratio), attr_id, device_id)
        return max(1, ratio.value)          # guard against 0 or negative
    except Exception:
        return _THROTTLE_THRESHOLD          # safe fallback


def detect_gpu_dtype() -> type:
    """
    Return the recommended NumPy dtype for GPU computation on the current device.

    Decision rule
    -------------
    * Query ``CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO``.
    * ratio >= 32  →  ``np.float32``  (consumer GPU; FP64 is throttled to ≤ 1/32 speed)
    * ratio <  32  →  ``np.float64``  (datacenter GPU; FP64 runs at near-FP32 speed)

    Falls back to ``np.float32`` if the device cannot be queried.

    Returns
    -------
    np.float32 or np.float64
    """
    if not _cuda_available():
        return np.float64               # no GPU → float64 for CPU paths
    ratio = _query_fp64_ratio()
    return np.float32 if ratio >= _THROTTLE_THRESHOLD else np.float64


# ---------------------------------------------------------------------------
# Runtime detection helpers
# ---------------------------------------------------------------------------

def _numba_available() -> bool:
    """Return True if the ``numba`` package is importable."""
    try:
        import numba  # noqa: F401
        return True
    except ImportError:
        return False


def _cuda_available() -> bool:
    """
    Return True if Numba CUDA is importable **and** at least one CUDA
    device is present and accessible.

    Calling ``numba.cuda.is_available()`` is safe even in CPU-only
    environments — it returns False rather than raising.
    """
    try:
        from numba import cuda
        return bool(cuda.is_available())
    except (ImportError, Exception):
        return False


def resolve_backend(
    requested: str,
    f_jit,
    f_device,
) -> str:
    """
    Resolve the effective backend string from a user request.

    Parameters
    ----------
    requested : str
        One of ``'auto'``, ``'cpu'``, ``'gpu'``, ``'python'``.
    f_jit : callable or None
        A ``@numba.njit`` compiled function (required for CPU backend).
    f_device : callable or None
        A ``@numba.cuda.jit(device=True)`` compiled function (required for
        GPU backend).

    Returns
    -------
    str
        Resolved backend: ``'gpu'``, ``'cpu'``, or ``'python'``.

    Raises
    ------
    ValueError
        If the explicitly requested backend cannot be satisfied.
    """
    if requested == "auto":
        if f_device is not None and _cuda_available():
            return BACKEND_GPU
        if f_jit is not None and _numba_available():
            return BACKEND_CPU
        return BACKEND_PYTHON

    if requested == BACKEND_GPU:
        if f_device is None:
            raise ValueError(
                "backend='gpu' requires a @cuda.jit(device=True) function "
                "passed as f_device."
            )
        if not _cuda_available():
            raise RuntimeError(
                "backend='gpu' requested but no CUDA device was found.  "
                "Check that the CUDA toolkit is installed and a compatible "
                "GPU is available."
            )
        return BACKEND_GPU

    if requested == BACKEND_CPU:
        if f_jit is None:
            raise ValueError(
                "backend='cpu' requires a @numba.njit function passed as f_jit."
            )
        if not _numba_available():
            raise RuntimeError(
                "backend='cpu' requested but the 'numba' package is not installed."
            )
        return BACKEND_CPU

    if requested == BACKEND_PYTHON:
        return BACKEND_PYTHON

    raise ValueError(
        f"Unknown backend '{requested}'.  "
        f"Choose from 'auto', 'gpu', 'cpu', 'python'."
    )
