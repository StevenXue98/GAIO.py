"""
gaio/cuda/cpu_backend.py
========================
Multi-threaded CPU backend using Numba ``@njit(parallel=True)``.

Corresponds to Phase 3 CPU acceleration target.

How it works
------------
``_map_loop_parallel(f_jit, test_pts, mapped)`` is a Numba-compiled
function that applies ``f_jit`` to every row of ``test_pts`` using
``numba.prange`` — a parallel range that distributes loop iterations
across all available CPU threads (controlled by the ``NUMBA_NUM_THREADS``
environment variable, defaulting to the physical core count).

Numba specialises ``_map_loop_parallel`` at first call for the concrete
type of ``f_jit``.  Subsequent calls with the same ``f_jit`` type reuse
the compiled specialisation — the JIT cost is paid only once per session.

Requirements on ``f_jit``
--------------------------
* Must be decorated with ``@numba.njit`` (or ``@numba.jit(nopython=True)``).
* Must accept a 1-D ``float64`` array of shape ``(n,)`` and return a 1-D
  ``float64`` array of shape ``(n,)`` (same n).
* Must be a pure function (no shared mutable state) — multiple threads
  call ``f_jit`` concurrently.

Example
-------
>>> import numpy as np
>>> from numba import njit
>>> from gaio.cuda.cpu_backend import map_parallel

>>> @njit
... def f_jit(x):
...     return np.array([x[1], -x[0]])

>>> test_pts = np.random.rand(1000, 2).astype(np.float64)
>>> mapped = map_parallel(f_jit, test_pts)
>>> mapped.shape
(1000, 2)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gaio.core.box import F64


# ---------------------------------------------------------------------------
# Lazy import of Numba so the module remains importable in CPU-only
# environments (the error surfaces only when map_parallel is actually called).
# ---------------------------------------------------------------------------
def _get_numba():
    try:
        import numba
        return numba
    except ImportError as exc:
        raise ImportError(
            "The CPU Numba backend requires the 'numba' package.  "
            "Install it with: conda install numba  or  pip install numba"
        ) from exc


def _build_parallel_kernel():
    """
    Build and return the @njit(parallel=True) kernel function.

    Deferred to avoid JIT compilation at import time.  Called once by
    ``map_parallel`` and cached in ``_KERNEL``.
    """
    numba = _get_numba()
    prange = numba.prange

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def _map_loop_parallel(f_jit, test_pts, mapped):
        """
        Apply f_jit to each row of test_pts in parallel.

        Parameters
        ----------
        f_jit : @njit callable
            Map function: (n,) float64 → (n,) float64.
        test_pts : ndarray, shape (N, n), float64, C-contiguous
            Input test points — one row per point.
        mapped : ndarray, shape (N, n), float64, C-contiguous
            Pre-allocated output buffer.  Written in-place.

        Notes
        -----
        ``prange`` distributes iterations across Numba thread pool
        (NUMBA_NUM_THREADS workers).  Each iteration is independent:
        thread ``t`` writes only to ``mapped[i]`` for its own ``i``
        values — no false-sharing if row size n ≥ cache-line / 8 bytes.
        Phase 3 CUDA: this function is the CPU analogue of the CUDA
        kernel in gpu_backend.py.  Both have identical semantics; only
        the execution model differs.
        """
        N = test_pts.shape[0]
        for i in prange(N):                     # ← parallel loop
            result = f_jit(test_pts[i])         # one call per test point
            for j in range(result.shape[0]):
                mapped[i, j] = result[j]

    return _map_loop_parallel


# Module-level cache — populated on first call to map_parallel
_KERNEL = None


def map_parallel(
    f_jit,
    test_pts: NDArray[F64],
) -> NDArray[F64]:
    """
    Apply a Numba-compiled map function to every row of *test_pts* using
    all available CPU threads.

    Parameters
    ----------
    f_jit : @numba.njit callable
        The map function: shape ``(n,)`` → shape ``(n,)``.
    test_pts : ndarray, shape (N, n), float64
        Test points.  Will be cast to C-contiguous float64 if necessary.

    Returns
    -------
    ndarray, shape (N, n), float64, C-contiguous
        Mapped points ``[f_jit(test_pts[i]) for i in range(N)]``.

    Raises
    ------
    ImportError
        If ``numba`` is not installed.
    """
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _build_parallel_kernel()

    pts = np.ascontiguousarray(test_pts, dtype=F64)
    N, n = pts.shape
    mapped = np.empty((N, n), dtype=F64)
    _KERNEL(f_jit, pts, mapped)
    return mapped
