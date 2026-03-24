"""
gaio/cuda/gpu_backend.py
========================
Explicit CUDA backend using Numba ``@cuda.jit``.

Architecture
------------
Two-layer design, matching GAIO.jl's acceleration philosophy:

1. ``make_map_key_kernel(f_device, ndim)``
   A **kernel factory** that captures a user-supplied device function
   ``f_device`` and the spatial dimension ``ndim`` by closure, and returns
   a compiled ``@cuda.jit`` kernel.  The kernel operates entirely in integer
   key space on the input side and output side:

     Input  : (K,)   int64 box keys  +  (M, n) unit test points (on device)
     Output : (K*M,) int64 hit keys  (-1 for out-of-domain)

   Each thread handles one (box, test-point) pair and performs three steps:
     a. Decode the input key to a world-space test point (inline key → coords)
     b. Apply ``f_device``
     c. Encode the output coordinates back to a key (inline coords → key)

   Compared to the previous coordinate-based kernel this eliminates:
     - The (K*M, n) float64 test-point upload every subdivision step
     - The (K*M, n) float64 mapped-coordinate download every step
     - The CPU ``point_to_key_batch`` call (Stage 3)
   The only transfers are (K,) int64 in and (K*M,) int64 out, matching
   GAIO.jl's ``GPUSampledBoxMap`` transfer profile.

2. ``CUDADispatcher``
   Owns one compiled kernel and the on-device unit-point array.  The
   unit-point array is uploaded once at construction and reused across all
   subdivision steps.  Per-call cost: transfer ``(K,) int64`` in,
   ``(K*M,) int64`` out, plus tiny partition-geometry arrays (~24 bytes each).

Thread layout
-------------
1-D grid of 1-D blocks; each thread handles one (box key, test-point) pair:

    total threads  N = K * M
    thread index   idx = blockIdx.x * blockDim.x + threadIdx.x
    box index      key_idx = idx // M
    test-pt index  pt_idx  = idx  % M

Requirements on ``f_device``
-----------------------------
Unchanged from the previous design:
* Must be ``@numba.cuda.jit(device=True)``.
* Signature: ``f_device(x, out)`` — reads from ``x``, writes result into
  ``out`` in-place (output-parameter pattern).
* Both ``x`` and ``out`` are 1-D ``float64`` device-array views of length n.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from gaio.core.box import F64, I64
from gaio.cuda.backends import (
    THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK, detect_gpu_dtype,
)


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------

def make_map_key_kernel(f_device, ndim: int):
    """
    Build a ``@cuda.jit`` kernel that maps box keys to hit keys.

    Each thread decodes one input box key to a world-space test point,
    applies ``f_device``, and encodes the output coordinates back to an
    integer partition key.  Out-of-domain outputs are written as ``-1``.

    Parameters
    ----------
    f_device : ``@cuda.jit(device=True)`` callable
        Device function: ``f_device(x, out) -> None``.
        Reads from 1-D float64 view ``x``, writes result into ``out``.
    ndim : int
        Spatial dimension.  Captured as a compile-time constant so that
        ``cuda.local.array(ndim, ...)`` inside the kernel is a static
        allocation (registers / L1 scratchpad, not global memory).

    Returns
    -------
    kernel : Numba CUDA JIT function
        ``kernel[blocks, threads](in_keys, unit_pts, lo, cell_radius, dims, out_keys)``
    """
    try:
        import numba
        from numba import cuda
    except ImportError as exc:
        raise ImportError(
            "GPU backend requires the 'numba' package with CUDA support."
        ) from exc

    @cuda.jit
    def _map_key_kernel(in_keys, unit_pts, lo, cell_radius, dims, out_keys):
        """
        CUDA kernel: decode key → test point → apply f_device → encode key.

        Parameters (device arrays)
        --------------------------
        in_keys     : (K,)     int64   — input box keys
        unit_pts    : (M, n)   float64 — unit test points in [-1, 1]^n
        lo          : (n,)     float64 — partition domain lower bound
        cell_radius : (n,)     float64 — half-width of each grid cell
        dims        : (n,)     int64   — grid resolution per dimension
        out_keys    : (K*M,)   int64   — output hit keys (-1 = miss)
        """
        idx = cuda.grid(1)
        M = unit_pts.shape[0]
        if idx >= in_keys.shape[0] * M:
            return

        key_idx = idx // M
        pt_idx  = idx % M

        in_key = in_keys[key_idx]

        # ── Allocate local scratch arrays (registers / L1) ───────────────
        # ndim is a Python int captured from closure → compile-time constant
        x   = cuda.local.array(ndim, numba.float64)
        out = cuda.local.array(ndim, numba.float64)

        # ── Step 1: decode in_key to world-space test point ──────────────
        # Row-major unravel: for dims = (d0, d1, ..., d_{n-1}),
        #   key = i0*d1*...*d_{n-1} + i1*d2*...*d_{n-1} + ... + i_{n-1}
        # Recover from least-significant dimension first:
        #   i_{n-1} = key % d_{n-1};  remainder = key // d_{n-1}  ; ...
        # Cell center and test point for dimension i:
        #   center[i]  = lo[i] + cell_radius[i] * (2 * mi + 1)
        #   x[i]       = center[i] + cell_radius[i] * unit_pts[pt_idx, i]
        #              = lo[i] + cell_radius[i] * (2*mi + 1 + unit_pts[pt_idx, i])
        rem = in_key
        for i in range(ndim - 1, -1, -1):
            mi  = rem % dims[i]
            rem = rem // dims[i]
            x[i] = lo[i] + cell_radius[i] * (2.0 * mi + 1.0 + unit_pts[pt_idx, i])

        # ── Step 2: apply map ────────────────────────────────────────────
        f_device(x, out)

        # ── Step 3: encode output point to partition key ─────────────────
        # Compute   fi = (out[i] - lo[i]) / (2 * cell_radius[i])
        # If fi < 0 or fi >= dims[i]: out of domain → write -1.
        # Otherwise: mi_out = floor(fi); accumulate row-major key.
        out_key   = numba.int64(0)
        in_domain = True
        for i in range(ndim):
            fi = (out[i] - lo[i]) / (2.0 * cell_radius[i])
            if fi < 0.0 or fi >= numba.float64(dims[i]):
                in_domain = False
                break
            mi_out = numba.int64(fi)
            # Safety clamp for floating-point boundary edge cases
            if mi_out >= dims[i]:
                mi_out = dims[i] - numba.int64(1)
            out_key = out_key * dims[i] + mi_out

        out_keys[idx] = out_key if in_domain else numba.int64(-1)

    return _map_key_kernel


# ---------------------------------------------------------------------------
# CUDADispatcher — unit-point cache + kernel launch
# ---------------------------------------------------------------------------

class CUDADispatcher:
    """
    Manages the on-device unit-point array and launches the compiled kernel.

    The unit-point array ``(M, n) float64`` is uploaded to VRAM once at
    construction and reused across all subdivision steps — eliminating the
    per-step test-point upload of the previous coordinate-based design.

    Per-call transfers (negligible):
        In  : ``(K,) int64``    — input box keys for this rank's shard
        Out : ``(K*M,) int64``  — hit keys (-1 for out-of-domain)
        Partition geometry (lo, cell_radius, dims): 3 × (n,) arrays, ~72 bytes

    Parameters
    ----------
    f_device : ``@cuda.jit(device=True)`` callable
    unit_points : ndarray, shape (M, n), float64
        Test points in the unit cube ``[-1, 1]^n``.  Uploaded to VRAM here.
    threads_per_block : int, optional
        Default: 256.  Must be a multiple of 32.
    dtype : np.float32, np.float64, or None
        Retained for API compatibility; the key-based kernel always runs in
        float64 (correct on datacenter GPUs with full FP64 throughput).
    """

    def __init__(
        self,
        f_device,
        unit_points: NDArray[F64],
        threads_per_block: int = THREADS_PER_BLOCK,
        dtype=None,
    ) -> None:
        if threads_per_block % 32 != 0:
            raise ValueError(
                f"threads_per_block must be a multiple of 32 (warp size); "
                f"got {threads_per_block}."
            )
        if threads_per_block > MAX_THREADS_PER_BLOCK:
            raise ValueError(
                f"threads_per_block ({threads_per_block}) exceeds hardware "
                f"maximum ({MAX_THREADS_PER_BLOCK})."
            )
        try:
            from numba import cuda
        except ImportError as exc:
            raise ImportError("GPU backend requires numba.") from exc

        self.threads_per_block = threads_per_block
        self.dtype = detect_gpu_dtype() if dtype is None else np.dtype(dtype)

        unit_pts = np.ascontiguousarray(unit_points, dtype=np.float64)
        ndim = unit_pts.shape[1]

        # Compile kernel once — amortises PTX compilation across all steps
        self.kernel = make_map_key_kernel(f_device, ndim)

        # Upload unit points once — reused every subdivision step
        self.d_unit_pts = cuda.to_device(unit_pts)

    # ------------------------------------------------------------------
    # Grid geometry (unchanged)
    # ------------------------------------------------------------------

    def _grid_dims(self, N: int) -> tuple[int, int]:
        """Return (blocks_per_grid, threads_per_block) covering N work items."""
        tpb = self.threads_per_block
        bpg = math.ceil(N / tpb)
        return bpg, tpb

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------

    def __call__(
        self,
        in_keys: NDArray[I64],
        partition,
    ) -> NDArray[I64]:
        """
        Map box keys to hit keys via the CUDA kernel.

        Parameters
        ----------
        in_keys : ndarray, shape (K,), int64
            Flat partition keys of the boxes to map (this rank's shard).
        partition : BoxPartition
            Current partition (geometry changes each subdivision step).

        Returns
        -------
        out_keys : Numba CUDA device array, shape (K*M,), int64
            Hit keys on device; -1 for out-of-domain outputs.
            Wrap with ``cp.asarray(out_keys)`` for zero-copy CuPy access,
            then filter and deduplicate on-device:
            ``cp.unique(cp_keys[cp_keys >= 0])``.
        """
        from numba import cuda

        keys = np.ascontiguousarray(in_keys, dtype=I64)
        K    = len(keys)
        M    = self.d_unit_pts.shape[0]
        N    = K * M
        bpg, tpb = self._grid_dims(N)

        # Transfer input keys to device (K × 8 bytes — small)
        d_in_keys  = cuda.to_device(keys)
        d_out_keys = cuda.device_array(N, dtype=I64)

        # Partition geometry — tiny arrays; Numba auto-transfers numpy arrays
        lo          = np.ascontiguousarray(partition.domain.lo,   dtype=np.float64)
        cell_radius = np.ascontiguousarray(partition.cell_radius, dtype=np.float64)
        dims        = np.ascontiguousarray(partition.dims,        dtype=I64)

        self.kernel[bpg, tpb](
            d_in_keys, self.d_unit_pts, lo, cell_radius, dims, d_out_keys
        )

        return d_out_keys   # Numba device array (K*M,) int64 — caller deduplicates on-device
