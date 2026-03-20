"""
gaio/cuda/gpu_backend.py
========================
Explicit CUDA backend using Numba ``@cuda.jit``.

Architecture
------------
Two-layer design, matching GAIO.jl's acceleration philosophy:

1. ``make_map_kernel(f_device)``
   A **kernel factory** that captures a user-supplied device function
   ``f_device`` via closure and returns a compiled ``@cuda.jit`` kernel.
   Numba compiles the kernel specialisation at first call — the closure
   technique means the device function is a compile-time constant to
   the PTX compiler, enabling full inlining and register allocation.

2. ``CUDADispatcher``
   Owns one compiled kernel, manages all host↔device memory transfers,
   and performs grid/block arithmetic.  All ``cuda.to_device`` and
   ``.copy_to_host`` calls are isolated here — the Phase 4 MPI path
   will replace these two calls with GPUDirect RDMA transfers while
   leaving the kernel call site unchanged.

Thread layout
-------------
We use a **1-D grid** of 1-D blocks.  Each thread handles exactly one
test point (one row of the ``(N, n)`` ``test_pts`` array):

    Thread global index:  idx = blockIdx.x * blockDim.x + threadIdx.x
                          (equivalently: cuda.grid(1))

    Grid dimensions:
        threads_per_block = 256          (multiple of warp size 32)
        blocks_per_grid   = ⌈N / 256⌉   (covers all N points)

    Guard:  ``if idx < N`` — handles the tail when N % 256 ≠ 0.

The 1-D layout is optimal because:
* Our work items (test points) are a 1-D list of length N = K × M.
* All threads execute an identical instruction sequence (SIMT ideal).
* The C-contiguous ``(N, n)`` layout means ``test_pts[idx]`` is a
  contiguous n-element slice — coalesced global memory access when
  threads in the same warp access consecutive rows AND n ≤ 32 bytes
  (i.e., n ≤ 4 dimensions of float64).

Requirements on ``f_device``
-----------------------------
* Must be decorated with ``@numba.cuda.jit(device=True)``.
* Must accept **two** 1-D device-array views:
    - ``x``   — input row, shape ``(n,)``  (read-only in practice)
    - ``out`` — output row, shape ``(n,)`` (write-only; pre-allocated by kernel)
* Must write the result into ``out`` in-place.  Do **not** return a value.

This "output-parameter" pattern is required by Numba CUDA because device
functions may not return heap-allocated arrays.  The kernel passes a direct
view of one row of the ``mapped`` output buffer as ``out``, so no
intermediate allocation occurs.

Example user-defined device function (2-D harmonic oscillator)
--------------------------------------------------------------
>>> from numba import cuda
>>> @cuda.jit(device=True)
... def f_device_harmonic(x, out):
...     out[0] =  x[1]
...     out[1] = -x[0]
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from numpy.typing import NDArray

from gaio.core.box import F64
from gaio.cuda.backends import (
    THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK, detect_gpu_dtype,
)


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------

def make_map_kernel(f_device):
    """
    Build a ``@cuda.jit`` kernel that applies *f_device* to every row of
    ``test_pts``.

    The kernel is compiled lazily on its first call and cached by Numba.
    Subsequent calls with the same *f_device* reuse the compiled PTX.

    Parameters
    ----------
    f_device : ``@cuda.jit(device=True)`` callable
        Device function with signature::

            f_device(x: device_array[float64, 1D])
                -> cuda.local.array[float64, 1D]

        The output local array size must match ``test_pts.shape[1]``
        at runtime (verified by the kernel via ``mapped.shape[1]``).

    Returns
    -------
    kernel : Numba CUDA JIT function
        Kernel callable as ``kernel[blocks, threads](test_pts, mapped)``.

    Notes
    -----
    The closure captures ``f_device`` by reference.  Numba's CUDA JIT
    compiler resolves ``f_device`` as a compile-time constant when it
    lowers the kernel to PTX, enabling full inlining of the device
    function body — no indirect function call overhead at runtime.
    """
    try:
        from numba import cuda
    except ImportError as exc:
        raise ImportError(
            "GPU backend requires the 'numba' package with CUDA support."
        ) from exc

    @cuda.jit
    def _map_kernel(test_pts, mapped):
        """
        CUDA kernel: apply f_device to one row of test_pts per thread.

        Grid / block layout
        -------------------
        Dimension: 1-D grid of 1-D blocks.

        ``cuda.grid(1)`` computes the absolute thread index:
            idx = blockIdx.x * blockDim.x + threadIdx.x

        Each thread with ``idx < N`` processes exactly one test point:
            input  : test_pts[idx]   — row view, shape (n,)
            output : mapped[idx]     — row view, shape (n,)

        The guard ``if idx < test_pts.shape[0]`` handles the tail block
        when N is not a multiple of ``threads_per_block``.

        Memory access pattern
        ---------------------
        Threads in a warp have consecutive ``idx`` values, so they
        access consecutive rows of ``test_pts``.  For C-contiguous
        row-major storage this gives coalesced reads when the row
        stride (n * 8 bytes) aligns to 128-byte cache lines — true
        for n ≥ 2 (≥ 16 bytes per row).  Writing to ``mapped[idx]``
        (the output row view) is likewise coalesced across the warp.

        Output-parameter pattern
        ------------------------
        ``out = mapped[idx]`` is a **direct view** into one row of the
        output buffer in global memory.  Passing it to ``f_device(x, out)``
        lets the device function write its result directly — zero
        intermediate allocation, zero extra copies.  Numba CUDA requires
        this pattern because device functions cannot return heap arrays.
        """
        # ── One thread per test point ────────────────────────────────────
        idx = cuda.grid(1)                        # absolute thread index
        if idx < test_pts.shape[0]:
            # Slice: 1-D view of the idx-th row in global memory
            x   = test_pts[idx]
            out = mapped[idx]          # direct output row view — no alloc

            # Device function writes result into out in-place.
            # PTX compiler inlines f_device here (no indirect call).
            f_device(x, out)

    return _map_kernel


# ---------------------------------------------------------------------------
# CUDADispatcher — memory management + kernel launch
# ---------------------------------------------------------------------------

class CUDADispatcher:
    """
    Manages host↔device memory transfers and launches the compiled CUDA
    kernel for the map-boxes inner loop.

    Parameters
    ----------
    f_device : ``@cuda.jit(device=True)`` callable
        Device function (see :func:`make_map_kernel`).
    threads_per_block : int, optional
        Number of CUDA threads per block.  Must be a multiple of 32
        (warp size).  Default: 256.
    dtype : np.float32 or np.float64 or None, optional
        Compute dtype for GPU arrays.  If ``None`` (default), calls
        :func:`~gaio.cuda.backends.detect_gpu_dtype` to choose
        automatically based on the device's FP64 throttle ratio:

        * Consumer GPU (ratio ≥ 32, e.g. RTX 4080 ratio=64) → ``np.float32``
        * Datacenter GPU (ratio < 32, e.g. A100 ratio=2)    → ``np.float64``

        Set explicitly to ``np.float64`` to force double precision on any GPU.

    Attributes
    ----------
    kernel : Numba CUDA JIT function
    threads_per_block : int
    dtype : numpy dtype
        Effective compute dtype used inside the GPU kernel.

    Precision note
    --------------
    ``test_pts`` (generated in float64 by the partition arithmetic) is cast
    to ``self.dtype`` before ``cuda.to_device``.  The kernel runs entirely
    in ``self.dtype``.  The result is cast back to ``float64`` on the host
    after ``copy_to_host`` so that downstream code (``point_to_key_batch``)
    always receives ``float64``.  The host-side cast is a vectorised NumPy
    operation — negligible cost compared to the GPU kernel.

    Phase 4 note
    ------------
    The two lines::

        d_test_pts = cuda.to_device(test_pts)      # host → device
        result = d_mapped.copy_to_host()           # device → host

    are the **only** host↔device transfer points in the entire library.
    In Phase 4, these will be replaced with::

        d_test_pts = cuda.to_device(local_slice)   # MPI rank's slice
        comm.Send(d_mapped, dest=root, ...)        # GPUDirect RDMA

    The kernel call site (``self.kernel[...](...)``) is unchanged.
    """

    def __init__(
        self,
        f_device,
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
        self.threads_per_block = threads_per_block
        # Resolve dtype: auto-detect if not specified
        self.dtype = detect_gpu_dtype() if dtype is None else np.dtype(dtype)
        # Compile kernel once at construction — amortises PTX compilation cost
        self.kernel = make_map_kernel(f_device)

    # ------------------------------------------------------------------
    # Grid geometry
    # ------------------------------------------------------------------

    def _grid_dims(self, N: int) -> tuple[int, int]:
        """
        Compute (blocks_per_grid, threads_per_block) for N work items.

        The formula  blocks = ⌈N / TPB⌉  ensures every work item gets
        a thread.  The guard in the kernel discards the surplus threads
        in the last block when N % threads_per_block ≠ 0.

        Parameters
        ----------
        N : int
            Total number of work items (= K * M, test points).

        Returns
        -------
        (blocks_per_grid, threads_per_block) : tuple[int, int]
        """
        tpb = self.threads_per_block
        bpg = math.ceil(N / tpb)         # ⌈N / TPB⌉ — covers all N threads
        return bpg, tpb

    # ------------------------------------------------------------------
    # Main dispatch — the only host↔device transfer site
    # ------------------------------------------------------------------

    def __call__(self, test_pts: NDArray[F64]) -> NDArray[F64]:
        """
        Apply the CUDA kernel to *test_pts* and return the mapped array.

        Steps
        -----
        1. ``cuda.to_device``        — transfer input to VRAM
        2. ``cuda.device_array``     — allocate output in VRAM
        3. ``kernel[bpg, tpb]``      — launch GPU kernel
        4. ``copy_to_host``          — retrieve result into pinned host RAM
        5. Return the host array

        Parameters
        ----------
        test_pts : ndarray, shape (N, n), float64, C-contiguous
            Test points generated by the calling BoxMap.

        Returns
        -------
        ndarray, shape (N, n), float64, C-contiguous
            Mapped points.
        """
        try:
            from numba import cuda
        except ImportError as exc:
            raise ImportError("GPU backend requires numba.") from exc

        # Cast to compute dtype (float32 on consumer GPU, float64 on datacenter).
        # Halves PCIe transfer size when dtype=float32; kernel arithmetic runs
        # at full TFLOPS instead of 1/64 speed on consumer cards.
        pts = np.ascontiguousarray(test_pts, dtype=self.dtype)
        N, n = pts.shape
        bpg, tpb = self._grid_dims(N)

        # ── Step 1: host → device ────────────────────────────────────────
        # Phase 4 replacement: cuda.to_device(mpi_local_slice)
        d_test_pts = cuda.to_device(pts)

        # ── Step 2: allocate output buffer in VRAM ───────────────────────
        d_mapped = cuda.device_array((N, n), dtype=self.dtype)

        # ── Step 3: kernel launch ────────────────────────────────────────
        # Grid: [bpg blocks] × [tpb threads/block]
        # Total threads launched: bpg * tpb ≥ N  (surplus guarded in kernel)
        self.kernel[bpg, tpb](d_test_pts, d_mapped)

        # ── Step 4: device → host, then upcast to float64 ────────────────
        # Phase 4 replacement: GPUDirect RDMA send to MPI root rank.
        # The upcast to float64 is a vectorised NumPy op on CPU — negligible
        # cost.  Downstream code (point_to_key_batch) always sees float64.
        result = d_mapped.copy_to_host()
        if result.dtype != np.float64:
            result = result.astype(np.float64)
        return result
