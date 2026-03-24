"""
gaio/cuda/accelerated_map.py
============================
AcceleratedBoxMap — a drop-in replacement for SampledBoxMap with
CPU and GPU acceleration backends.

Public interface
----------------
``AcceleratedBoxMap`` has the same public surface as ``SampledBoxMap``:
    • ``map_boxes(source: BoxSet) -> BoxSet``
    • ``__call__(source: BoxSet) -> BoxSet``
    • ``n_test_points : int``
    • ``ndim : int``

The only difference is the constructor, which accepts two optional
Numba-compiled counterparts of the Python map ``f``:

    ``f_jit``    — ``@numba.njit`` version for the CPU parallel backend
    ``f_device`` — ``@numba.cuda.jit(device=True)`` version for GPU

Backend selection (``backend`` parameter)
------------------------------------------
    'auto'   → GPU  if CUDA device present AND f_device provided
             → CPU  if Numba installed    AND f_jit provided
             → Python fallback otherwise
    'gpu'    → Force GPU (raises if CUDA unavailable or f_device missing)
    'cpu'    → Force CPU (raises if Numba unavailable or f_jit missing)
    'python' → Always use the pure-Python NumPy fallback

Correspondence with GAIO.jl
----------------------------
In GAIO.jl all map evaluation is compiled at first call by Julia's JIT.
``AcceleratedBoxMap`` exposes explicit Numba-compiled functions to achieve
the same effect in Python, at the cost of requiring the user to write
``f_jit`` / ``f_device`` separately.  This is the only unavoidable
difference between the two implementations.

Phase 3 CUDA target (§4 of PHASE2_ARCHITECTURE_NOTES.md)
----------------------------------------------------------
The three-stage pipeline inherited from SampledBoxMap:

    Stage 1 — test-point generation   (vectorised NumPy, unchanged)
    Stage 2 — map application         ← THIS FILE accelerates this stage
    Stage 3 — partition key lookup    (vectorised NumPy, unchanged)

Example
-------
>>> import numpy as np
>>> from numba import njit
>>> from gaio.core.box import Box
>>> from gaio.core.partition import BoxPartition
>>> from gaio.core.boxset import BoxSet
>>> from gaio.cuda.accelerated_map import AcceleratedBoxMap

>>> domain = Box([0.0, 0.0], [1.0, 1.0])
>>> pts = np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])

>>> def f(x): return x * 0.5                    # Python version
>>> @njit
... def f_jit(x): return x * 0.5               # CPU Numba version

>>> F = AcceleratedBoxMap(f, domain, pts, f_jit=f_jit, backend='cpu')
>>> P = BoxPartition(domain, [4, 4])
>>> image = F(BoxSet.full(P))
>>> isinstance(image, BoxSet)
True
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gaio.core.box import Box, F64, I64
from gaio.core.boxset import BoxSet
from gaio.cuda.backends import (
    BACKEND_PYTHON, BACKEND_CPU, BACKEND_GPU,
    THREADS_PER_BLOCK, resolve_backend, detect_gpu_dtype,
)


class AcceleratedBoxMap:
    """
    A BoxMap that accelerates the core map-evaluation loop via Numba.

    Parameters
    ----------
    f : callable
        Python map function: ``(n,) float64 → (n,) float64``.
        Always stored as the Python fallback regardless of backend.
    domain : Box
        Spatial domain.  Points outside are silently dropped.
    unit_points : ndarray, shape (M, n), float64
        Test points in the unit cube ``[-1, 1]^n``.  Same semantics as
        :class:`~gaio.maps.base.SampledBoxMap`.
    f_jit : @numba.njit callable, optional
        Numba-compiled CPU version of ``f``.  Required when
        ``backend='cpu'`` or ``backend='auto'`` with no GPU.
    f_device : @cuda.jit(device=True) callable, optional
        CUDA device function version of ``f``.  Required when
        ``backend='gpu'`` or ``backend='auto'`` with CUDA available.
    backend : str, optional
        One of ``'auto'``, ``'gpu'``, ``'cpu'``, ``'python'``.
        Default: ``'auto'``.
    threads_per_block : int, optional
        GPU threads per block.  Ignored for non-GPU backends.
        Default: 256.

    Attributes
    ----------
    backend : str
        Resolved effective backend (``'gpu'``, ``'cpu'``, or ``'python'``).
    """

    __slots__ = (
        "map", "domain", "_unit_points",
        "f_jit", "f_device",
        "backend", "dtype", "_cpu_dispatch", "_gpu_dispatch",
    )

    def __init__(
        self,
        f,
        domain: Box,
        unit_points: NDArray[F64],
        *,
        f_jit=None,
        f_device=None,
        backend: str = "auto",
        threads_per_block: int = THREADS_PER_BLOCK,
        dtype=None,
    ) -> None:
        """
        dtype : np.float32, np.float64, or None
            Compute precision for the GPU kernel.  ``None`` (default) calls
            :func:`detect_gpu_dtype` which queries
            ``CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO``
            from the CUDA driver:

            * ratio ≥ 32 → ``np.float32``  (consumer / gaming GPU)
            * ratio  < 32 → ``np.float64``  (datacenter GPU)

            Ignored when ``backend`` is not ``'gpu'``.
        """
        self.map = f
        self.domain = domain
        self._unit_points = np.ascontiguousarray(unit_points, dtype=F64)
        if self._unit_points.ndim != 2:
            raise ValueError(
                f"unit_points must be 2-D (M, n), got shape {self._unit_points.shape}."
            )
        self.f_jit = f_jit
        self.f_device = f_device

        # Resolve effective backend
        self.backend: str = resolve_backend(backend, f_jit, f_device)
        # Store dtype (resolved lazily for GPU when needed)
        self.dtype = dtype

        # Prepare dispatchers — compile/instantiate once at construction
        self._cpu_dispatch = None
        self._gpu_dispatch = None

        if self.backend == BACKEND_CPU:
            # Import lazily to allow module import without Numba
            from gaio.cuda.cpu_backend import map_parallel
            self._cpu_dispatch = map_parallel

        elif self.backend == BACKEND_GPU:
            from gaio.cuda.gpu_backend import CUDADispatcher
            self._gpu_dispatch = CUDADispatcher(
                f_device, self._unit_points,
                threads_per_block=threads_per_block, dtype=dtype,
            )

    # ------------------------------------------------------------------
    # Properties (mirror SampledBoxMap interface)
    # ------------------------------------------------------------------

    @property
    def n_test_points(self) -> int:
        """Number of test points per cell."""
        return int(self._unit_points.shape[0])

    @property
    def ndim(self) -> int:
        """Spatial dimension of the domain."""
        return self.domain.ndim

    # ------------------------------------------------------------------
    # Core operation
    # ------------------------------------------------------------------

    def __call__(self, source: BoxSet) -> BoxSet:
        """Apply the AcceleratedBoxMap to *source*."""
        return self.map_boxes(source)

    def map_boxes(self, source: BoxSet) -> BoxSet:
        """
        Compute the outer-approximation image of *source*.

        GPU path (key-based, no coordinate round-trip)::

            in_keys  = source._keys                    (K,) int64
            out_keys = CUDADispatcher(in_keys, P)      (K*M,) int64
            — kernel decodes key → test point, applies f_device,
              encodes output point → key, all on-device.

        CPU / Python path (coordinate-based, unchanged)::

            Stage 1: test_pts = centers + unit_pts * cell_r   (K*M, n) float64
            Stage 2: mapped   = map_parallel / Python loop
            Stage 3: hit_keys = partition.point_to_key_batch(mapped)

        Parameters
        ----------
        source : BoxSet

        Returns
        -------
        BoxSet
            Image cells on the same partition as *source*.
        """
        P = source.partition
        if len(source) == 0:
            return BoxSet.empty(P)

        # ── GPU path: keys in, keys out — no float64 coordinate transfer ─
        if self.backend == BACKEND_GPU:
            d_out_keys = self._gpu_dispatch(source._keys, P)  # Numba device array
            try:
                import cupy as cp
                cp_keys = cp.asarray(d_out_keys)                   # zero-copy wrap
                unique_keys = cp.unique(cp_keys[cp_keys >= 0])     # filter + dedup on GPU
                return BoxSet(P, cp.asnumpy(unique_keys).astype(I64))
            except ImportError:
                # CuPy not installed — fall back to host copy
                out_keys = d_out_keys.copy_to_host()
                valid = out_keys[out_keys >= 0]
                return BoxSet(P, np.unique(valid).astype(I64))

        # ── CPU / Python path ─────────────────────────────────────────────
        unit_pts = self._unit_points     # (M, n)
        cell_r   = P.cell_radius         # (n,)
        centers  = source.centers()      # (K, n)
        K = len(centers)
        M = self.n_test_points
        n = P.ndim

        test_pts = (
            centers[:, np.newaxis, :]
            + unit_pts[np.newaxis, :, :] * cell_r[np.newaxis, np.newaxis, :]
        ).reshape(K * M, n)

        mapped   = self._apply_map(test_pts)
        hit_keys = P.point_to_key_batch(mapped)
        valid    = hit_keys[hit_keys >= 0]
        return BoxSet(P, np.unique(valid).astype(I64))

    def _apply_map(self, test_pts: NDArray[F64]) -> NDArray[F64]:
        """
        Dispatch Stage 2 to the appropriate backend.

        Parameters
        ----------
        test_pts : ndarray, shape (N, n), float64, C-contiguous

        Returns
        -------
        ndarray, shape (N, n), float64, C-contiguous
        """
        if self.backend == BACKEND_GPU:
            # CUDADispatcher handles to_device / kernel launch / copy_to_host
            return self._gpu_dispatch(test_pts)

        if self.backend == BACKEND_CPU:
            # map_parallel handles prange over f_jit
            return self._cpu_dispatch(self.f_jit, test_pts)

        # BACKEND_PYTHON — identical to SampledBoxMap (loop retained for
        # forward-compatibility and as a reference baseline)
        N, n = test_pts.shape
        mapped = np.empty((N, n), dtype=F64)
        for i, p in enumerate(test_pts):
            mapped[i] = np.asarray(self.map(p), dtype=F64)
        return mapped

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AcceleratedBoxMap("
            f"backend='{self.backend}', "
            f"n_test_points={self.n_test_points}, "
            f"domain={self.domain})"
        )
