"""
gaio/maps/nonautonomous.py
==========================
NonautonomousBoxMap — a SampledBoxMap for time-dependent (nonautonomous) ODEs.

Problem
-------
For autonomous systems, the map ``f: x → Φ^T(x)`` is fixed and can be
compiled once (Numba CPU/GPU).  For nonautonomous systems, the flow map
``Φ_{t₀}^{t₀+T}`` changes with the start time ``t₀``, making per-frame
recompilation impractical.

Two acceleration strategies
----------------------------

**Vectorized NumPy (CPU fast path)**
    Supply ``f_batch(pts, t) → ndarray`` — a function that evaluates the
    vector field on an ``(N, ndim)`` array at scalar time ``t``.
    ``_apply_map`` runs the full RK4 integration using NumPy broadcasting
    over all N test points simultaneously.  Avoids K×M individual Python
    calls; gives 20–50× speedup over the Python loop.

**CUDA GPU path (autonomization)**
    Supply ``f_device_3d`` — a ``@cuda.jit(device=True)`` function for the
    *autonomized* (ndim+1)-dimensional system where the last component is
    time ``τ``, evolving as ``τ̇ = 1``.

    ``_apply_map`` augments the 2D test points with a ``t₀`` column to form
    3D input, dispatches to the GPU via :class:`~gaio.cuda.gpu_backend.CUDADispatcher`,
    then projects the output back to 2D.  The same compiled kernel is reused
    for every frame; only the ``t₀`` column of the input changes.

    Concretely, for a 2D nonautonomous system with vector field ``v(x, t)``,
    write the 3D autonomous version::

        @cuda.jit(device=True)
        def v_device_3d(x, out):
            # x = [x₀, x₁, τ]
            t = x[2]
            out[0] = ...   # v₁(x, t)
            out[1] = ...   # v₂(x, t)
            out[2] = 1.0   # τ̇ = 1

    Then pass it to ``make_cuda_rk4_flow_map`` with ``ndim=3``, and supply
    the resulting device function as ``f_device_3d``.

Cross-frame reuse
-----------------
Use ``with_t0(new_t0)`` to get a new instance for the next animation frame.
It shares the compiled ``_gpu_dispatch`` and ``_f_batch`` references — no
recompilation, no memory copies::

    base_F = NonautonomousBoxMap(f, domain, unit_pts, t0=0.0, T=1.0,
                                  f_batch=v_batch, f_device_3d=f_dev3d)
    for t0 in t0_frames:
        F = base_F.with_t0(t0)
        T_op = TransferOperator(F, S, S)
        ...
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gaio.core.box import Box, F64, I64
from gaio.maps.base import SampledBoxMap


class NonautonomousBoxMap(SampledBoxMap):
    """
    A SampledBoxMap for nonautonomous ODEs with vectorised CPU and GPU paths.

    Parameters
    ----------
    f : callable
        Python map: ``f(x: ndarray[n]) → ndarray[n]`` for the flow
        ``Φ_{t₀}^{t₀+T}``.  Used as the fallback Python path and stored
        as ``self.map``.
    domain : Box
        Spatial domain (ndim-dimensional).
    unit_points : ndarray, shape (M, ndim)
        Test points in the unit cube ``[-1,1]^ndim``.
    t0 : float
        Integration start time.
    T : float
        Integration duration (``t1 = t0 + T``).
    step_size : float, optional
        RK4 step size.  Number of steps is ``round(T / step_size)``.
        Default: 0.01.
    f_batch : callable, optional
        Vectorised vector field: ``f_batch(pts, t) → ndarray`` where
        ``pts`` has shape ``(N, ndim)`` and the return value has the same
        shape.  When provided, the CPU path runs a fully vectorised NumPy
        RK4 loop — no per-point Python calls.
    f_device_3d : ``@cuda.jit(device=True)`` callable, optional
        CUDA device function for the autonomized ``(ndim+1)``-dimensional
        system (last component = time, last derivative = 1.0).  When
        provided, the GPU path is used.  Takes priority over ``f_batch``.
    """

    def __init__(
        self,
        f,
        domain: Box,
        unit_points: NDArray[F64],
        t0: float,
        T: float,
        *,
        step_size: float = 0.01,
        f_batch=None,
        f_device_3d=None,
    ) -> None:
        super().__init__(f, domain, unit_points)
        self._t0: float = float(t0)
        self._T: float = float(T)
        n_steps = max(1, int(round(T / step_size)))
        self._n_steps: int = n_steps
        self._h: float = T / n_steps
        self._f_batch = f_batch
        self._gpu_dispatch = None

        if f_device_3d is not None:
            from gaio.cuda.gpu_backend import CUDADispatcher
            self._gpu_dispatch = CUDADispatcher(f_device_3d)

    # ------------------------------------------------------------------
    # Core dispatch — override Stage 2
    # ------------------------------------------------------------------

    def _apply_map(self, test_pts: NDArray[F64]) -> NDArray[F64]:
        """
        Apply the flow map to all rows of *test_pts* via the fastest
        available backend.

        Priority: GPU > vectorised NumPy > Python loop (base class).
        """
        # ── GPU: autonomize → dispatch → project ──────────────────────
        if self._gpu_dispatch is not None:
            N, ndim = test_pts.shape
            pts_aug = np.empty((N, ndim + 1), dtype=F64)
            pts_aug[:, :ndim] = test_pts
            pts_aug[:, ndim] = self._t0
            mapped_aug = self._gpu_dispatch(pts_aug)  # (N, ndim+1)
            return np.ascontiguousarray(mapped_aug[:, :ndim], dtype=F64)

        # ── Vectorised NumPy CPU: batch RK4 ───────────────────────────
        if self._f_batch is not None:
            state = np.ascontiguousarray(test_pts, dtype=F64)
            t = self._t0
            h = self._h
            for _ in range(self._n_steps):
                k1 = self._f_batch(state, t)
                k2 = self._f_batch(state + 0.5 * h * k1, t + 0.5 * h)
                k3 = self._f_batch(state + 0.5 * h * k2, t + 0.5 * h)
                k4 = self._f_batch(state + h * k3, t + h)
                state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                t += h
            return state

        # ── Python loop fallback ───────────────────────────────────────
        return super()._apply_map(test_pts)

    # ------------------------------------------------------------------
    # Frame-level update
    # ------------------------------------------------------------------

    def with_t0(self, new_t0: float) -> "NonautonomousBoxMap":
        """
        Return a new instance for start time *new_t0*, reusing the
        compiled GPU kernel and batch function — no recompilation.

        This is the intended pattern for animation loops::

            base_F = NonautonomousBoxMap(...)
            for t0 in frame_times:
                F = base_F.with_t0(t0)
                T_op = TransferOperator(F, S, S)
                ...

        Parameters
        ----------
        new_t0 : float
            New integration start time.

        Returns
        -------
        NonautonomousBoxMap
            Shallow copy with ``_t0 = new_t0`` and shared dispatchers.
        """
        m = NonautonomousBoxMap.__new__(NonautonomousBoxMap)
        m.map          = self.map
        m.domain       = self.domain
        m._unit_points = self._unit_points
        m._t0          = float(new_t0)
        m._T           = self._T
        m._n_steps     = self._n_steps
        m._h           = self._h
        m._f_batch     = self._f_batch
        m._gpu_dispatch = self._gpu_dispatch   # shared — no recompilation
        return m

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        backend = (
            "gpu" if self._gpu_dispatch is not None
            else "cpu-vectorised" if self._f_batch is not None
            else "python"
        )
        return (
            f"NonautonomousBoxMap("
            f"backend='{backend}', "
            f"t0={self._t0:.4f}, T={self._T}, "
            f"n_steps={self._n_steps}, "
            f"n_test_points={self.n_test_points})"
        )
