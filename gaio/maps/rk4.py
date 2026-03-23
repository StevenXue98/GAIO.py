"""
gaio/maps/rk4.py
================
Classical 4th-order Runge-Kutta integrator.

Correspondence with GAIO.jl
----------------------------
``rk4_step``     ↔ ``rk4(f, x, τ)``       in ``src/algorithms/maps.jl``
``rk4_flow_map`` ↔ ``rk4_flow_map(f, ...)`` in ``src/algorithms/maps.jl``

Phase 3 CUDA target
-------------------
``rk4_step`` is written in pure NumPy arithmetic — every operation is an
element-wise array op on ``float64`` vectors.  To accelerate batch
integration over many initial conditions, replace the Python loop in
``rk4_flow_map`` with a ``@cuda.jit`` kernel that calls the same four-stage
formula in parallel threads.

Usage
-----
>>> import numpy as np
>>> from gaio.maps.rk4 import rk4_step, rk4_flow_map
>>> from gaio.core.box import Box
>>> from gaio.maps.base import SampledBoxMap

>>> # Van der Pol ODE (μ=0 → simple harmonic oscillator)
>>> def vdp(x): return np.array([x[1], -x[0]])
>>> flow = rk4_flow_map(vdp, step_size=0.01, steps=100)  # integrate T=1s
>>> x0 = np.array([1.0, 0.0])
>>> x1 = flow(x0)
>>> round(float(np.linalg.norm(x1)), 1)  # energy ~conserved
1.0

>>> # Wrap as a SampledBoxMap for use in algorithms
>>> domain = Box([0.0, 0.0], [2.0, 2.0])
>>> pts = np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
>>> F = SampledBoxMap(flow, domain, pts)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gaio.core.box import F64


def rk4_step(f, x: NDArray[F64], tau: float) -> NDArray[F64]:
    """
    Perform one step of the classical RK4 scheme.

    Parameters
    ----------
    f : callable
        ODE right-hand side  f(x) → dx/dt.  Must accept and return a 1-D
        ``float64`` array of the same shape as ``x``.
    x : ndarray, shape (n,)
        Current state.
    tau : float
        Step size.

    Returns
    -------
    ndarray, shape (n,)
        State after one RK4 step.

    Notes
    -----
    Uses the classic Butcher tableau  (1/6, 1/3, 1/3, 1/6)  weights and
    (0, 1/2, 1/2, 1)  nodes.  Pure NumPy arithmetic — compatible with
    ``@numba.njit`` decoration after replacing the ``f`` call with a
    Numba-compiled function.

    Phase 3: vectorise this over a batch of initial conditions using a
    ``@cuda.jit`` kernel instead of the Python loop in :func:`rk4_flow_map`.
    """
    x = np.asarray(x, dtype=F64)
    tau_half = 0.5 * tau

    k1 = np.asarray(f(x), dtype=F64)
    k2 = np.asarray(f(x + tau_half * k1), dtype=F64)
    k3 = np.asarray(f(x + tau_half * k2), dtype=F64)
    k4 = np.asarray(f(x + tau * k3), dtype=F64)

    return x + (tau / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_flow_map(f, step_size: float = 0.01, steps: int = 20):
    """
    Build a flow map  Φ^T  by composing ``steps`` RK4 steps of size
    ``step_size`` (total integration time  T = step_size * steps).

    Parameters
    ----------
    f : callable
        ODE right-hand side  f(x) → dx/dt.
    step_size : float, optional
        Size of each RK4 step.  Default: 0.01.
    steps : int, optional
        Number of RK4 steps.  Default: 20.

    Returns
    -------
    callable
        A function  g(x) = Φ^T(x)  that maps an initial condition
        ``x`` (shape (n,)) to its state after time  T.  Pass this
        directly as ``f`` to :class:`SampledBoxMap`.

    Examples
    --------
    >>> import numpy as np
    >>> from gaio.maps.rk4 import rk4_flow_map
    >>> def linear(x): return -x          # x' = -x  →  x(t) = e^{-t} x₀
    >>> flow = rk4_flow_map(linear, step_size=0.1, steps=10)   # T=1
    >>> x0 = np.array([1.0])
    >>> x1 = flow(x0)
    >>> abs(float(x1[0]) - np.exp(-1.0)) < 1e-5
    True
    """
    def g(x: NDArray[F64]) -> NDArray[F64]:
        # Phase 3: replace this loop with a @cuda.jit batch kernel.
        state = np.asarray(x, dtype=F64)
        for _ in range(steps):
            state = rk4_step(f, state, step_size)
        return state

    return g


def make_njit_rk4_flow_map(vfield_jit, step_size: float = 0.01, steps: int = 20):
    """
    Build a ``@numba.njit`` flow map from a ``@numba.njit`` vector field.

    CPU-side counterpart of :func:`gaio.cuda.rk4_cuda.make_cuda_rk4_flow_map`.
    Pass the returned callable as ``f_jit`` to :class:`~gaio.cuda.AcceleratedBoxMap`
    when using ``backend='cpu'``.

    Parameters
    ----------
    vfield_jit : ``@numba.njit`` callable
        ODE right-hand side ``v(x) → dx/dt``.  Must accept and return a 1-D
        ``float64`` array of shape ``(n,)``.
    step_size : float, optional
        RK4 step size (dt).  Total integration time = ``step_size * steps``.
    steps : int, optional
        Number of RK4 steps.  Default: 20.

    Returns
    -------
    ``@numba.njit`` callable
        Flow map ``Φ^T(x)`` with the same ``(n,) → (n,)`` signature.
        Pass this as ``f_jit`` to :class:`~gaio.cuda.AcceleratedBoxMap`.

    Raises
    ------
    ImportError
        If ``numba`` is not installed.
    """
    try:
        import numba
    except ImportError as exc:
        raise ImportError(
            "make_njit_rk4_flow_map requires the 'numba' package.  "
            "Install with: conda install numba"
        ) from exc

    @numba.njit
    def flow(x):
        state = x.copy()
        for _ in range(steps):
            k1 = vfield_jit(state)
            k2 = vfield_jit(state + (step_size * 0.5) * k1)
            k3 = vfield_jit(state + (step_size * 0.5) * k2)
            k4 = vfield_jit(state + step_size * k3)
            state = state + (step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return state

    return flow


def rk4_flow_map_tspan(f, t0: float, t1: float, step_size: float = 0.01):
    """
    Build a flow map  Φ_{t0}^{t1}  for a **nonautonomous** ODE by composing
    RK4 steps of size ``step_size`` from ``t0`` to ``t1``.

    Parameters
    ----------
    f : callable
        ODE right-hand side  f(x, t) → dx/dt.  Signature must accept a
        1-D ``float64`` state array and a scalar time ``t``.
    t0 : float
        Integration start time.
    t1 : float
        Integration end time.  Must satisfy ``t1 > t0``.
    step_size : float, optional
        Nominal RK4 step size.  The actual step is adjusted so that an
        integer number of steps exactly spans ``[t0, t1]``.  Default: 0.01.

    Returns
    -------
    callable
        A function  g(x) = Φ_{t0}^{t1}(x)  that maps an initial condition
        ``x`` (shape (n,)) to its state at time ``t1``.  Pass this directly
        as ``f`` to :class:`SampledBoxMap`.

    Examples
    --------
    >>> import numpy as np
    >>> from gaio.maps.rk4 import rk4_flow_map_tspan
    >>> # Simple harmonic oscillator: x'' + x = 0  →  f(x,t) = [x1, -x0]
    >>> def shm(x, t): return np.array([x[1], -x[0]])
    >>> flow = rk4_flow_map_tspan(shm, t0=0.0, t1=np.pi)  # half period
    >>> x0 = np.array([1.0, 0.0])
    >>> x1 = flow(x0)
    >>> round(float(x1[0]), 4)  # x(π) ≈ -1 for cos(t)
    -1.0
    """
    T = t1 - t0
    if T <= 0:
        raise ValueError(f"t1 must be greater than t0, got t0={t0}, t1={t1}")
    n_steps = max(1, int(round(T / step_size)))
    h = T / n_steps

    def g(x: NDArray[F64]) -> NDArray[F64]:
        state = np.asarray(x, dtype=F64)
        t = t0
        for _ in range(n_steps):
            k1 = np.asarray(f(state, t), dtype=F64)
            k2 = np.asarray(f(state + 0.5 * h * k1, t + 0.5 * h), dtype=F64)
            k3 = np.asarray(f(state + 0.5 * h * k2, t + 0.5 * h), dtype=F64)
            k4 = np.asarray(f(state + h * k3, t + h), dtype=F64)
            state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
        return state

    return g
