"""
gaio/examples/_double_gyre.py
==============================
Shared double-gyre system definition for the three double-gyre examples.

Provides three implementations of the same flow map, in order of speed:

1. ``make_map_python(t0)``        — plain Python (``rk4_flow_map_tspan``), slowest
2. ``make_map_numpy(t0)``         — vectorised NumPy RK4, ~30× faster than Python
3. ``make_map_gpu(t0, unit_pts)`` — CUDA GPU via autonomization, fastest for large N

All three implement the same double-gyre flow and produce numerically
identical results to within floating-point rounding.

Double-gyre vector field
------------------------
    ẋ = -π A sin(π f(x,t)) cos(π y)
    ẏ =  π A cos(π f(x,t)) sin(π y) · ∂f/∂x

    f(x,t)  = ε sin(ωt) x² + (1 - 2ε sin(ωt)) x
    ∂f/∂x   = 2ε sin(ωt) x + (1 - 2ε sin(ωt))

Parameters: A=0.25, ε=0.25, ω=2π, period T=1.
Domain: [0,2] × [0,1]  →  Box(center=[1,0.5], radius=[1,0.5]).
"""
from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray

from gaio.core.box import Box, F64
from gaio.maps.nonautonomous import NonautonomousBoxMap
from gaio.maps.rk4 import rk4_flow_map_tspan

# ── System parameters ─────────────────────────────────────────────────────────
A_DG     = 0.25
EPS_DG   = 0.25
OMEGA    = 2.0 * math.pi
T_PERIOD = 1.0

# ── Domain ────────────────────────────────────────────────────────────────────
DOMAIN = Box(np.array([1.0, 0.5]), np.array([1.0, 0.5]))   # [0,2]×[0,1]


# ── Scalar Python vector field (for rk4_flow_map_tspan) ──────────────────────

def double_gyre_v(x: NDArray[F64], t: float) -> NDArray[F64]:
    """Double-gyre nonautonomous vector field.  x = [x_coord, y_coord]."""
    eps_sin = EPS_DG * math.sin(OMEGA * t)
    fx      = eps_sin * x[0] * x[0] + (1.0 - 2.0 * eps_sin) * x[0]
    dfdx    = 2.0 * eps_sin * x[0] + (1.0 - 2.0 * eps_sin)
    u = -math.pi * A_DG * math.sin(math.pi * fx)  * math.cos(math.pi * x[1])
    v =  math.pi * A_DG * math.cos(math.pi * fx)  * math.sin(math.pi * x[1]) * dfdx
    return np.array([u, v])


# ── Batched NumPy vector field (for vectorised CPU path) ─────────────────────

def double_gyre_v_batch(pts: NDArray[F64], t: float) -> NDArray[F64]:
    """
    Batched double-gyre vector field.

    Parameters
    ----------
    pts : ndarray, shape (N, 2)
        Current positions of N particles.
    t : float
        Current time.

    Returns
    -------
    ndarray, shape (N, 2)
        Velocity field evaluated at each particle.
    """
    x, y    = pts[:, 0], pts[:, 1]
    eps_sin = EPS_DG * np.sin(OMEGA * t)
    fx      = eps_sin * x * x + (1.0 - 2.0 * eps_sin) * x
    dfdx    = 2.0 * eps_sin * x + (1.0 - 2.0 * eps_sin)
    u = -np.pi * A_DG * np.sin(np.pi * fx) * np.cos(np.pi * y)
    v =  np.pi * A_DG * np.cos(np.pi * fx) * np.sin(np.pi * y) * dfdx
    return np.stack([u, v], axis=1)


# ── Factory functions ─────────────────────────────────────────────────────────

def make_map_python(unit_pts: NDArray[F64], t0: float,
                    step_size: float = 0.01) -> NonautonomousBoxMap:
    """
    Build a NonautonomousBoxMap using the plain Python RK4 path (slowest).
    Use as a fallback when neither NumPy-vectorised nor GPU paths are needed.
    """
    f_cpu = rk4_flow_map_tspan(double_gyre_v, t0=t0, t1=t0 + T_PERIOD,
                                step_size=step_size)
    return NonautonomousBoxMap(
        f_cpu, DOMAIN, unit_pts, t0=t0, T=T_PERIOD, step_size=step_size,
    )


def make_map_numpy(unit_pts: NDArray[F64], t0: float,
                   step_size: float = 0.01) -> NonautonomousBoxMap:
    """
    Build a NonautonomousBoxMap using the vectorised NumPy RK4 path.

    All K×M test points are integrated simultaneously using NumPy
    broadcasting — no per-point Python calls.  Typically 20–50× faster
    than the Python loop on CPU.
    """
    f_cpu = rk4_flow_map_tspan(double_gyre_v, t0=t0, t1=t0 + T_PERIOD,
                                step_size=step_size)
    return NonautonomousBoxMap(
        f_cpu, DOMAIN, unit_pts, t0=t0, T=T_PERIOD, step_size=step_size,
        f_batch=double_gyre_v_batch,
    )


def make_map_gpu(unit_pts: NDArray[F64], t0: float,
                 step_size: float = 0.01) -> NonautonomousBoxMap:
    """
    Build a NonautonomousBoxMap using the CUDA GPU path.

    Uses the autonomized 3D system (x, y, τ) with τ̇ = 1.  The compiled
    CUDA kernel is shared across all frames via ``with_t0()``.

    Raises ``RuntimeError`` if CUDA is unavailable or Numba is not installed.
    """
    from numba import cuda
    from gaio.cuda.rk4_cuda import make_cuda_rk4_flow_map

    @cuda.jit(device=True)
    def _dg_vfield_3d(x, out):
        """
        Autonomized double-gyre vector field for CUDA.

        x[0] = x-coordinate
        x[1] = y-coordinate
        x[2] = time τ  (τ̇ = 1, captured as 3rd state variable)
        """
        t       = x[2]
        eps_sin = EPS_DG * math.sin(OMEGA * t)
        fx      = eps_sin * x[0] * x[0] + (1.0 - 2.0 * eps_sin) * x[0]
        dfdx    = 2.0 * eps_sin * x[0] + (1.0 - 2.0 * eps_sin)
        out[0]  = -math.pi * A_DG * math.sin(math.pi * fx)  * math.cos(math.pi * x[1])
        out[1]  =  math.pi * A_DG * math.cos(math.pi * fx)  * math.sin(math.pi * x[1]) * dfdx
        out[2]  = 1.0   # τ̇ = 1

    n_steps = max(1, int(round(T_PERIOD / step_size)))
    f_device_3d = make_cuda_rk4_flow_map(
        _dg_vfield_3d, ndim=3, step_size=step_size, steps=n_steps,
    )

    f_cpu = rk4_flow_map_tspan(double_gyre_v, t0=t0, t1=t0 + T_PERIOD,
                                step_size=step_size)
    return NonautonomousBoxMap(
        f_cpu, DOMAIN, unit_pts, t0=t0, T=T_PERIOD, step_size=step_size,
        f_batch=double_gyre_v_batch,   # numpy fallback if GPU dispatch fails
        f_device_3d=f_device_3d,
    )


def build_box_map(unit_pts: NDArray[F64], t0: float,
                  use_gpu: bool = True,
                  step_size: float = 0.01) -> NonautonomousBoxMap:
    """
    Build the fastest available NonautonomousBoxMap for the double-gyre.

    Tries GPU → NumPy vectorised → Python loop in that order.

    Parameters
    ----------
    unit_pts : ndarray, shape (M, 2)
    t0 : float
        Initial start time (use ``with_t0()`` to change per frame).
    use_gpu : bool
        Whether to attempt the GPU path.  Default: True.
    step_size : float
        RK4 step size.  Default: 0.01.

    Returns
    -------
    NonautonomousBoxMap
        The fastest available map.  Check ``.backend`` property.
    """
    if use_gpu:
        try:
            from gaio import cuda_available
            if cuda_available():
                F = make_map_gpu(unit_pts, t0, step_size=step_size)
                return F
        except Exception as e:
            print(f"[double_gyre] GPU init failed ({e}), falling back to NumPy")

    return make_map_numpy(unit_pts, t0, step_size=step_size)
