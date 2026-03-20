"""
gaio/examples/four_wing.py
============================
Four-wing attractor via the relative attractor algorithm.

Port of: references/GAIO.jl-master/examples/advanced/four-wing.jl

The four-wing ODE system
------------------------
    ẋ = a·x + y·z
    ẏ = d·y + b·x - z·y        (note: Julia uses b*x - z*y, we match that)
    ż = -z - x·y

with  a=0.2, b=-0.01, d=-0.4.

Visualisation
-------------
* PyVista: 3-D hex-cell plot of the attractor (main view)
* matplotlib: two 2-D projections of the invariant measure
  (xy-plane and yz-plane) — matching the Julia ``plot(μ, projection=...)``

Usage
-----
    python -m gaio.examples.four_wing
    python -m gaio.examples.four_wing --steps 21 --no-show
"""
from __future__ import annotations

import argparse
import numpy as np

from gaio import (
    Box, BoxPartition, BoxSet, GridMap, rk4_flow_map,
    relative_attractor, TransferOperator, cuda_available, AcceleratedBoxMap,
)
from gaio.viz import plot_boxset_3d, plot_boxmeasure


# ── Four-wing parameters ──────────────────────────────────────────────────────
A_FW, B_FW, D_FW = 0.2, -0.01, -0.4


def four_wing_v(x: np.ndarray) -> np.ndarray:
    """Four-wing vector field."""
    return np.array([
        A_FW * x[0] + x[1] * x[2],
        D_FW * x[1] + B_FW * x[0] - x[2] * x[1],
        -x[2] - x[0] * x[1],
    ])


def f_four_wing(x: np.ndarray) -> np.ndarray:
    """RK4 flow map: 20 steps × dt=0.01."""
    return rk4_flow_map(four_wing_v, x, dt=0.01, n_steps=20)


def _make_gpu_map(domain, unit_pts):
    from numba import cuda

    @cuda.jit(device=True)
    def f_device(x, out):
        out[0] = A_FW * x[0] + x[1] * x[2]
        out[1] = D_FW * x[1] + B_FW * x[0] - x[2] * x[1]
        out[2] = -x[2] - x[0] * x[1]
        # Note: this is one RK4 step; for multi-step we fall back to Python
        # because device functions cannot call host-side rk4_flow_map.
        # For a fair GPU example, use n_steps=1 at a smaller dt.

    # Use dt=0.01, n_steps=1 for GPU (single RK4 step per call)
    # Compensate by running more subdivision steps
    return AcceleratedBoxMap(f_four_wing, domain, unit_pts,
                              f_device=f_device, backend="gpu")


def run(
    grid_res: int = 16,
    steps: int = 21,
    use_gpu: bool = True,
    show_3d: bool = True,
    show_2d: bool = True,
    off_screen: bool = False,
):
    """
    Compute and visualise the four-wing attractor.

    Parameters
    ----------
    grid_res : int
        Resolution per spatial dimension.  Default: 16
        (Julia uses 2; we use 16 for a finer attractor image).
    steps : int
        Subdivision steps.  Default: 21.
    use_gpu : bool
        Use GPU if available.  Default: True.
    show_3d : bool
        Open pyvista window.  Default: True.
    show_2d : bool
        Open matplotlib window.  Default: True.
    off_screen : bool
        Off-screen 3-D render.  Default: False.
    """
    import matplotlib.pyplot as plt

    # ── Domain ────────────────────────────────────────────────────────────────
    center = np.array([0.0, 0.0, 0.0])
    radius = np.array([5.0, 5.0, 5.0])
    domain = Box(center, radius)
    P = BoxPartition(domain, [grid_res, grid_res, grid_res])

    # ── Box map ───────────────────────────────────────────────────────────────
    t = np.array([-0.5, 0.0, 0.5])
    gx, gy, gz = np.meshgrid(t, t, t)
    unit_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    gpu_used = False
    F = GridMap(f_four_wing, domain, unit_pts)
    print(f"[four_wing] Using Python backend  (GPU={gpu_used})")

    # ── Relative attractor ────────────────────────────────────────────────────
    S = BoxSet.full(P)
    print(f"[four_wing] Computing attractor ({steps} steps, {len(S)} initial cells) …")
    A = relative_attractor(F, S, steps=steps)
    print(f"[four_wing] Attractor: {len(A)} cells")

    # ── Invariant measure ─────────────────────────────────────────────────────
    print("[four_wing] Building TransferOperator …")
    T = TransferOperator(F, A, A)
    print(f"[four_wing] {T}")
    print("[four_wing] Computing leading eigenvector …")
    eigenvalues, eigenmeasures = T.eigs(k=1)
    mu_inv = eigenmeasures[0]
    print(f"[four_wing] Leading eigenvalue: {eigenvalues[0]:.6f}")

    # Apply log|·| for visual contrast (matching Julia's  μ = log ∘ abs ∘ ev[1])
    from gaio.core.boxmeasure import BoxMeasure
    log_weights = np.log(np.abs(mu_inv._weights) + 1e-12)
    mu_log = BoxMeasure(P, mu_inv._keys.copy(), log_weights)

    # ── 3-D plot ──────────────────────────────────────────────────────────────
    pl = plot_boxset_3d(
        A,
        color="steelblue",
        opacity=0.5,
        show_edges=False,
        title=f"Four-wing attractor  ({len(A)} cells)",
        show=show_3d,
        off_screen=off_screen,
    )

    # ── 2-D projection plots (matching Julia four-wing.jl) ───────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    plot_boxmeasure(mu_log, ax=ax1, projection=[0, 1], colormap="viridis",
                    absolute_value=False, colorbar=True,
                    title="log|ψ₁|  projection x–y")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")

    plot_boxmeasure(mu_log, ax=ax2, projection=[1, 2], colormap="viridis",
                    absolute_value=False, colorbar=True,
                    title="log|ψ₁|  projection y–z")
    ax2.set_xlabel("y"); ax2.set_ylabel("z")

    plt.suptitle("Four-wing attractor — invariant measure projections")
    plt.tight_layout()

    if show_2d:
        plt.show()

    return A, mu_log, pl, fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Four-wing attractor example")
    parser.add_argument("--grid-res", type=int, default=16)
    parser.add_argument("--steps", type=int, default=21)
    parser.add_argument("--no-gpu", dest="gpu", action="store_false")
    parser.add_argument("--no-show", dest="show", action="store_false")
    args = parser.parse_args()
    run(grid_res=args.grid_res, steps=args.steps, use_gpu=args.gpu,
        show_3d=args.show, show_2d=args.show)
