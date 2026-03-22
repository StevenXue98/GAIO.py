"""
gaio/examples/unstable_manifold.py
====================================
Unstable manifold of the Lorenz system.

Port of: references/GAIO.jl-master/examples/unstable_manifold.jl

Algorithm
---------
1. Start with a box covering an equilibrium point of the Lorenz system.
2. ``unstable_set(F, S)`` grows ``S`` forward in time:
       repeat: S ← F(S) ∪ S
   until convergence or ``max_steps`` are reached.
3. The resulting set approximates the 2-D unstable manifold emanating
   from the equilibrium.

ODE integration
---------------
The vector field is integrated with ``rk4_flow_map(v, x, dt, n_steps)``,
which applies n_steps of RK4 with step size dt.

3-D rendering (PyVista)
-----------------------
The ~50 k–200 k box set is converted to a single pyvista UnstructuredGrid
of hexahedral cells via ``boxset_to_pyvista``.  All N cells are sent to
VTK in one draw call — far faster than adding N separate actors.

Usage
-----
    python -m gaio.examples.unstable_manifold
    python -m gaio.examples.unstable_manifold --steps 18 --no-show
"""
from __future__ import annotations

import argparse
import math
import numpy as np

from gaio import Box, BoxPartition, BoxSet, SampledBoxMap, rk4_flow_map, unstable_set


# ── Lorenz parameters ─────────────────────────────────────────────────────────
SIGMA, RHO, BETA = 10.0, 28.0, 0.4


def lorenz_v(x: np.ndarray) -> np.ndarray:
    """Lorenz vector field."""
    return np.array([
        SIGMA * (x[1] - x[0]),
        RHO * x[0] - x[1] - x[0] * x[2],
        x[0] * x[1] - BETA * x[2],
    ])


f_lorenz = rk4_flow_map(lorenz_v, step_size=0.05, steps=5)


def run(
    grid_res: int = 32,
    steps: int = 18,
    show: bool = True,
    off_screen: bool = False,
):
    """
    Compute and plot the Lorenz unstable manifold.

    Parameters
    ----------
    grid_res : int
        Resolution per spatial dimension.  Default: 32.
        (128 matches the Julia example but requires more RAM.)
    steps : int
        Number of forward-image iterations.  Default: 18.
    show : bool
        Open the pyvista render window.  Default: True.
    off_screen : bool
        Render off-screen (no window, for testing).  Default: False.
    """
    # ── Domain ────────────────────────────────────────────────────────────────
    center = np.array([0.0, 0.0, 25.0])
    radius = np.array([30.0, 30.0, 30.0])
    domain = Box(center, radius)
    P = BoxPartition(domain, [grid_res, grid_res, grid_res])

    # ── Box map (3×3 grid of test points) ─────────────────────────────────────
    t = np.array([-0.5, 0.0, 0.5])
    gx, gy, gz = np.meshgrid(t, t, t)
    unit_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # (27, 3)
    F = SampledBoxMap(f_lorenz, domain, unit_pts)

    # ── Seed: box covering the equilibrium  (√(β(ρ-1)), √(β(ρ-1)), ρ-1) ─────
    eq_x = math.sqrt(BETA * (RHO - 1.0))
    equilibrium = np.array([eq_x, eq_x, RHO - 1.0])
    eq_key = P.point_to_key(equilibrium)
    S = BoxSet(P, np.array([eq_key], dtype=np.int64))
    print(f"[unstable_manifold] Seed: {equilibrium}, key={eq_key}")

    # ── Grow unstable set ─────────────────────────────────────────────────────
    print(f"[unstable_manifold] Growing unstable set ({steps} iterations) …")
    W = unstable_set(F, S)
    print(f"[unstable_manifold] Unstable manifold: {len(W)} cells")

    # ── 3-D plot with PyVista ─────────────────────────────────────────────────
    from gaio.viz import plot_boxset_3d
    pl = plot_boxset_3d(
        W,
        color="steelblue",
        opacity=0.6,
        show_edges=False,
        title=f"Lorenz unstable manifold  ({len(W)} cells, {steps} iters)",
        show=show,
        off_screen=off_screen,
    )
    return W, pl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lorenz unstable manifold example")
    parser.add_argument("--grid-res", type=int, default=32,
                        help="Spatial resolution per dimension (default: 32)")
    parser.add_argument("--steps", type=int, default=18,
                        help="Forward-image iterations (default: 18)")
    parser.add_argument("--no-show", dest="show", action="store_false",
                        help="Do not open the render window")
    args = parser.parse_args()
    run(grid_res=args.grid_res, steps=args.steps, show=args.show)
