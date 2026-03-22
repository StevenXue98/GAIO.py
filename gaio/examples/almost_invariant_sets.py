"""
gaio/examples/almost_invariant_sets.py
=========================================
Almost-invariant sets of Chua's circuit.

Port of: references/GAIO.jl-master/examples/advanced/almost_invariant_sets.jl

Algorithm
---------
1. Seed two equilibria of Chua's circuit.
2. Grow the unstable manifold with ``unstable_set``.
3. Build a ``TransferOperator`` restricted to the manifold.
4. Compute the *second* eigenvector (λ₂ near 1 but < 1); its sign
   partition identifies two almost-invariant sets (lobes of the attractor).
5. Visualise with PyVista.

Chua's circuit
--------------
    ẋ = a·(y - m₀·x - m₁/3·x³)
    ẏ = x - y + z
    ż = -b·y

with  a=16, b=33, m₀=-0.2, m₁=0.01.

Usage
-----
    python -m gaio.examples.almost_invariant_sets
    python -m gaio.examples.almost_invariant_sets --no-show
"""
from __future__ import annotations

import argparse
import math
import numpy as np

from gaio import (
    Box, BoxPartition, BoxSet, SampledBoxMap, rk4_flow_map,
    unstable_set, TransferOperator,
)
from gaio.core.boxmeasure import BoxMeasure


# ── Chua's circuit parameters ─────────────────────────────────────────────────
A_CHUA, B_CHUA = 16.0, 33.0
M0, M1 = -0.2, 0.01


def chua_v(x: np.ndarray) -> np.ndarray:
    """Chua's circuit vector field."""
    return np.array([
        A_CHUA * (x[1] - M0 * x[0] - M1 / 3.0 * x[0] ** 3),
        x[0] - x[1] + x[2],
        -B_CHUA * x[1],
    ])


f_chua = rk4_flow_map(chua_v, step_size=0.05, steps=5)


def run(grid_res: int = 32, steps: int = 16, show: bool = True, off_screen: bool = False):
    """
    Compute and plot the Chua almost-invariant sets.

    Parameters
    ----------
    grid_res : int
        Resolution per spatial dimension.  Default: 32.
    steps : int
        Unstable manifold growth iterations.  Default: 16.
    show : bool
        Open pyvista render window.  Default: True.
    off_screen : bool
        Off-screen render (no window).  Default: False.
    """
    # ── Domain ────────────────────────────────────────────────────────────────
    center = np.array([0.0, 0.0, 0.0])
    radius = np.array([12.0, 3.0, 20.0])
    domain = Box(center, radius)
    P = BoxPartition(domain, [grid_res, grid_res, grid_res])

    # ── Box map ───────────────────────────────────────────────────────────────
    t = np.array([-0.5, 0.0, 0.5])
    gx, gy, gz = np.meshgrid(t, t, t)
    unit_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    F = SampledBoxMap(f_chua, domain, unit_pts)

    # ── Equilibria: x* = (±√(-3m₀/m₁), 0, ∓√(-3m₀/m₁)) ───────────────────
    eq_x = math.sqrt(-3.0 * M0 / M1)
    eq_pos = np.array([ eq_x, 0.0, -eq_x])
    eq_neg = np.array([-eq_x, 0.0,  eq_x])

    keys = []
    for eq in [eq_pos, eq_neg]:
        k = P.point_to_key(eq)
        if k >= 0:
            keys.append(k)
    S = BoxSet(P, np.array(keys, dtype=np.int64))
    print(f"[almost_invariant] Seed: {len(S)} cells at equilibria")

    # ── Grow unstable manifold ────────────────────────────────────────────────
    print(f"[almost_invariant] Growing unstable set ({steps} iterations) …")
    W = unstable_set(F, S)
    print(f"[almost_invariant] Unstable manifold: {len(W)} cells")

    # ── Transfer operator ─────────────────────────────────────────────────────
    print("[almost_invariant] Building TransferOperator …")
    T = TransferOperator(F, W, W)
    print(f"[almost_invariant] {T}")

    # ── Second eigenvector (almost-invariant partition) ───────────────────────
    print("[almost_invariant] Computing 2 leading eigenvectors …")
    n = len(W)
    v0 = np.ones(n, dtype=np.float64)
    eigenvalues, eigenmeasures = T.eigs(k=2, which="LR", v0=v0,
                                        maxiter=1000,
                                        tol=np.finfo(float).eps ** 0.25)
    print(f"[almost_invariant] Eigenvalues: {np.abs(eigenvalues)}")

    # The second eigenvector — its sign splits the attractor into two lobes
    mu2 = eigenmeasures[1]

    # ── 3-D plot ──────────────────────────────────────────────────────────────
    from gaio.viz import plot_boxmeasure_3d
    pl = plot_boxmeasure_3d(
        mu2,
        colormap="RdBu",
        opacity=0.8,
        show_edges=False,
        absolute_value=False,
        scalar_bar_title="ψ₂",
        title=f"Chua almost-invariant sets  ({len(W)} cells)",
        show=show,
        off_screen=off_screen,
    )
    return mu2, pl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chua almost-invariant sets")
    parser.add_argument("--grid-res", type=int, default=32)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--no-show", dest="show", action="store_false")
    args = parser.parse_args()
    run(grid_res=args.grid_res, steps=args.steps, show=args.show)
