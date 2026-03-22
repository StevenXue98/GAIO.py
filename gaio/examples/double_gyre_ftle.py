"""
gaio/examples/double_gyre_ftle.py
==================================
Finite-time Lyapunov exponent (FTLE) field of the periodically driven
double-gyre.

Port of: references/GAIO.jl-master/examples/ftle.jl
Reference: https://gaioguys.github.io/GAIO.jl/algorithms/ftle/

What the FTLE field shows
--------------------------
The FTLE at x measures the maximum exponential rate of separation of
nearby trajectories over [t₀, t₀+T]:

    σ(x) = (1/T) · log( σ_max( DΦ^T(x) ) )

Ridges of the FTLE field are **Lagrangian coherent structures (LCS)** —
the material boundaries that organise mixing.  For the double-gyre the
main ridge separates the two gyres and oscillates with the forcing.

Algorithm
---------
For each cell centre x:
1. Evaluate flow Φ^T at x and x ± δeᵢ (finite-difference Jacobian) in
   one batched ``F._apply_map`` call — GPU-accelerated if available.
2. Largest singular value σ_max of the (2×2) Jacobian via batch SVD.
3. FTLE(x) = log(σ_max) / T.

Acceleration
------------
* GPU: NonautonomousBoxMap with autonomized CUDA kernel.  The same kernel
  handles both the ``relative_attractor``-style set ops and the FTLE
  Jacobian estimation — ``_apply_map`` dispatches both.
* CPU: vectorised NumPy RK4 (~30× faster than Python loop).

Usage
-----
    python -m gaio.examples.double_gyre_ftle
    python -m gaio.examples.double_gyre_ftle --frames 30 --res 256 128
    python -m gaio.examples.double_gyre_ftle --save ftle.gif
    python -m gaio.examples.double_gyre_ftle --no-gpu --no-show
"""
from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection

from gaio import Box, BoxPartition, BoxSet
from gaio.algorithms.ftle import finite_time_lyapunov_exponents
from gaio.examples._double_gyre import DOMAIN, T_PERIOD, build_box_map


def run(
    nx: int = 128,
    ny: int = 64,
    n_frames: int = 24,
    use_gpu: bool = True,
    show: bool = True,
    save: str | None = None,
):
    """
    Compute and animate the FTLE field of the double-gyre.

    Parameters
    ----------
    nx, ny : int
        Grid resolution.  Default: 128×64 (reveals LCS ridges clearly).
    n_frames : int
        Animation frames.  Default: 24.
    use_gpu : bool
        Use GPU if available.  Default: True.
    show : bool
        Display the animation window.
    save : str or None
        Save the animation (e.g. ``'ftle.gif'``).
    """
    P = BoxPartition(DOMAIN, [nx, ny])
    S = BoxSet.full(P)

    # FTLE uses a single centre test point — Jacobian is estimated via
    # finite differences in finite_time_lyapunov_exponents, not from
    # the spread of the test-point cloud.
    unit_pts = np.array([[0.0, 0.0]])

    t0_vals = np.linspace(0.0, 1.0, n_frames, endpoint=False)
    base_F = build_box_map(unit_pts, t0=t0_vals[0], use_gpu=use_gpu)
    print(f"[double_gyre_ftle] {base_F}")
    print(f"[double_gyre_ftle] Grid: {nx}×{ny} = {len(S)} cells, {n_frames} frames")

    ftle_frames = []
    for i, t0 in enumerate(t0_vals):
        print(f"[double_gyre_ftle] Frame {i+1}/{n_frames}  t₀={t0:.3f} …", end="\r")
        F = base_F.with_t0(t0)
        ftle_frames.append(finite_time_lyapunov_exponents(F, S, T=T_PERIOD))
    print()

    # Build vertex array once — same partition every frame
    centers = S.centers()
    r = S.cell_radius()
    c0, c1, r0, r1 = centers[:, 0], centers[:, 1], float(r[0]), float(r[1])
    N = len(c0)
    verts = np.empty((N, 4, 2))
    verts[:, 0] = np.stack([c0 - r0, c1 - r1], axis=1)
    verts[:, 1] = np.stack([c0 + r0, c1 - r1], axis=1)
    verts[:, 2] = np.stack([c0 + r0, c1 + r1], axis=1)
    verts[:, 3] = np.stack([c0 - r0, c1 + r1], axis=1)

    all_vals = np.concatenate([m._weights for m in ftle_frames])
    norm = mcolors.Normalize(vmin=float(all_vals.min()), vmax=float(all_vals.max()))
    cmap = plt.get_cmap("inferno")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, 2); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    pc = PolyCollection(verts, array=ftle_frames[0]._weights,
                        cmap=cmap, norm=norm, edgecolor="none")
    ax.add_collection(pc)
    fig.colorbar(pc, ax=ax, label="FTLE  σ(x)")
    title = ax.set_title("")

    def _update(i):
        pc.set_array(ftle_frames[i]._weights)
        title.set_text(f"FTLE field — double-gyre   t₀={t0_vals[i]:.3f},  T={T_PERIOD}")
        return pc, title

    anim = FuncAnimation(fig, _update, frames=n_frames, interval=120, blit=False)
    if save:
        print(f"[double_gyre_ftle] Saving to {save} …")
        anim.save(save, writer=PillowWriter(fps=8))
    if show:
        plt.tight_layout(); plt.show()
    return anim, fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FTLE field of the double-gyre")
    parser.add_argument("--res", type=int, nargs=2, default=[128, 64], metavar=("NX", "NY"))
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--no-gpu", dest="gpu", action="store_false")
    parser.add_argument("--no-show", dest="show", action="store_false")
    args = parser.parse_args()
    run(nx=args.res[0], ny=args.res[1], n_frames=args.frames,
        show=args.show, use_gpu=args.gpu, save=args.save)
