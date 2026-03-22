"""
gaio/examples/double_gyre_almost_invariant.py
==============================================
Almost-invariant (metastable) sets of the periodically driven double-gyre.

Port of: references/GAIO.jl-master/examples/almost_invariant.jl (double-gyre variant)
Reference: https://gaioguys.github.io/GAIO.jl/algorithms/almost_invariant/

The double-gyre system
-----------------------
A nonautonomous flow on [0,2] × [0,1] mimicking two counter-rotating
gyres that periodically exchange fluid:

    ẋ = -π A sin(π f(x,t)) cos(π y)
    ẏ =  π A cos(π f(x,t)) sin(π y) · ∂f/∂x

    f(x,t) = ε sin(ωt) x² + (1 - 2ε sin(ωt)) x

with  A=0.25, ε=0.25, ω=2π  (period T=1).

Algorithm
---------
For each start time t₀ in [0, T):

1. Build a NonautonomousBoxMap for the flow Φ_{t₀}^{t₀+T}.
2. Compute a TransferOperator on the full domain.
3. Find the second eigenvector (eigenvalue λ₂ ≲ 1); its sign partitions
   the domain into two almost-invariant sets.
4. Animate as t₀ sweeps through one period.

Acceleration
------------
* GPU available  → CUDA path via autonomized 3D system (τ̇=1 appended).
  The compiled kernel is built once and reused across all frames.
* GPU unavailable → vectorised NumPy RK4 (~30× faster than Python loop).

Usage
-----
    python -m gaio.examples.double_gyre_almost_invariant
    python -m gaio.examples.double_gyre_almost_invariant --frames 30 --res 128 64
    python -m gaio.examples.double_gyre_almost_invariant --save almost_invariant.gif
    python -m gaio.examples.double_gyre_almost_invariant --no-gpu --no-show
"""
from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection

from gaio import Box, BoxPartition, BoxSet, TransferOperator
from gaio.examples._double_gyre import DOMAIN, build_box_map


def _boxset_verts(bs: BoxSet) -> np.ndarray:
    if bs.is_empty():
        return np.empty((0, 4, 2))
    c = bs.centers()
    r = bs.cell_radius()
    c0, c1, r0, r1 = c[:, 0], c[:, 1], float(r[0]), float(r[1])
    N = len(c0)
    v = np.empty((N, 4, 2))
    v[:, 0] = np.stack([c0 - r0, c1 - r1], axis=1)
    v[:, 1] = np.stack([c0 + r0, c1 - r1], axis=1)
    v[:, 2] = np.stack([c0 + r0, c1 + r1], axis=1)
    v[:, 3] = np.stack([c0 - r0, c1 + r1], axis=1)
    return v


def run(
    nx: int = 64,
    ny: int = 32,
    n_frames: int = 24,
    n_test: int = 4,
    use_gpu: bool = True,
    show: bool = True,
    save: str | None = None,
):
    """
    Compute and animate almost-invariant sets of the double-gyre.

    Parameters
    ----------
    nx, ny : int
        Grid resolution.  Default: 64×32.
    n_frames : int
        Animation frames (t₀ steps over one period).  Default: 24.
    n_test : int
        Test points per dimension (n_test² per cell).  Default: 4.
    use_gpu : bool
        Use GPU if available.  Default: True.
    show : bool
        Display the animation window.
    save : str or None
        Save the animation to this path (e.g. ``'out.gif'``).
    """
    P = BoxPartition(DOMAIN, [nx, ny])
    S = BoxSet.full(P)

    t1d = np.linspace(-0.9, 0.9, n_test)
    gx, gy = np.meshgrid(t1d, t1d)
    unit_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

    # Build base map once — compiles GPU kernel on first construction
    t0_vals = np.linspace(0.0, 1.0, n_frames, endpoint=False)
    base_F = build_box_map(unit_pts, t0=t0_vals[0], use_gpu=use_gpu)
    print(f"[double_gyre_ai] {base_F}")
    print(f"[double_gyre_ai] Grid: {nx}×{ny} = {len(S)} cells, {n_frames} frames")

    # Pre-compute all frames (with_t0 reuses compiled kernel, no recompilation)
    frames_data = []
    for i, t0 in enumerate(t0_vals):
        print(f"[double_gyre_ai] Frame {i+1}/{n_frames}  t₀={t0:.3f} …", end="\r")
        F = base_F.with_t0(t0)
        T_op = TransferOperator(F, S, S)
        vals, vecs = T_op.eigs(k=3)
        order = np.argsort(-np.abs(vals))
        mu2 = vecs[order[1]]
        w = mu2._weights
        pos = BoxSet(P, mu2._keys[w >  0]) if (w >  0).any() else BoxSet.empty(P)
        neg = BoxSet(P, mu2._keys[w <= 0]) if (w <= 0).any() else BoxSet.empty(P)
        frames_data.append((pos, neg, float(np.abs(vals[order[1]]))))
    print()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, 2); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    coll_pos = PolyCollection([], facecolor="#2166ac", alpha=0.85)
    coll_neg = PolyCollection([], facecolor="#d6604d", alpha=0.85)
    ax.add_collection(coll_pos); ax.add_collection(coll_neg)
    title = ax.set_title("")

    def _update(i):
        pos, neg, lam2 = frames_data[i]
        coll_pos.set_verts(_boxset_verts(pos))
        coll_neg.set_verts(_boxset_verts(neg))
        title.set_text(
            f"Almost-invariant sets — double-gyre  t₀={t0_vals[i]:.3f}  |λ₂|={lam2:.5f}")
        return coll_pos, coll_neg, title

    anim = FuncAnimation(fig, _update, frames=n_frames, interval=120, blit=False)
    if save:
        print(f"[double_gyre_ai] Saving to {save} …")
        anim.save(save, writer=PillowWriter(fps=8))
    if show:
        plt.tight_layout(); plt.show()
    return anim, fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Almost-invariant sets of the double-gyre")
    parser.add_argument("--res", type=int, nargs=2, default=[64, 32], metavar=("NX", "NY"))
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--test-pts", type=int, default=4)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--no-gpu", dest="gpu", action="store_false")
    parser.add_argument("--no-show", dest="show", action="store_false")
    args = parser.parse_args()
    run(nx=args.res[0], ny=args.res[1], n_frames=args.frames,
        n_test=args.test_pts, use_gpu=args.gpu, show=args.show, save=args.save)
