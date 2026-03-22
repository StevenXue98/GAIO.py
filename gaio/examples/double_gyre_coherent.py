"""
gaio/examples/double_gyre_coherent.py
======================================
Coherent sets of the periodically driven double-gyre.

Port of: references/GAIO.jl-master/examples/coherent.jl
Reference: https://gaioguys.github.io/GAIO.jl/algorithms/coherent/

Coherent sets vs almost-invariant sets
---------------------------------------
* **Almost-invariant sets** (eigenvectors) suit *autonomous* systems.
* **Coherent sets** (singular vectors) suit *nonautonomous* systems:
  pairs (A₀, A_T) such that points in A₀ are transported to A_T with
  minimal mixing.  A₀ comes from the right singular vector (domain) and
  A_T from the left singular vector (codomain).

Algorithm
---------
For each start time t₀:
1. Build NonautonomousBoxMap for flow Φ_{t₀}^{t₀+T}.
2. Compute TransferOperator T.
3. SVD: U Σ Vᵀ = T.  Second right singular vector → A₀; second left → A_T.
4. Animate over t₀ ∈ [0, T).

Acceleration
------------
Same as double_gyre_almost_invariant: GPU (autonomized CUDA) if available,
otherwise vectorised NumPy.  The compiled kernel is built once and shared
across all frames via ``with_t0()``.

Usage
-----
    python -m gaio.examples.double_gyre_coherent
    python -m gaio.examples.double_gyre_coherent --frames 30 --res 128 64
    python -m gaio.examples.double_gyre_coherent --save coherent.gif
    python -m gaio.examples.double_gyre_coherent --no-gpu --no-show
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
    Compute and animate coherent sets of the double-gyre.

    Parameters
    ----------
    nx, ny : int
        Grid resolution.  Default: 64×32.
    n_frames : int
        Animation frames.  Default: 24.
    n_test : int
        Test points per dimension.  Default: 4.
    use_gpu : bool
        Use GPU if available.  Default: True.
    show : bool
        Display the animation window.
    save : str or None
        Save the animation (e.g. ``'coherent.gif'``).
    """
    P = BoxPartition(DOMAIN, [nx, ny])
    S = BoxSet.full(P)

    t1d = np.linspace(-0.9, 0.9, n_test)
    gx, gy = np.meshgrid(t1d, t1d)
    unit_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

    t0_vals = np.linspace(0.0, 1.0, n_frames, endpoint=False)
    base_F = build_box_map(unit_pts, t0=t0_vals[0], use_gpu=use_gpu)
    print(f"[double_gyre_coherent] {base_F}")
    print(f"[double_gyre_coherent] Grid: {nx}×{ny} = {len(S)} cells, {n_frames} frames")

    frames_data = []
    for i, t0 in enumerate(t0_vals):
        print(f"[double_gyre_coherent] Frame {i+1}/{n_frames}  t₀={t0:.3f} …", end="\r")
        F = base_F.with_t0(t0)
        T_op = TransferOperator(F, S, S)
        U, s, V = T_op.svds(k=3)
        # svds returns ascending order — reverse
        idx = np.argsort(-s)
        v2, u2 = V[idx[1]], U[idx[1]]

        def _split(keys, weights):
            pos = keys[weights >  0]
            neg = keys[weights <= 0]
            return (BoxSet(P, pos) if len(pos) else BoxSet.empty(P),
                    BoxSet(P, neg) if len(neg) else BoxSet.empty(P))

        A0p, A0n = _split(v2._keys, v2._weights)
        ATp, ATn = _split(u2._keys, u2._weights)
        frames_data.append((A0p, A0n, ATp, ATn, float(s[idx[1]])))
    print()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 4))
    for ax, lbl in [(ax0, "A₀  (start, right sing. vec.)"),
                    (ax1, "A_T  (end, left sing. vec.)")]:
        ax.set_xlim(0, 2); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(lbl)

    c_blue, c_red = "#2166ac", "#d6604d"
    colls = [PolyCollection([], facecolor=c_blue, alpha=0.85),
             PolyCollection([], facecolor=c_red,  alpha=0.85),
             PolyCollection([], facecolor=c_blue, alpha=0.85),
             PolyCollection([], facecolor=c_red,  alpha=0.85)]
    ax0.add_collection(colls[0]); ax0.add_collection(colls[1])
    ax1.add_collection(colls[2]); ax1.add_collection(colls[3])
    sup = fig.suptitle("")

    def _update(i):
        A0p, A0n, ATp, ATn, sig2 = frames_data[i]
        colls[0].set_verts(_boxset_verts(A0p))
        colls[1].set_verts(_boxset_verts(A0n))
        colls[2].set_verts(_boxset_verts(ATp))
        colls[3].set_verts(_boxset_verts(ATn))
        sup.set_text(f"Coherent sets — double-gyre   t₀={t0_vals[i]:.3f}   σ₂={sig2:.5f}")
        return (*colls, sup)

    anim = FuncAnimation(fig, _update, frames=n_frames, interval=120, blit=False)
    if save:
        print(f"[double_gyre_coherent] Saving to {save} …")
        anim.save(save, writer=PillowWriter(fps=8))
    if show:
        plt.tight_layout()
        plt.show()
    return anim, fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coherent sets of the double-gyre")
    parser.add_argument("--res", type=int, nargs=2, default=[64, 32], metavar=("NX", "NY"))
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--test-pts", type=int, default=4)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--no-gpu", dest="gpu", action="store_false")
    parser.add_argument("--no-show", dest="show", action="store_false")
    args = parser.parse_args()
    run(nx=args.res[0], ny=args.res[1], n_frames=args.frames,
        n_test=args.test_pts, use_gpu=args.gpu, show=args.show, save=args.save)
