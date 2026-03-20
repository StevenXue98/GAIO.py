"""
gaio/examples/invariant_measure_2d.py
========================================
Invariant measure of the Hénon map (2-D).

Port of: references/GAIO.jl-master/examples/advanced/invariant_measure_2d.jl

Algorithm
---------
1. Compute the Hénon attractor with ``relative_attractor``.
2. Build ``TransferOperator(F, A, A)`` restricted to the attractor.
3. Find the leading eigenvector (eigenvalue ≈ 1) — this is the SRB
   (Sinai-Ruelle-Bowen) invariant measure supported on the attractor.
4. Plot as a colormapped heatmap with matplotlib.

GPU acceleration
----------------
When a CUDA device is available the ``AcceleratedBoxMap`` GPU backend is
used for both the attractor computation and the transfer operator seed.

Usage
-----
    python -m gaio.examples.invariant_measure_2d
    python -m gaio.examples.invariant_measure_2d --steps 14 --no-gpu
"""
from __future__ import annotations

import argparse
import numpy as np

from gaio import (
    Box, BoxPartition, BoxSet, GridMap, relative_attractor, TransferOperator,
    cuda_available, AcceleratedBoxMap,
)
from gaio.viz import plot_boxmeasure


A_PARAM, B_PARAM = 1.4, 0.3


def f_henon(x: np.ndarray) -> np.ndarray:
    return np.array([1.0 - A_PARAM * x[0] ** 2 + x[1], B_PARAM * x[0]])


def _make_gpu_map(domain, unit_pts):
    from numba import cuda

    @cuda.jit(device=True)
    def f_device(x, out):
        out[0] = 1.0 - A_PARAM * x[0] * x[0] + x[1]
        out[1] = B_PARAM * x[0]

    return AcceleratedBoxMap(f_henon, domain, unit_pts, f_device=f_device, backend="gpu")


def run(steps: int = 14, use_gpu: bool = True, show: bool = True):
    """
    Compute and plot the Hénon invariant measure.

    Parameters
    ----------
    steps : int
        Subdivision steps for the attractor computation.  Default: 14.
    use_gpu : bool
        Use GPU if available.  Default: True.
    show : bool
        Call ``plt.show()``.  Default: True.
    """
    import matplotlib.pyplot as plt

    # ── Domain ────────────────────────────────────────────────────────────────
    center = np.array([0.0, 0.0])
    radius = np.array([3.0, 3.0])
    domain = Box(center, radius)
    P = BoxPartition(domain, [1, 1])   # relative_attractor subdivides this

    # ── Box map ───────────────────────────────────────────────────────────────
    t = np.linspace(-0.9, 0.9, 4)
    gx, gy = np.meshgrid(t, t)
    unit_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

    gpu_used = False
    if use_gpu and cuda_available():
        try:
            F = _make_gpu_map(domain, unit_pts)
            gpu_used = True
        except Exception as e:
            print(f"[inv_measure_2d] GPU failed ({e}), using GridMap.")
            F = GridMap(f_henon, domain, unit_pts)
    else:
        F = GridMap(f_henon, domain, unit_pts)

    print(f"[inv_measure_2d] GPU={gpu_used}, computing attractor ({steps} steps) …")

    # ── Attractor ─────────────────────────────────────────────────────────────
    S = BoxSet.full(P)
    A = relative_attractor(F, S, steps=steps)
    print(f"[inv_measure_2d] Attractor: {len(A)} cells")

    # ── Transfer operator on attractor ────────────────────────────────────────
    print("[inv_measure_2d] Building TransferOperator …")
    T = TransferOperator(F, A, A)
    print(f"[inv_measure_2d] {T}")

    print("[inv_measure_2d] Computing leading eigenvector …")
    eigenvalues, eigenmeasures = T.eigs(k=1)
    print(f"[inv_measure_2d] Leading eigenvalue: {eigenvalues[0]:.6f}")

    mu_inv = eigenmeasures[0]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: attractor box set
    from gaio.viz import plot_boxset
    plot_boxset(A, ax=axes[0], color="steelblue", alpha=0.8,
                title=f"Hénon attractor  ({len(A)} cells)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

    # Right: invariant measure (absolute value; eigenvector sign is arbitrary)
    plot_boxmeasure(
        mu_inv, ax=axes[1], colormap="hot", absolute_value=True,
        title=f"SRB invariant measure  (λ₁ ≈ {abs(eigenvalues[0]):.4f})",
        colorbar=True,
    )
    axes[1].set_xlabel("x")

    plt.suptitle(f"Hénon invariant measure  (GPU={gpu_used})")
    plt.tight_layout()

    if show:
        plt.show()
    return mu_inv, fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hénon invariant measure example")
    parser.add_argument("--steps", type=int, default=14,
                        help="Attractor subdivision steps (default: 14)")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false")
    args = parser.parse_args()
    run(steps=args.steps, use_gpu=args.gpu)
