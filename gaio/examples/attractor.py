"""
gaio/examples/attractor.py
==========================
Relative attractor of the Hénon map.

Port of: references/GAIO.jl-master/examples/attractor.jl

Algorithm
---------
``relative_attractor(F, S, steps=16)`` repeatedly refines ``S`` by:
    1. Subdivide each active cell.
    2. Keep only cells whose image intersects the current set.

After 16 steps the box covering has converged to the Hénon attractor.

GPU acceleration
----------------
When a CUDA device is available, ``AcceleratedBoxMap`` with a
``@numba.cuda.jit(device=True)`` device function is used instead of the
pure-Python ``GridMap``.  The GPU path evaluates all test points in
parallel on the VRAM.

Usage
-----
    python -m gaio.examples.attractor
    python -m gaio.examples.attractor --steps 20 --no-gpu
"""
from __future__ import annotations

import argparse
import numpy as np

import gaio
from gaio import Box, BoxPartition, BoxSet, GridMap, relative_attractor
from gaio import cuda_available, AcceleratedBoxMap


# ── Hénon map parameters ──────────────────────────────────────────────────────
A, B = 1.4, 0.3


def f_henon(x: np.ndarray) -> np.ndarray:
    """Pure-Python Hénon map: (x,y) → (1 - a·x² + y, b·x)."""
    return np.array([1.0 - A * x[0] ** 2 + x[1], B * x[0]])


def _make_gpu_map(domain: Box, unit_pts: np.ndarray, threads: int = 256):
    """
    Build an AcceleratedBoxMap using a CUDA device function for the Hénon map.

    The device function follows the **output-parameter pattern** required by
    Numba CUDA: ``f_device(x, out)`` writes the result into ``out`` in-place
    rather than returning a new array (Numba cannot return heap-allocated
    arrays from device functions).
    """
    from numba import cuda

    @cuda.jit(device=True)
    def f_device_henon(x, out):
        out[0] = 1.0 - A * x[0] * x[0] + x[1]
        out[1] = B * x[0]

    return AcceleratedBoxMap(
        f_henon, domain, unit_pts,
        f_device=f_device_henon,
        backend="gpu",
        threads_per_block=threads,
    )


def run(steps: int = 16, use_gpu: bool = True, show: bool = True):
    """
    Compute and plot the Hénon attractor.

    Parameters
    ----------
    steps : int
        Number of subdivision steps.  Default: 16.
    use_gpu : bool
        Use GPU acceleration if available.  Default: True.
    show : bool
        Call ``plt.show()`` at the end.  Default: True.
    """
    # ── Domain and partition ──────────────────────────────────────────────────
    # Start from a 1×1 coarse grid — relative_attractor subdivides it
    # `steps` times, so after 16 steps the resolution is [256, 256].
    # Starting from [128, 128] would make cells sub-pixel after subdivision.
    center = np.array([0.0, 0.0])
    radius = np.array([3.0, 3.0])
    domain = Box(center, radius)
    P = BoxPartition(domain, [1, 1])

    # ── Build box map ─────────────────────────────────────────────────────────
    # Unit test points: 4×4 Cartesian grid on [-1,1]²
    t = np.linspace(-0.9, 0.9, 4)
    gx, gy = np.meshgrid(t, t)
    unit_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (16, 2)

    gpu_used = False
    if use_gpu and cuda_available():
        try:
            F = _make_gpu_map(domain, unit_pts)
            gpu_used = True
            print(f"[attractor] Using GPU backend: {F}")
        except Exception as e:
            print(f"[attractor] GPU init failed ({e}), falling back to GridMap.")
            F = GridMap(f_henon, domain, unit_pts)
    else:
        F = GridMap(f_henon, domain, unit_pts)
        print(f"[attractor] Using Python backend: {F}")

    # ── Compute relative attractor ────────────────────────────────────────────
    S = BoxSet.full(P)
    print(f"[attractor] Starting with {len(S)} cells, running {steps} steps …")
    A_set = relative_attractor(F, S, steps=steps)
    print(f"[attractor] Attractor: {len(A_set)} cells  (GPU={gpu_used})")

    # ── Plot ──────────────────────────────────────────────────────────────────
    from gaio.viz import plot_boxset
    ax = plot_boxset(
        A_set,
        color="steelblue",
        alpha=0.9,
        title=f"Hénon attractor  ({len(A_set)} cells, {steps} steps)",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if show:
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.show()
    return A_set, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hénon attractor example")
    parser.add_argument("--steps", type=int, default=16,
                        help="Number of subdivision steps (default: 16)")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false",
                        help="Disable GPU acceleration")
    args = parser.parse_args()
    run(steps=args.steps, use_gpu=args.gpu)
