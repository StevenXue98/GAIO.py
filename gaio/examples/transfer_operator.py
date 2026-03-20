"""
gaio/examples/transfer_operator.py
=====================================
TransferOperator demo: push-forward and pull-back of box measures.

Port of: references/GAIO.jl-master/examples/transfer_operator.jl

Scenario
--------
Domain: [-1,1]² partitioned into a 16×8 grid.
Map:    horizontal translation  f(x,y) = (x+1, y)
        (shifts the left half into the right half)

Demonstrated operations
-----------------------
* Create ``BoxMeasure`` objects with uniform weights.
* Verify vector-space arithmetic: addition, subtraction, scalar mult.
* Build a ``TransferOperator`` and compute push-forward / pull-back.
* Plot before and after with matplotlib.

Usage
-----
    python -m gaio.examples.transfer_operator
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from gaio import Box, BoxPartition, BoxSet, SampledBoxMap, TransferOperator
from gaio.core.boxmeasure import BoxMeasure
from gaio.viz import plot_boxset, plot_boxmeasure


def f_shift(x: np.ndarray) -> np.ndarray:
    """Horizontal translation by 1: (x,y) → (x+1, y)."""
    return np.array([x[0] + 1.0, x[1]])


def run(show: bool = True):
    """
    Run the transfer operator demo.

    Parameters
    ----------
    show : bool
        Call ``plt.show()``.  Default: True.
    """
    # ── Domain and partition ──────────────────────────────────────────────────
    center = np.array([0.0, 0.0])
    radius = np.array([1.0, 1.0])
    domain = Box(center, radius)
    P = BoxPartition(domain, [16, 8])

    full = BoxSet.full(P)

    # Left half: x ∈ [-1, 0]  →  Box(center=(-0.5, 0), radius=(0.5, 1))
    left_center = np.array([-0.5, 0.0])
    left_radius = np.array([0.5, 1.0])
    left_box = Box(left_center, left_radius)
    left = BoxSet.from_box(P, left_box)

    # Right half: x ∈ [0, 1]
    right_center = np.array([0.5, 0.0])
    right_radius = np.array([0.5, 1.0])
    right_box = Box(right_center, right_radius)
    right = BoxSet.from_box(P, right_box)

    print(f"[transfer_operator] full={len(full)}, left={len(left)}, right={len(right)}")

    # ── Box measures with uniform weights ─────────────────────────────────────
    n_left  = len(left)
    n_right = len(right)
    mu_left  = BoxMeasure(P, left._keys,  np.ones(n_left,  dtype=np.float64))
    mu_right = BoxMeasure(P, right._keys, np.ones(n_right, dtype=np.float64))

    # ── Vector-space arithmetic checks ───────────────────────────────────────
    mu_sum = mu_left + mu_right
    print(f"[transfer_operator] |μ_left + μ_right| = {len(mu_sum)}  "
          f"(expected {len(full)})")

    mu_scaled = 2.0 * mu_left
    print(f"[transfer_operator] max(2·μ_left weights) = {mu_scaled._weights.max():.1f}  "
          "(expected 2.0)")

    # ── Box map: one centre sample point ──────────────────────────────────────
    unit_pts = np.array([[0.0, 0.0]])  # single centre point
    F = SampledBoxMap(f_shift, domain, unit_pts)

    # ── Transfer operator ─────────────────────────────────────────────────────
    print("[transfer_operator] Building TransferOperator …")
    T = TransferOperator(F, full, full)
    print(f"[transfer_operator] {T}")

    # Push-forward: left half → right half
    mu_pushed = T.push_forward(mu_left)
    print(f"[transfer_operator] |T·μ_left| = {len(mu_pushed)}  "
          "(should cover right half)")

    # Pull-back: right half → left half
    mu_pulled = T.pull_back(mu_right)
    print(f"[transfer_operator] |T'·μ_right| = {len(mu_pulled)}  "
          "(should cover left half)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    plot_boxset(left, ax=axes[0], color="royalblue", alpha=0.8,
                title="μ_left (initial)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

    plot_boxmeasure(mu_pushed, ax=axes[1], colormap="Oranges",
                    title="T · μ_left  (push-forward)", colorbar=True)
    axes[1].set_xlabel("x")

    plot_boxmeasure(mu_pulled, ax=axes[2], colormap="Greens",
                    title="T' · μ_right  (pull-back)", colorbar=True)
    axes[2].set_xlabel("x")

    plt.suptitle("TransferOperator: horizontal shift  f(x,y) = (x+1, y)")
    plt.tight_layout()

    if show:
        plt.show()
    return T, mu_pushed, mu_pulled, fig


if __name__ == "__main__":
    run()
