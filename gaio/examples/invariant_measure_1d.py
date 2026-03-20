"""
gaio/examples/invariant_measure_1d.py
======================================
Invariant measure of the logistic map via the Perron-Frobenius operator.

Port of: references/GAIO.jl-master/examples/invariant_measure_1d.jl

Algorithm
---------
1. Discretise [0,1] into 256 equal cells.
2. Build a ``TransferOperator`` for the logistic map f(x) = 4x(1-x).
3. Compute the leading eigenvalue/eigenvector of the column-stochastic matrix.
4. The eigenvector with eigenvalue ≈ 1 is the invariant density.

The exact invariant density for μ=4 is  ρ(x) = 1 / (π √(x(1-x))),
which diverges at 0 and 1 — visible as the U-shaped distribution.

Usage
-----
    python -m gaio.examples.invariant_measure_1d
"""
from __future__ import annotations

import numpy as np

from gaio import Box, BoxPartition, BoxSet, GridMap, TransferOperator
from gaio.core.boxmeasure import BoxMeasure


MU = 4.0


def f_logistic(x: np.ndarray) -> np.ndarray:
    """Logistic map: x → μ·x·(1-x)  (1-D)."""
    return np.array([MU * x[0] * (1.0 - x[0])])


def run(n_cells: int = 256, n_test: int = 400, k_eigs: int = 3, show: bool = True):
    """
    Compute and plot the invariant measure of the logistic map.

    Parameters
    ----------
    n_cells : int
        Number of cells in the 1-D partition.  Default: 256.
    n_test : int
        Number of test points per cell (uniform grid).  Default: 400.
    k_eigs : int
        Number of eigenvalues to compute.  Default: 3.
    show : bool
        Call ``plt.show()``.  Default: True.
    """
    import matplotlib.pyplot as plt

    # ── Domain: [0, 1] as Box with center=0.5, radius=0.5 ────────────────────
    center = np.array([0.5])
    radius = np.array([0.5])
    domain = Box(center, radius)
    P = BoxPartition(domain, [n_cells])

    # ── Box map with uniform 1-D test points ──────────────────────────────────
    unit_pts = np.linspace(-0.99, 0.99, n_test).reshape(-1, 1)  # (n_test, 1)
    F = GridMap(f_logistic, domain, unit_pts)

    # ── Transfer operator ─────────────────────────────────────────────────────
    S = BoxSet.full(P)
    print(f"[inv_measure_1d] Building TransferOperator on {n_cells} cells …")
    T = TransferOperator(F, S, S)
    print(f"[inv_measure_1d] {T}")

    # ── Leading eigenvectors ──────────────────────────────────────────────────
    print(f"[inv_measure_1d] Computing {k_eigs} eigenmodes …")
    eigenvalues, eigenmeasures = T.eigs(k=k_eigs)

    # Sort by descending |λ|
    order = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[order]
    eigenmeasures = [eigenmeasures[i] for i in order]

    print(f"[inv_measure_1d] Top {k_eigs} eigenvalues: {np.abs(eigenvalues)}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    mu_inv = eigenmeasures[0]
    # Cell centers along [0,1]
    centers = P.key_to_box(int(mu_inv._keys[0])).center  # just to get dtype
    all_centers = np.array([P.key_to_box(int(k)).center[0] for k in mu_inv._keys])
    weights = np.abs(mu_inv._weights)  # take absolute value (eigenvector sign)
    weights /= weights.sum()           # normalise to unit mass

    # Sort by center for a clean line plot
    order_x = np.argsort(all_centers)
    x_sorted = all_centers[order_x]
    w_sorted = weights[order_x]

    # Exact invariant density for comparison
    x_exact = np.linspace(0.01, 0.99, 500)
    rho_exact = 1.0 / (np.pi * np.sqrt(x_exact * (1.0 - x_exact)))
    rho_exact /= np.trapz(rho_exact, x_exact)   # normalise

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x_sorted, w_sorted * n_cells, width=1.0 / n_cells,
           color="steelblue", alpha=0.7, label="GAIO eigenvector |ψ₁|")
    ax.plot(x_exact, rho_exact, "r-", lw=1.5,
            label=r"Exact: $\rho(x)=1/(\pi\sqrt{x(1-x)})$")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title(f"Invariant measure — logistic map (μ={MU}, {n_cells} cells)")
    ax.legend()
    ax.set_xlim(0, 1)

    if show:
        plt.tight_layout()
        plt.show()
    return eigenvalues, eigenmeasures, ax


if __name__ == "__main__":
    run()
