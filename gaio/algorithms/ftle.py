"""
gaio/algorithms/ftle.py
=======================
Finite-Time Lyapunov Exponents (FTLE).

The FTLE at a point x measures the maximum exponential rate of separation
of nearby trajectories over a finite time interval [t0, t0+T]:

    σ(x) = (1/T) · log( σ_max(J) )

where J = DΦ^T(x) is the Jacobian of the flow map Φ^T at x and σ_max(J)
is its largest singular value (= √λ_max(JᵀJ)).

Algorithm
---------
For each cell in ``S`` with centre x:

1. Evaluate ``F._apply_map`` at x and at x + δ·eᵢ for each dimension i
   (forward finite-difference; (ndim+1)·K points in one batched call).
2. Form the (ndim × ndim) Jacobian J[k] from the finite-difference columns.
3. Compute the largest singular value via ``numpy.linalg.svd`` — batched
   over all K cells at once, no Python loop.
4. Return a ``BoxMeasure`` with FTLE values for each cell.

Correspondence with GAIO.jl
----------------------------
``finite_time_lyapunov_exponents(F, S; T)`` in
``src/algorithms/finite_time_lyapunov_exponents.jl``.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gaio.core.box import F64, I64
from gaio.core.boxset import BoxSet
from gaio.core.boxmeasure import BoxMeasure
from gaio.maps.base import SampledBoxMap


def finite_time_lyapunov_exponents(
    F: SampledBoxMap,
    S: BoxSet,
    T: float,
    delta: float | None = None,
) -> BoxMeasure:
    """
    Compute the finite-time Lyapunov exponent for each cell in ``S``.

    The FTLE field is returned as a ``BoxMeasure`` over ``S``.  Positive
    values indicate stretching (chaotic / hyperbolic regions); negative
    values indicate compression.

    Parameters
    ----------
    F : SampledBoxMap
        The box map representing the flow Φ^T.  For a nonautonomous system,
        build ``F`` with :func:`~gaio.maps.rk4.rk4_flow_map_tspan` using
        the desired integration interval.
    S : BoxSet
        The set of cells at which to evaluate the FTLE.
    T : float
        The integration time used to build ``F``.  Used only as the
        normalisation denominator  σ = log(σ_max) / T.
    delta : float, optional
        Finite-difference step size for Jacobian estimation.  Defaults to
        1 % of the smallest cell radius, which is accurate for smooth flows.
        Reduce for very small cells; increase if the map is noisy.

    Returns
    -------
    BoxMeasure
        FTLE values on the same partition as ``S``.  Keys = ``S._keys``.

    Notes
    -----
    The Jacobian estimation uses (ndim + 1) · K map evaluations in a single
    batched ``F._apply_map`` call — GPU-accelerated if ``F`` is an
    :class:`~gaio.cuda.AcceleratedBoxMap`.

    The batch SVD (``numpy.linalg.svd`` on an (K, ndim, ndim) array) is
    O(K · ndim³) with LAPACK's ``dgesdd`` driver.  For ndim ≤ 4 (the typical
    GAIO use case) this is essentially O(K).

    Examples
    --------
    >>> import numpy as np
    >>> from gaio import Box, BoxPartition, BoxSet, SampledBoxMap
    >>> from gaio.algorithms.ftle import finite_time_lyapunov_exponents
    >>> domain = Box([0.5], [0.5])
    >>> P = BoxPartition(domain, [8])
    >>> S = BoxSet.full(P)
    >>> f = lambda x: x * 2.0          # uniform stretching by 2
    >>> pts = np.array([[0.0]])
    >>> F = SampledBoxMap(f, domain, pts)
    >>> ftle = finite_time_lyapunov_exponents(F, S, T=1.0)
    >>> round(float(ftle._weights[0]), 4)   # log(2)/1 ≈ 0.6931
    0.6931
    """
    P = S.partition
    ndim = P.ndim
    centers = S.centers()   # (K, ndim)
    K = len(centers)

    if K == 0:
        return BoxMeasure(P, np.empty(0, dtype=I64), np.empty(0, dtype=F64))

    if delta is None:
        delta = float(np.min(P.cell_radius)) * 0.01

    # ── Build perturbed point array ────────────────────────────────────────
    # Layout: first K rows = unperturbed centres,
    #         next K rows  = centre + δ·e0,
    #         ...
    #         last K rows  = centre + δ·e_{ndim-1}
    pts = np.empty((K * (ndim + 1), ndim), dtype=F64)
    pts[:K] = centers
    for i in range(ndim):
        pts[(i + 1) * K:(i + 2) * K] = centers.copy()
        pts[(i + 1) * K:(i + 2) * K, i] += delta

    # ── Single batched map evaluation (GPU if available) ───────────────────
    mapped = F._apply_map(pts)          # (K*(ndim+1), ndim)

    f0 = mapped[:K]                     # (K, ndim) — images of centres

    # ── Vectorised Jacobian assembly ──────────────────────────────────────
    # J[k, :, i] = column i = (f(x+δeᵢ) - f(x)) / δ
    J = np.empty((K, ndim, ndim), dtype=F64)
    for i in range(ndim):
        fi = mapped[(i + 1) * K:(i + 2) * K]   # (K, ndim)
        J[:, :, i] = (fi - f0) / delta

    # ── Batch SVD → largest singular value per cell ───────────────────────
    # numpy.linalg.svd on (K, ndim, ndim) returns s of shape (K, ndim)
    # in descending order; s[:, 0] is σ_max for each cell.
    s = np.linalg.svd(J, compute_uv=False)     # (K, ndim)
    sigma_max = s[:, 0]                         # (K,)

    ftle = np.log(np.maximum(sigma_max, 1e-15)) / T
    return BoxMeasure(P, S._keys.copy(), ftle)
