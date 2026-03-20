"""
gaio/maps/montecarlo_map.py
===========================
MonteCarloMap — SampledBoxMap with random Monte-Carlo test points.

Correspondence with GAIO.jl
----------------------------
``MonteCarloMap(f, domain, n_points=k)``
↔ ``MonteCarloBoxMap(f, domain; n_points=k)`` (boxmap_sampled.jl)

Julia default: ``n_points = 16 * N`` where N is the spatial dimension.
Points are drawn uniformly from ``[-1, 1]^N``.
"""
from __future__ import annotations

import numpy as np

from gaio.core.box import Box, F64
from .base import SampledBoxMap


def MonteCarloMap(
    f,
    domain: Box,
    n_points: int | None = None,
    *,
    seed: int | None = None,
) -> SampledBoxMap:
    """
    Create a :class:`SampledBoxMap` with random Monte-Carlo test points.

    Parameters
    ----------
    f : callable
        The map ``f(x) -> y``, both shape ``(n,)`` float64.
    domain : Box
        Spatial domain.
    n_points : int, optional
        Number of random test points.  Default: ``16 * domain.ndim``
        (matches GAIO.jl default).
    seed : int, optional
        Seed for reproducibility.  ``None`` gives different points each call.

    Returns
    -------
    SampledBoxMap

    Notes
    -----
    The random test points are fixed at construction time — all cells in
    a :meth:`~SampledBoxMap.map_boxes` call use the **same** set of unit
    points, rescaled per cell.  This differs from drawing fresh random
    points per cell per call.

    Examples
    --------
    >>> import numpy as np
    >>> from gaio.core.box import Box
    >>> domain = Box([0.0, 0.0], [1.0, 1.0])
    >>> g = MonteCarloMap(lambda x: x ** 2, domain, seed=42)
    >>> g.n_test_points  # 16 * 2
    32
    >>> g2 = MonteCarloMap(lambda x: x ** 2, domain, n_points=100, seed=0)
    >>> g2.n_test_points
    100
    """
    ndim = domain.ndim
    if n_points is None:
        n_points = 16 * ndim

    rng = np.random.default_rng(seed)
    unit_pts = rng.uniform(-1.0, 1.0, size=(n_points, ndim)).astype(F64)
    return SampledBoxMap(f, domain, unit_pts)
