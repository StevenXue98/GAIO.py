"""
gaio/maps/grid_map.py
=====================
GridMap — SampledBoxMap with a uniform Cartesian grid of test points.

Grid layout
-----------
For ``n_points = (p₀, p₁, …, pₙ₋₁)``, the grid in the unit cube is::

    u[i][k] = -1 + k * (2 / p_i),   k = 0, 1, …, p_i - 1

This matches GAIO.jl's ``GridBoxMap`` exactly:
``Δp = 2 ./ n_points; points[i] = Δp .* (i.I .- 1) .- 1``

For the default ``n_points = 4`` in 2-D::

    u = [-1.0, -0.5, 0.0, 0.5]  per axis

giving 4² = 16 test points per cell.

Correspondence with GAIO.jl
----------------------------
``GridMap(f, domain, n_points)``
↔ ``GridBoxMap(f, domain; n_points=n_points)`` (boxmap_sampled.jl)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from gaio.core.box import Box, F64
from .base import SampledBoxMap


def GridMap(
    f,
    domain: Box,
    n_points: int | Sequence[int] | None = None,
) -> SampledBoxMap:
    """
    Create a :class:`SampledBoxMap` with a uniform grid of test points.

    Parameters
    ----------
    f : callable
        The map ``f(x) -> y``, both shape ``(n,)`` float64.
    domain : Box
        Spatial domain.
    n_points : int or sequence of int, optional
        Number of test points per dimension.  A single ``int`` is broadcast
        to all dimensions.  Default: 4 per dimension (matches GAIO.jl).

    Returns
    -------
    SampledBoxMap

    Examples
    --------
    >>> import numpy as np
    >>> from gaio.core.box import Box
    >>> domain = Box([0.0, 0.0], [1.0, 1.0])
    >>> g = GridMap(lambda x: x ** 2, domain)
    >>> g.n_test_points  # 4^2
    16
    >>> g2 = GridMap(lambda x: x ** 2, domain, n_points=(2, 3))
    >>> g2.n_test_points  # 2*3
    6
    """
    ndim = domain.ndim

    if n_points is None:
        n_pts: tuple[int, ...] = (4,) * ndim
    elif isinstance(n_points, int):
        n_pts = (n_points,) * ndim
    else:
        n_pts = tuple(int(k) for k in n_points)

    if len(n_pts) != ndim:
        raise ValueError(
            f"n_points has length {len(n_pts)} but domain has {ndim} dimensions."
        )
    if any(p < 1 for p in n_pts):
        raise ValueError("All n_points values must be >= 1.")

    # GAIO.jl: Δp = 2 / n_pts;  points[i] = Δp * (i - 1) - 1
    # => u[k] = -1 + k * (2 / n_pts),  k = 0, …, n_pts-1
    grids = [
        np.arange(n_pts[i], dtype=F64) * (2.0 / n_pts[i]) - 1.0
        for i in range(ndim)
    ]
    mesh = np.stack(
        [g.ravel() for g in np.meshgrid(*grids, indexing="ij")], axis=1
    ).astype(F64)  # (prod(n_pts), ndim)

    return SampledBoxMap(f, domain, mesh)
