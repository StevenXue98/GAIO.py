"""
gaio/algorithms/attractor.py
==============================
relative_attractor — outer approximation of the maximal backward-invariant
set (ω-limit set) via iterated subdivision and intersection.

Correspondence with GAIO.jl
----------------------------
``relative_attractor`` ↔ ``ω`` / ``relative_attractor`` in
``src/algorithms/invariant_sets.jl`` with ``subdivision=true``.

    const relative_attractor = ω

    function ω(F, B::BoxSet; subdivision=true, steps=12)
        iter = (S -> S ∩ F(S)) ∘ subdivide
        return iterate_until_equal(iter, B; max_iterations=steps)
    end

Algorithm
---------
Each iteration:

1. Subdivide B along dimension ``argmin(dims)`` — matches Julia's
   ``subdivide(B)`` which calls ``subdivide(B, argmin(B.partition.dims))``.
2. Restrict to cells whose forward image re-enters B:  B ← B ∩ F(B).

Since the partition changes at every step (dims grow), the Julia
``iterate_until_equal`` fixed-point check never triggers when
``subdivision=true``; all ``steps`` iterations are always performed.
"""
from __future__ import annotations

import numpy as np

from gaio.core.boxset import BoxSet
from gaio.maps.base import SampledBoxMap


def relative_attractor(
    F: SampledBoxMap,
    B0: BoxSet,
    steps: int = 12,
) -> BoxSet:
    """
    Compute an outer approximation of the relative attractor (ω-limit set)
    of *F* within *B0*.

    Parameters
    ----------
    F : SampledBoxMap
        The box map to iterate.
    B0 : BoxSet
        Initial covering set (typically ``BoxSet.full(partition)`` for a
        coarse 1×1…×1 partition covering the domain of interest).
    steps : int, optional
        Number of subdivision–intersection iterations.  Default: 12.
        Matches GAIO.jl ``ω(...; subdivision=true, steps=12)``.

    Returns
    -------
    BoxSet
        Outer approximation on the refined partition after ``steps``
        subdivision steps.  The result partition has each dimension
        doubled ``ceil(steps / ndim)`` times (rounded, with tie-breaking
        going to the smallest dimension first).

    Notes
    -----
    The partition is refined at each step along ``argmin(dims)`` —
    the dimension with the fewest cells — exactly matching Julia's
    ``subdivide(B)`` default.  No early-exit is performed; all ``steps``
    iterations always execute.

    Examples
    --------
    >>> import numpy as np
    >>> from gaio.core.box import Box
    >>> from gaio.core.partition import BoxPartition
    >>> from gaio.core.boxset import BoxSet
    >>> from gaio.maps.base import SampledBoxMap
    >>> domain = Box([0.0, 0.0], [1.0, 1.0])
    >>> f = lambda x: np.array([x[0], x[1] * 0.5])
    >>> pts = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    >>> g = SampledBoxMap(f, domain, pts)
    >>> P = BoxPartition(domain, [1, 1])
    >>> rga = relative_attractor(g, BoxSet.full(P), steps=4)
    >>> rga.partition.dims.tolist()
    [4, 4]
    """
    B = B0
    for _ in range(steps):
        dim = int(np.argmin(B.partition.dims))
        B = B.subdivide(dim)
        B = B & F(B)
    return B
