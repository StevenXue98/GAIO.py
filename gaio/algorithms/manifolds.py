"""
gaio/algorithms/manifolds.py
==============================
unstable_set — BFS flood-fill of the forward image of a seed set.

Correspondence with GAIO.jl
----------------------------
``unstable_set`` ↔ ``unstable_set`` in
``src/algorithms/invariant_sets.jl``.

    function unstable_set(F::BoxMap, B::BoxSet)
        B₀ = copy(B)
        B₁ = copy(B)
        while !isempty(B₁)
            B₁ = F(B₁)
            setdiff!(B₁, B₀)
            union!(B₀, B₁)
        end
        return B₀
    end

Algorithm
---------
Starting from a seed ``B0`` (typically a small neighbourhood of a fixed
point on a sufficiently fine partition):

1. ``accumulated = B0``           (all cells seen so far)
2. ``frontier    = B0``           (cells to map forward next)
3. Loop:
   a. ``frontier = F(frontier)``              forward image of frontier
   b. ``frontier = frontier - accumulated``   keep only NEW cells
   c. ``accumulated = accumulated | frontier`` add them to the result
4. Return ``accumulated`` when ``frontier`` is empty.

At termination ``F(accumulated) ⊆ accumulated`` (forward invariance
within the domain), so the result is the smallest forward-invariant set
containing ``B0`` that is representable on the given partition.

Notes
-----
The partition is **never changed**.  For a meaningful result, seed the
algorithm with a fine enough partition.  Using ``BoxSet.full(partition)``
as the seed returns the full partition immediately (trivially forward-
invariant), which is what GAIO.jl's ``test/algorithms.jl`` does.
"""
from __future__ import annotations

from gaio.core.boxset import BoxSet
from gaio.maps.base import SampledBoxMap


def unstable_set(F: SampledBoxMap, B0: BoxSet) -> BoxSet:
    """
    Compute the unstable set of *F* starting from seed *B0*.

    Parameters
    ----------
    F : SampledBoxMap
        The box map to iterate forward.
    B0 : BoxSet
        Seed set.  For a meaningful dynamical result, this should be a
        small covering of a fixed point or unstable equilibrium on a
        sufficiently fine partition.  Using ``BoxSet.full(partition)``
        returns ``BoxSet.full(partition)`` instantly.

    Returns
    -------
    BoxSet
        All cells reachable from *B0* under repeated forward iteration
        of *F* that remain within the domain.  The result is on the same
        partition as *B0*.

    Notes
    -----
    Forward invariance: after this function returns ``W``,
    ``F(W) ⊆ W`` holds (no cells outside ``W`` are mapped to from
    inside ``W``, within the domain).

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
    >>> P = BoxPartition(domain, [4, 4])
    >>> seed = BoxSet.cover(P, np.array([[0.0, 0.0]]))
    >>> W = unstable_set(g, seed)
    >>> isinstance(W, BoxSet)
    True
    >>> seed <= W
    True
    """
    accumulated = B0
    frontier = B0
    while not frontier.is_empty():
        frontier = F(frontier) - accumulated
        accumulated = accumulated | frontier
    return accumulated
