"""
gaio/algorithms/invariant_sets.py
==================================
Additional invariant-set algorithms based on the preimage operator.

Correspondence with GAIO.jl
----------------------------
``preimage``              ↔ ``preimage(F, B, Q)``         in ``invariant_sets.jl``
``alpha_limit_set``       ↔ ``α``                          in ``invariant_sets.jl``
``maximal_invariant_set`` ↔ ``maximal_invariant_set``       in ``invariant_sets.jl``

Julia also aliases:
    ``maximal_forward_invariant_set  = α``
    ``maximal_backward_invariant_set = ω  = relative_attractor``

Algorithm summary
-----------------
preimage(F, B, Q):
    F⁻¹(B) ∩ Q — cells in Q whose forward image intersects B.
    Built by constructing TransferOperator(F, Q, B) and finding columns
    with at least one non-zero entry (i.e. columns that map into B).

alpha_limit_set(F, B0, steps):
    Iterated subdivision + preimage intersection (dual of relative_attractor):
        for _ in range(steps):
            B = subdivide(B, argmin(dims))
            B = B ∩ F⁻¹(B) = B ∩ preimage(F, B, B)

maximal_invariant_set(F, B0, steps):
    Iterated subdivision + forward/backward intersection:
        for _ in range(steps):
            B = subdivide(B, argmin(dims))
            B = B ∩ F(B) ∩ preimage(F, B, B)
"""
from __future__ import annotations

import numpy as np

from gaio.core.box import I64
from gaio.core.boxset import BoxSet
from gaio.maps.base import SampledBoxMap
from gaio.transfer.operator import TransferOperator


def preimage(F: SampledBoxMap, B: BoxSet, Q: BoxSet) -> BoxSet:
    """
    Compute the restricted preimage  F⁻¹(B) ∩ Q.

    Parameters
    ----------
    F : SampledBoxMap
        The box map.
    B : BoxSet
        Target set — cells whose preimage we seek.
    Q : BoxSet
        Search domain — restrict the preimage to this set.

    Returns
    -------
    BoxSet
        All cells in Q that have at least one test point landing in B
        under F.  The result is on the same partition as Q.

    Notes
    -----
    Builds ``TransferOperator(F, Q, B)`` (shape ``|B|×|Q|``) and returns
    all domain cells (columns) with at least one non-zero entry.

    Examples
    --------
    >>> import numpy as np
    >>> from gaio.core.box import Box
    >>> from gaio.core.partition import BoxPartition
    >>> from gaio.core.boxset import BoxSet
    >>> from gaio.maps.base import SampledBoxMap
    >>> from gaio.algorithms.invariant_sets import preimage
    >>> domain = Box([0.0, 0.0], [1.0, 1.0])
    >>> f = lambda x: x        # identity
    >>> pts = np.array([[0., 0.]])
    >>> F = SampledBoxMap(f, domain, pts)
    >>> P = BoxPartition(domain, [4, 4])
    >>> B = BoxSet.full(P)
    >>> result = preimage(F, B, B)
    >>> len(result) == len(B)   # every cell maps to itself
    True
    """
    T = TransferOperator(F, Q, B)
    # T.mat has shape (|B|, |Q|); nonzero columns = cells in Q that map into B
    col_sums = np.asarray(T.mat.sum(axis=0)).ravel()
    active = col_sums > 0.0
    result_keys = T.domain._keys[active]
    return BoxSet(Q.partition, result_keys)


def alpha_limit_set(
    F: SampledBoxMap,
    B0: BoxSet,
    steps: int = 12,
) -> BoxSet:
    """
    Compute an outer approximation of the α-limit set (maximal forward-
    invariant set) of *F* within *B0*.

    Each iteration:

    1. Subdivide B along ``argmin(dims)`` (same rule as ``relative_attractor``).
    2. Restrict to cells whose preimage re-enters B:  B ← B ∩ F⁻¹(B).

    Parameters
    ----------
    F : SampledBoxMap
    B0 : BoxSet
        Initial covering (typically ``BoxSet.full(coarse_partition)``).
    steps : int, optional
        Number of subdivision–intersection iterations.  Default: 12.

    Returns
    -------
    BoxSet
        Outer approximation on the refined partition.

    Notes
    -----
    Matches Julia's ``α(F, B; subdivision=true, steps=12)`` which is
    aliased as ``maximal_forward_invariant_set``.

    Examples
    --------
    >>> import numpy as np
    >>> from gaio.core.box import Box
    >>> from gaio.core.partition import BoxPartition
    >>> from gaio.core.boxset import BoxSet
    >>> from gaio.maps.base import SampledBoxMap
    >>> from gaio.algorithms.invariant_sets import alpha_limit_set
    >>> domain = Box([0.0, 0.0], [1.0, 1.0])
    >>> f = lambda x: np.array([x[0] * 0.5, x[1]])   # contracts x[0]
    >>> pts = np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
    >>> F = SampledBoxMap(f, domain, pts)
    >>> P = BoxPartition(domain, [1, 1])
    >>> alpha = alpha_limit_set(F, BoxSet.full(P), steps=4)
    >>> alpha.partition.dims.tolist()
    [4, 4]
    """
    B = B0
    for _ in range(steps):
        dim = int(np.argmin(B.partition.dims))
        B = B.subdivide(dim)
        B = B & preimage(F, B, B)
    return B


def maximal_invariant_set(
    F: SampledBoxMap,
    B0: BoxSet,
    steps: int = 12,
) -> BoxSet:
    """
    Compute an outer approximation of the maximal invariant set of *F*
    within *B0* (cells that are both forward- and backward-invariant).

    Each iteration:

    1. Subdivide B along ``argmin(dims)``.
    2. B ← B ∩ F(B) ∩ F⁻¹(B).

    Parameters
    ----------
    F : SampledBoxMap
    B0 : BoxSet
    steps : int, optional
        Default: 12.

    Returns
    -------
    BoxSet
        Outer approximation of the maximal invariant set.

    Notes
    -----
    Matches Julia's ``maximal_invariant_set(F, B; subdivision=true, steps=12)``.

    Examples
    --------
    >>> import numpy as np
    >>> from gaio.core.box import Box
    >>> from gaio.core.partition import BoxPartition
    >>> from gaio.core.boxset import BoxSet
    >>> from gaio.maps.base import SampledBoxMap
    >>> from gaio.algorithms.invariant_sets import maximal_invariant_set
    >>> domain = Box([0.0, 0.0], [1.0, 1.0])
    >>> f = lambda x: x               # identity: everything is invariant
    >>> pts = np.array([[0., 0.]])
    >>> F = SampledBoxMap(f, domain, pts)
    >>> P = BoxPartition(domain, [1, 1])
    >>> mis = maximal_invariant_set(F, BoxSet.full(P), steps=2)
    >>> mis.partition.dims.tolist()
    [4, 1]
    """
    B = B0
    for _ in range(steps):
        dim = int(np.argmin(B.partition.dims))
        B = B.subdivide(dim)
        # G(B) = F(B) ∩ preimage(F, B, B): both forward- and backward-invariant
        B = B & F(B) & preimage(F, B, B)
    return B
