"""
gaio/algorithms/morse.py
========================
Morse decomposition — partition the chain-recurrent set into Morse sets
(non-trivial strongly-connected components of the transfer graph).

Correspondence with GAIO.jl
----------------------------
``morse_sets``  ↔ ``morse_sets(F, B)``  in ``src/algorithms/morse_graph.jl``
``morse_tiles`` ↔ ``morse_tiles(F, B)`` in ``src/algorithms/morse_graph.jl``

Julia uses MatrixNetworks.jl for SCC.  Python uses
``scipy.sparse.csgraph.connected_components``.

Algorithm
---------
1. Build ``TransferOperator(F, B, B)`` → sparse ``n×n`` matrix.
2. Compute SCCs of the directed graph (adjacency = ``T.mat.T``).
3. A component is **non-trivial** (a Morse set) if:
   * it contains more than one cell, OR
   * it contains a cell that maps to itself (self-loop in the transfer graph).
4. ``morse_sets``  returns the union of all Morse sets as a ``BoxSet``.
5. ``morse_tiles`` returns a ``BoxMeasure`` mapping each cell to its
   1-indexed Morse component number (0 = trivial, not returned).

``recurrent_set`` iterates ``morse_sets`` with subdivision, matching
Julia's ``recurrent_set``.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse.csgraph as csgraph

from gaio.core.box import I64
from gaio.core.boxset import BoxSet
from gaio.core.boxmeasure import BoxMeasure
from gaio.maps.base import SampledBoxMap
from gaio.transfer.operator import TransferOperator


def _scc_labeling(T: TransferOperator):
    """
    Return (labels, sizes, has_self_loop) for the transfer graph of *T*.

    ``labels[i]`` = SCC index of the i-th cell in ``T.domain``.
    """
    adj = T.mat.T.tocsr()   # A[i,j] > 0 ⟹ box i maps to box j
    n_comps, labels = csgraph.connected_components(
        adj, directed=True, connection="strong"
    )
    sizes = np.bincount(labels, minlength=n_comps)
    diag = np.asarray(adj.diagonal()).ravel()
    return labels, sizes, diag > 0.0


def morse_sets(F: SampledBoxMap, B: BoxSet) -> BoxSet:
    """
    Return the union of all Morse sets (non-trivial SCCs) of *F* on *B*.

    Parameters
    ----------
    F : SampledBoxMap
    B : BoxSet
        Domain and codomain for the transfer operator.

    Returns
    -------
    BoxSet
        Union of all non-trivial SCCs, on the same partition as *B*.

    Examples
    --------
    >>> import numpy as np
    >>> from gaio.core.box import Box
    >>> from gaio.core.partition import BoxPartition
    >>> from gaio.core.boxset import BoxSet
    >>> from gaio.maps.base import SampledBoxMap
    >>> from gaio.algorithms.morse import morse_sets
    >>> domain = Box([0.0, 0.0], [1.0, 1.0])
    >>> f = lambda x: x          # identity: every cell is its own SCC with self-loop
    >>> pts = np.array([[0., 0.]])
    >>> F = SampledBoxMap(f, domain, pts)
    >>> P = BoxPartition(domain, [4, 4])
    >>> B = BoxSet.full(P)
    >>> ms = morse_sets(F, B)
    >>> isinstance(ms, BoxSet)
    True
    """
    T = TransferOperator(F, B, B)
    labels, sizes, has_self_loop = _scc_labeling(T)
    n = len(T.domain._keys)
    n_comps = len(sizes)

    active_comps: set[int] = set()
    for comp in range(n_comps):
        mask = labels == comp
        if sizes[comp] > 1 or np.any(has_self_loop[mask]):
            active_comps.add(comp)

    active_mask = np.array([labels[i] in active_comps for i in range(n)], dtype=bool)
    return BoxSet(B.partition, T.domain._keys[active_mask])


def morse_tiles(F: SampledBoxMap, B: BoxSet) -> BoxMeasure:
    """
    Return a :class:`BoxMeasure` mapping each cell in a Morse set to its
    1-indexed Morse component number.

    Parameters
    ----------
    F : SampledBoxMap
    B : BoxSet

    Returns
    -------
    BoxMeasure
        Keys = all cells in Morse sets.
        Weights = integer component labels (1, 2, 3, …).
        Trivial cells (not in any Morse set) are excluded.

    Examples
    --------
    >>> import numpy as np
    >>> from gaio.core.box import Box
    >>> from gaio.core.partition import BoxPartition
    >>> from gaio.core.boxset import BoxSet
    >>> from gaio.maps.base import SampledBoxMap
    >>> from gaio.algorithms.morse import morse_tiles
    >>> domain = Box([0.0, 0.0], [1.0, 1.0])
    >>> f = lambda x: x
    >>> pts = np.array([[0., 0.]])
    >>> F = SampledBoxMap(f, domain, pts)
    >>> P = BoxPartition(domain, [4, 4])
    >>> B = BoxSet.full(P)
    >>> mt = morse_tiles(F, B)
    >>> isinstance(mt, BoxMeasure)
    True
    """
    T = TransferOperator(F, B, B)
    labels, sizes, has_self_loop = _scc_labeling(T)
    n = len(T.domain._keys)
    n_comps = len(sizes)

    # Assign 1-indexed Morse component numbers (0 = trivial)
    morse_idx = np.zeros(n_comps, dtype=np.int64)
    count = 0
    for comp in range(n_comps):
        mask = labels == comp
        if sizes[comp] > 1 or np.any(has_self_loop[mask]):
            count += 1
            morse_idx[comp] = count

    cell_morse = morse_idx[labels]   # shape (n,)
    active_mask = cell_morse > 0
    active_keys = T.domain._keys[active_mask]
    active_vals = cell_morse[active_mask].astype(np.float64)

    return BoxMeasure(B.partition, active_keys, active_vals)


def recurrent_set(
    F: SampledBoxMap,
    B0: BoxSet,
    steps: int = 12,
) -> BoxSet:
    """
    Compute the (chain) recurrent set of *F* within *B0* via iterated
    subdivision and Morse-set extraction.

    Each iteration:

    1. Subdivide B along ``argmin(dims)``.
    2. B ← morse_sets(F, B).

    Parameters
    ----------
    F : SampledBoxMap
    B0 : BoxSet
    steps : int, optional
        Default: 12.

    Returns
    -------
    BoxSet

    Notes
    -----
    Matches Julia's ``recurrent_set(F, B; subdivision=true, steps=12)``.
    """
    B = B0
    for _ in range(steps):
        if B.is_empty():
            break
        dim = int(np.argmin(B.partition.dims))
        B = B.subdivide(dim)
        B = morse_sets(F, B)
    return B
