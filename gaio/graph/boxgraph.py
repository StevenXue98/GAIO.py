"""
gaio/graph/boxgraph.py
======================
BoxGraph — directed graph representation of a TransferOperator, using
scipy.sparse.csgraph for strongly-connected-component (SCC) analysis.

Correspondence with GAIO.jl
----------------------------
``BoxGraph``                  ↔  ``BoxGraph`` in ``src/boxgraph.jl``
``union_strongly_connected``  ↔  ``union_strongly_connected_components``

Julia uses Graphs.jl; Python uses ``scipy.sparse.csgraph`` which is
already available in the scientific-Python stack and requires no extra
dependencies.

Graph convention
----------------
An edge  i → j  exists when box ``domain[i]`` has a non-zero transfer
weight to box ``domain[j]``.  Equivalently, edge i → j iff
``T.mat[j, i] > 0``.  The adjacency matrix is therefore ``T.mat.T``.

For the SCC a non-trivial component is one with:
* size > 1  (multiple cells strongly connected), OR
* size == 1 but with a self-loop (the cell maps back to itself).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph as csgraph
from numpy.typing import NDArray

from gaio.core.box import I64
from gaio.core.boxset import BoxSet
from gaio.transfer.operator import TransferOperator


class BoxGraph:
    """
    Directed graph representation of a :class:`TransferOperator`.

    Parameters
    ----------
    T : TransferOperator
        Must have ``domain == codomain`` (square matrix).

    Examples
    --------
    >>> import numpy as np
    >>> from gaio.core.box import Box
    >>> from gaio.core.partition import BoxPartition
    >>> from gaio.core.boxset import BoxSet
    >>> from gaio.maps.base import SampledBoxMap
    >>> from gaio.transfer.operator import TransferOperator
    >>> from gaio.graph.boxgraph import BoxGraph
    >>> domain = Box([0.0, 0.0], [1.0, 1.0])
    >>> f = lambda x: x * 0.5  # contraction to origin
    >>> pts = np.array([[0., 0.]])
    >>> F = SampledBoxMap(f, domain, pts)
    >>> P = BoxPartition(domain, [4, 4])
    >>> B = BoxSet.full(P)
    >>> T = TransferOperator(F, B, B)
    >>> G = BoxGraph(T)
    >>> isinstance(G.union_strongly_connected_components(), BoxSet)
    True
    """

    def __init__(self, T: TransferOperator) -> None:
        if T.domain.partition != T.codomain.partition:
            raise ValueError(
                "BoxGraph requires domain and codomain on the same partition."
            )
        self.T = T

    @classmethod
    def from_boxmap(cls, F, boxset: BoxSet) -> BoxGraph:
        """Convenience: build from a BoxMap and a BoxSet directly."""
        T = TransferOperator(F, boxset, boxset)
        return cls(T)

    # ------------------------------------------------------------------
    # Adjacency
    # ------------------------------------------------------------------

    def adjacency_matrix(self) -> scipy.sparse.csr_matrix:
        """
        Return the ``n × n`` adjacency matrix where ``A[i, j] > 0`` means
        there is an edge from box ``i`` to box ``j``.

        Derived from ``T.mat.T`` (rows = source, columns = target).
        """
        return self.T.mat.T.tocsr()

    # ------------------------------------------------------------------
    # SCC analysis
    # ------------------------------------------------------------------

    def strongly_connected_components(self) -> list[list[int]]:
        """
        Return a list of SCCs, each as a list of *local* node indices
        (positions in ``T.domain._keys``).
        """
        adj = self.adjacency_matrix()
        n_comps, labels = csgraph.connected_components(
            adj, directed=True, connection="strong"
        )
        components: list[list[int]] = [[] for _ in range(n_comps)]
        for node, comp in enumerate(labels):
            components[comp].append(node)
        return components

    def union_strongly_connected_components(self) -> BoxSet:
        """
        Return a :class:`BoxSet` of all cells that belong to a
        **non-trivial** strongly-connected component.

        A component is non-trivial if it has size > 1 OR contains a
        cell that maps to itself (self-loop in the transfer graph).

        Matches Julia's ``union_strongly_connected_components(g::BoxGraph)``.
        """
        adj = self.adjacency_matrix()
        n = adj.shape[0]
        n_comps, labels = csgraph.connected_components(
            adj, directed=True, connection="strong"
        )

        sizes = np.bincount(labels, minlength=n_comps)
        diag = np.asarray(adj.diagonal()).ravel()

        active_comps: set[int] = set()
        for comp in range(n_comps):
            mask = labels == comp
            if sizes[comp] > 1 or np.any(diag[mask] > 0.0):
                active_comps.add(comp)

        active_mask = np.array(
            [labels[i] in active_comps for i in range(n)], dtype=bool
        )
        result_keys = self.T.domain._keys[active_mask]
        return BoxSet(self.T.domain.partition, result_keys)
