"""
Tests for gaio/graph/boxgraph.py — BoxGraph and SCC analysis.
"""
import numpy as np
import pytest

from gaio.core.box import Box
from gaio.core.partition import BoxPartition
from gaio.core.boxset import BoxSet
from gaio.maps.base import SampledBoxMap
from gaio.transfer.operator import TransferOperator
from gaio.graph.boxgraph import BoxGraph


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_simple(f_fn, dims=(4, 4)):
    domain = Box([0.0, 0.0], [1.0, 1.0])
    pts = np.array([[0.0, 0.0]])
    F = SampledBoxMap(f_fn, domain, pts)
    P = BoxPartition(domain, list(dims))
    B = BoxSet.full(P)
    return F, P, B


# ===========================================================================
# A. Behavioural tests
# ===========================================================================

class TestBoxGraphConstruction:
    def test_from_transfer_operator(self):
        F, P, B = make_simple(lambda x: x)
        T = TransferOperator(F, B, B)
        G = BoxGraph(T)
        assert isinstance(G, BoxGraph)

    def test_from_boxmap(self):
        F, P, B = make_simple(lambda x: x)
        G = BoxGraph.from_boxmap(F, B)
        assert isinstance(G, BoxGraph)

    def test_adjacency_matrix_shape(self):
        F, P, B = make_simple(lambda x: x)
        T = TransferOperator(F, B, B)
        G = BoxGraph(T)
        adj = G.adjacency_matrix()
        assert adj.shape == (16, 16)

    def test_adjacency_is_square(self):
        F, P, B = make_simple(lambda x: x)
        T = TransferOperator(F, B, B)
        G = BoxGraph(T)
        adj = G.adjacency_matrix()
        assert adj.shape[0] == adj.shape[1]

    def test_mismatched_partition_raises(self):
        domain = Box([0.0, 0.0], [1.0, 1.0])
        F = SampledBoxMap(lambda x: x, domain, np.array([[0., 0.]]))
        P1 = BoxPartition(domain, [4, 4])
        P2 = BoxPartition(domain, [2, 2])
        B1 = BoxSet.full(P1)
        B2 = BoxSet.full(P2)
        T = TransferOperator(F, B1, B2)
        with pytest.raises(ValueError):
            BoxGraph(T)


class TestSCC:
    def test_identity_each_cell_is_scc(self):
        """Identity map: each cell maps to itself → n trivial self-loop SCCs."""
        F, P, B = make_simple(lambda x: x)
        T = TransferOperator(F, B, B)
        G = BoxGraph(T)
        sccs = G.strongly_connected_components()
        # Every component is size 1, so we have exactly n SCCs
        assert sum(len(c) for c in sccs) == len(B)

    def test_scc_count_equals_num_cells_for_identity(self):
        F, P, B = make_simple(lambda x: x)
        T = TransferOperator(F, B, B)
        G = BoxGraph(T)
        sccs = G.strongly_connected_components()
        assert len(sccs) == len(B)

    def test_scc_total_nodes_equals_n(self):
        F, P, B = make_simple(lambda x: x * 0.5)
        T = TransferOperator(F, B, B)
        G = BoxGraph(T)
        sccs = G.strongly_connected_components()
        assert sum(len(c) for c in sccs) == len(B)


class TestUnionSCC:
    def test_identity_all_cells_are_non_trivial_sccs(self):
        """Identity: every cell has a self-loop → all are non-trivial."""
        F, P, B = make_simple(lambda x: x)
        T = TransferOperator(F, B, B)
        G = BoxGraph(T)
        uscc = G.union_strongly_connected_components()
        assert isinstance(uscc, BoxSet)
        assert len(uscc) == len(B)

    def test_strong_contraction_no_cycles(self):
        """
        Strict contraction to origin: every cell maps uniquely toward the
        origin cell.  The chain has no cycles except possibly the origin.
        uscc should be empty OR just the origin cell.
        """
        domain = Box([0.0, 0.0], [1.0, 1.0])
        # Only map to the origin (0,0) point
        f = lambda x: np.zeros_like(x)
        pts = np.array([[0.0, 0.0]])
        F = SampledBoxMap(f, domain, pts)
        P = BoxPartition(domain, [4, 4])
        B = BoxSet.full(P)
        T = TransferOperator(F, B, B)
        G = BoxGraph(T)
        uscc = G.union_strongly_connected_components()
        # Origin cell self-maps → non-trivial; all others map away → trivial
        assert isinstance(uscc, BoxSet)
        # At most the origin cell
        assert len(uscc) <= 1

    def test_returns_boxset(self):
        F, P, B = make_simple(lambda x: x)
        T = TransferOperator(F, B, B)
        G = BoxGraph(T)
        result = G.union_strongly_connected_components()
        assert isinstance(result, BoxSet)
        assert result.partition == B.partition

    def test_result_is_subset_of_domain(self):
        F, P, B = make_simple(lambda x: x * 0.5)
        T = TransferOperator(F, B, B)
        G = BoxGraph(T)
        uscc = G.union_strongly_connected_components()
        # Result must be a subset of B
        assert uscc <= B
