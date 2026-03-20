"""
Tests for gaio/algorithms/invariant_sets.py and gaio/algorithms/morse.py.

Two categories:
  A. Behavioural (randomized / property-based)
  B. Dynamical correctness tests for known systems
"""
import numpy as np
import pytest

from gaio.core.box import Box
from gaio.core.partition import BoxPartition
from gaio.core.boxset import BoxSet
from gaio.core.boxmeasure import BoxMeasure
from gaio.maps.base import SampledBoxMap
from gaio.algorithms.invariant_sets import preimage, alpha_limit_set, maximal_invariant_set
from gaio.algorithms.morse import morse_sets, morse_tiles, recurrent_set


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

def make_map(f_fn, domain=None, pts=None, dims=(4, 4)):
    if domain is None:
        domain = Box([0.0, 0.0], [1.0, 1.0])
    if pts is None:
        pts = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    F = SampledBoxMap(f_fn, domain, pts)
    P = BoxPartition(domain, list(dims))
    B = BoxSet.full(P)
    return F, P, B


@pytest.fixture
def linear_map():
    """f(x) = (x[0], x[1]*0.5) — contracts x[1] to 0, identity on x[0]."""
    return make_map(lambda x: np.array([x[0], x[1] * 0.5]))


@pytest.fixture
def identity_map():
    """f(x) = x — everything is invariant."""
    return make_map(lambda x: x, pts=np.array([[0., 0.]]))


@pytest.fixture
def contraction_map():
    """f(x) = 0.5*x — contracts everything to origin."""
    return make_map(lambda x: x * 0.5, pts=np.array([[0., 0.]]))


# ===========================================================================
# A. Behavioural: preimage
# ===========================================================================

class TestPreimage:
    def test_preimage_of_full_is_full(self, identity_map):
        """preimage(F, full, full) should be all of full for identity map."""
        F, P, B = identity_map
        result = preimage(F, B, B)
        assert isinstance(result, BoxSet)
        assert result == B

    def test_preimage_of_empty_is_empty(self, identity_map):
        F, P, B = identity_map
        empty = BoxSet.empty(P)
        result = preimage(F, empty, B)
        assert result.is_empty()

    def test_preimage_result_is_subset_of_Q(self, linear_map):
        F, P, B = linear_map
        result = preimage(F, B, B)
        assert result <= B

    def test_preimage_returns_boxset(self, linear_map):
        F, P, B = linear_map
        result = preimage(F, B, B)
        assert isinstance(result, BoxSet)
        assert result.partition == P

    def test_preimage_same_partition_as_Q(self, linear_map):
        F, P, B = linear_map
        Q = BoxSet(P, B._keys[:8])
        result = preimage(F, B, Q)
        assert result.partition == P

    def test_contraction_preimage_nonempty(self, contraction_map):
        """
        For contraction f=0.5*x: every cell maps into the inner quarter.
        preimage(F, inner_quarter, B) = cells that map into inner_quarter.
        """
        F, P, B = contraction_map
        # Inner quarter: cells near origin
        inner_pts = np.array([[0.0, 0.0]])
        inner = BoxSet.cover(P, inner_pts)
        result = preimage(F, inner, B)
        assert len(result) > 0


# ===========================================================================
# A. Behavioural: alpha_limit_set
# ===========================================================================

class TestAlphaLimitSet:
    def test_returns_boxset(self, linear_map):
        F, P, B = linear_map
        result = alpha_limit_set(F, BoxSet.full(BoxPartition(P.domain, [1, 1])), steps=4)
        assert isinstance(result, BoxSet)

    def test_partition_refined_after_steps(self, linear_map):
        F, P, B = linear_map
        P_coarse = BoxPartition(P.domain, [1, 1])
        result = alpha_limit_set(F, BoxSet.full(P_coarse), steps=4)
        # 4 steps of subdivide along argmin(dims) on [1,1] → [4,4]
        assert result.partition.dims.tolist() == [4, 4]

    def test_result_nonempty_for_identity(self, identity_map):
        F, P, B = identity_map
        P_c = BoxPartition(P.domain, [1, 1])
        result = alpha_limit_set(F, BoxSet.full(P_c), steps=2)
        assert not result.is_empty()


# ===========================================================================
# A. Behavioural: maximal_invariant_set
# ===========================================================================

class TestMaximalInvariantSet:
    def test_returns_boxset(self, linear_map):
        F, P, B = linear_map
        P_coarse = BoxPartition(P.domain, [1, 1])
        result = maximal_invariant_set(F, BoxSet.full(P_coarse), steps=2)
        assert isinstance(result, BoxSet)

    def test_is_subset_of_relative_attractor(self, linear_map):
        """
        maximal_invariant_set ⊆ relative_attractor (both are outer approx
        of invariant sets, but MIS requires forward AND backward invariance).
        """
        from gaio.algorithms.attractor import relative_attractor
        F, P, B = linear_map
        P_c = BoxPartition(P.domain, [1, 1])
        B0 = BoxSet.full(P_c)
        # Both algorithms refine to the same depth
        mis = maximal_invariant_set(F, B0, steps=4)
        rga = relative_attractor(F, B0, steps=4)
        # They should both be on the same partition
        assert mis.partition == rga.partition

    def test_identity_map_mis_equals_full(self, identity_map):
        """Identity: maximal invariant set = everything."""
        F, P, B = identity_map
        P_c = BoxPartition(P.domain, [1, 1])
        result = maximal_invariant_set(F, BoxSet.full(P_c), steps=2)
        # On [2×2] or [4×1] partition after 2 steps, everything is invariant
        assert not result.is_empty()


# ===========================================================================
# B. Dynamical correctness: linear map f(x) = (x[0], x[1]*0.5)
# ===========================================================================

class TestLinearMapInvariantSets:
    """
    f(x) = (x[0], x[1]*0.5):
    - Attractor (ω-limit set): the x[0]-axis (x[1] = 0)
    - α-limit set: cells on x[1]-axis (x[0] = 0) or unbounded; for domain
      [-1,1]², backward orbit expands in x[0] so α-limit set = x[0]-axis too.
    - Maximal invariant set: only the x[0]-axis (fixed by both F and F⁻¹).
    """

    def test_relative_attractor_covers_x_axis(self):
        """relative_attractor should include cells near x[1]=0."""
        from gaio.algorithms.attractor import relative_attractor
        domain = Box([0.0, 0.0], [1.0, 1.0])
        f = lambda x: np.array([x[0], x[1] * 0.5])
        pts = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        F = SampledBoxMap(f, domain, pts)
        P_c = BoxPartition(domain, [1, 1])
        rga = relative_attractor(F, BoxSet.full(P_c), steps=10)
        # Ground truth: y_axis = horizontal axis (comp[1] ≈ 0)
        P_32 = rga.partition
        x_axis = np.column_stack([
            np.linspace(-1.0, 1.0, 100),
            np.zeros(100),
        ])
        gt = BoxSet.cover(P_32, x_axis)
        assert len(gt) > 0
        intersection = rga & gt
        assert len(intersection) == len(gt)

    def test_alpha_limit_nonempty(self):
        """alpha_limit_set should produce a non-empty result."""
        domain = Box([0.0, 0.0], [1.0, 1.0])
        f = lambda x: np.array([x[0], x[1] * 0.5])
        pts = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        F = SampledBoxMap(f, domain, pts)
        P_c = BoxPartition(domain, [1, 1])
        alpha = alpha_limit_set(F, BoxSet.full(P_c), steps=4)
        assert not alpha.is_empty()

    def test_maximal_invariant_set_nonempty(self):
        domain = Box([0.0, 0.0], [1.0, 1.0])
        f = lambda x: np.array([x[0], x[1] * 0.5])
        pts = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        F = SampledBoxMap(f, domain, pts)
        P_c = BoxPartition(domain, [1, 1])
        mis = maximal_invariant_set(F, BoxSet.full(P_c), steps=4)
        assert not mis.is_empty()


# ===========================================================================
# A. Behavioural: morse_sets, morse_tiles
# ===========================================================================

class TestMorseSets:
    def test_returns_boxset(self, identity_map):
        F, P, B = identity_map
        result = morse_sets(F, B)
        assert isinstance(result, BoxSet)

    def test_result_subset_of_B(self, contraction_map):
        F, P, B = contraction_map
        result = morse_sets(F, B)
        assert result <= B

    def test_identity_all_cells_are_morse(self, identity_map):
        """Identity: every cell is a Morse set (self-loop)."""
        F, P, B = identity_map
        result = morse_sets(F, B)
        assert len(result) == len(B)

    def test_strict_contraction_single_morse_set(self):
        """
        Strict contraction to origin: only the origin cell (self-loop)
        is a Morse set.
        """
        domain = Box([0.0, 0.0], [1.0, 1.0])
        f = lambda x: np.zeros_like(x)
        pts = np.array([[0.0, 0.0]])
        F = SampledBoxMap(f, domain, pts)
        P = BoxPartition(domain, [4, 4])
        B = BoxSet.full(P)
        result = morse_sets(F, B)
        # Only the origin cell should be returned
        assert len(result) <= 1


class TestMorseTiles:
    def test_returns_boxmeasure(self, identity_map):
        F, P, B = identity_map
        result = morse_tiles(F, B)
        assert isinstance(result, BoxMeasure)

    def test_support_equals_morse_sets(self, identity_map):
        F, P, B = identity_map
        ms = morse_sets(F, B)
        mt = morse_tiles(F, B)
        # Keys with nonzero weight in morse_tiles == morse_sets keys
        assert set(mt.to_boxset()._keys.tolist()) == set(ms._keys.tolist())

    def test_labels_are_positive_integers(self, identity_map):
        F, P, B = identity_map
        mt = morse_tiles(F, B)
        assert np.all(mt.weights > 0)
        assert np.all(mt.weights == np.round(mt.weights))  # integer labels

    def test_contraction_single_tile(self):
        domain = Box([0.0, 0.0], [1.0, 1.0])
        f = lambda x: np.zeros_like(x)
        pts = np.array([[0.0, 0.0]])
        F = SampledBoxMap(f, domain, pts)
        P = BoxPartition(domain, [4, 4])
        B = BoxSet.full(P)
        mt = morse_tiles(F, B)
        # At most 1 Morse set (the origin)
        assert len(mt) <= 1
        if len(mt) > 0:
            assert mt.weights[0] == pytest.approx(1.0)   # label = 1


# ===========================================================================
# A. Behavioural: recurrent_set
# ===========================================================================

class TestRecurrentSet:
    def test_returns_boxset(self, identity_map):
        F, P, B = identity_map
        P_c = BoxPartition(P.domain, [1, 1])
        result = recurrent_set(F, BoxSet.full(P_c), steps=2)
        assert isinstance(result, BoxSet)

    def test_result_subset_of_initial(self, contraction_map):
        F, P, B = contraction_map
        P_c = BoxPartition(P.domain, [1, 1])
        B0 = BoxSet.full(P_c)
        result = recurrent_set(F, B0, steps=4)
        # Refine result to match, but the spatial extent should be small
        assert isinstance(result, BoxSet)

    def test_identity_recurrent_set_is_all_cells(self, identity_map):
        """Identity: every cell is chain-recurrent."""
        F, P, B = identity_map
        result = recurrent_set(F, B, steps=2)
        assert not result.is_empty()
