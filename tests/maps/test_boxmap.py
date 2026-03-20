"""
tests/maps/test_boxmap.py
=========================
Tests for gaio.maps: SampledBoxMap, GridMap, MonteCarloMap.

Structure mirrors test_box.py and test_boxset.py:
  - TestSampledBoxMapConstruction  — invariants enforced at construction time
  - TestGridMapConstruction         — GridMap factory validation
  - TestGridMapGeometry             — exact grid point layout (GAIO.jl parity)
  - TestMonteCarloMapConstruction   — MonteCarloMap factory validation
  - TestMapBoxesBasic               — map_boxes correctness on known maps
  - TestMapBoxesProperties          — mathematical laws outer approx must satisfy
  - TestGAIOJL                      — exact port of GAIO.jl/test/boxmap.jl
"""
import numpy as np
import pytest

from gaio import Box, BoxPartition, BoxSet, F64, I64
from gaio.maps import SampledBoxMap, GridMap, MonteCarloMap


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def domain_1d():
    return Box(np.array([0.0]), np.array([1.0]))


@pytest.fixture
def domain_2d():
    return Box(np.zeros(2), np.ones(2))


@pytest.fixture
def domain_3d():
    return Box(np.zeros(3), np.ones(3))


@pytest.fixture
def domain_4d():
    return Box(np.zeros(4), np.ones(4))


@pytest.fixture
def partition_4x4(domain_2d):
    return BoxPartition(domain_2d, [4, 4])


@pytest.fixture
def partition_8x8(domain_2d):
    return BoxPartition(domain_2d, [8, 8])


@pytest.fixture
def partition_32(domain_2d):
    """32 × 32 partition — matches GAIO.jl boxmap.jl test."""
    return BoxPartition(domain_2d, [32, 32])


@pytest.fixture
def identity_map():
    return lambda x: x.copy()


@pytest.fixture
def square_map():
    """f(x) = x² componentwise — matches GAIO.jl test."""
    return lambda x: x ** 2


@pytest.fixture
def contraction_map():
    """f(x) = 0.5 * x — contractive, maps domain into itself."""
    return lambda x: 0.5 * x


@pytest.fixture
def neg_map():
    """f(x) = -x — maps each point to its reflection through origin."""
    return lambda x: -x


# ── TestSampledBoxMapConstruction ─────────────────────────────────────────────

class TestSampledBoxMapConstruction:
    """Invariants that must hold immediately after __init__."""

    def test_basic_2d(self, domain_2d, identity_map):
        unit_pts = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        g = SampledBoxMap(identity_map, domain_2d, unit_pts)
        assert isinstance(g, SampledBoxMap)

    def test_1d_domain(self, domain_1d, identity_map):
        unit_pts = np.array([[-1.0], [0.0], [1.0]])
        g = SampledBoxMap(identity_map, domain_1d, unit_pts)
        assert g.ndim == 1
        assert g.n_test_points == 3

    def test_3d_domain(self, domain_3d, identity_map):
        unit_pts = np.zeros((5, 3))
        g = SampledBoxMap(identity_map, domain_3d, unit_pts)
        assert g.ndim == 3
        assert g.n_test_points == 5

    def test_domain_stored(self, domain_2d, identity_map):
        unit_pts = np.zeros((4, 2))
        g = SampledBoxMap(identity_map, domain_2d, unit_pts)
        assert g.domain == domain_2d

    def test_map_callable_stored(self, domain_2d):
        f = lambda x: x * 2
        unit_pts = np.zeros((2, 2))
        g = SampledBoxMap(f, domain_2d, unit_pts)
        assert g.map is f

    def test_n_test_points_property(self, domain_2d, identity_map):
        unit_pts = np.zeros((7, 2))
        g = SampledBoxMap(identity_map, domain_2d, unit_pts)
        assert g.n_test_points == 7

    def test_ndim_property(self, domain_2d, identity_map):
        unit_pts = np.zeros((3, 2))
        g = SampledBoxMap(identity_map, domain_2d, unit_pts)
        assert g.ndim == 2

    def test_unit_points_dtype_enforced(self, domain_2d, identity_map):
        unit_pts = np.array([[0, 0], [1, 1]], dtype=np.int32)
        g = SampledBoxMap(identity_map, domain_2d, unit_pts)
        assert g._unit_points.dtype == F64

    def test_unit_points_c_contiguous(self, domain_2d, identity_map):
        unit_pts = np.array([[-1.0, 0.0], [1.0, 0.0]])
        g = SampledBoxMap(identity_map, domain_2d, unit_pts)
        assert g._unit_points.flags["C_CONTIGUOUS"]

    def test_single_test_point(self, domain_2d, identity_map):
        unit_pts = np.zeros((1, 2))
        g = SampledBoxMap(identity_map, domain_2d, unit_pts)
        assert g.n_test_points == 1

    def test_many_test_points(self, domain_2d, identity_map):
        unit_pts = np.random.default_rng(0).uniform(-1, 1, size=(500, 2))
        g = SampledBoxMap(identity_map, domain_2d, unit_pts)
        assert g.n_test_points == 500

    def test_1d_unit_points_raises(self, domain_2d, identity_map):
        with pytest.raises(ValueError, match="2-D"):
            SampledBoxMap(identity_map, domain_2d, np.array([0.0, 1.0]))

    def test_repr_contains_class_name(self, domain_2d, identity_map):
        g = SampledBoxMap(identity_map, domain_2d, np.zeros((3, 2)))
        assert "SampledBoxMap" in repr(g)

    def test_repr_contains_n_test_points(self, domain_2d, identity_map):
        g = SampledBoxMap(identity_map, domain_2d, np.zeros((5, 2)))
        assert "n_test_points=5" in repr(g)

    def test_repr_contains_domain(self, domain_2d, identity_map):
        g = SampledBoxMap(identity_map, domain_2d, np.zeros((3, 2)))
        assert "domain" in repr(g)


# ── TestGridMapConstruction ───────────────────────────────────────────────────

class TestGridMapConstruction:
    """GridMap factory validation: inputs, defaults, error handling."""

    def test_returns_sampled_box_map(self, domain_2d, square_map):
        assert isinstance(GridMap(square_map, domain_2d), SampledBoxMap)

    def test_default_n_points_2d(self, domain_2d, square_map):
        g = GridMap(square_map, domain_2d)
        assert g.n_test_points == 4 ** 2

    def test_default_n_points_3d(self, domain_3d, square_map):
        g = GridMap(square_map, domain_3d)
        assert g.n_test_points == 4 ** 3

    def test_default_n_points_4d(self, domain_4d, square_map):
        g = GridMap(square_map, domain_4d)
        assert g.n_test_points == 4 ** 4

    def test_int_n_points_broadcast(self, domain_2d, square_map):
        g = GridMap(square_map, domain_2d, n_points=8)
        assert g.n_test_points == 8 * 8

    def test_tuple_n_points_asymmetric(self, domain_2d, square_map):
        g = GridMap(square_map, domain_2d, n_points=(2, 3))
        assert g.n_test_points == 6

    def test_n_points_1_per_dim(self, domain_2d, square_map):
        g = GridMap(square_map, domain_2d, n_points=1)
        assert g.n_test_points == 1

    def test_1d_domain(self, domain_1d, square_map):
        g = GridMap(square_map, domain_1d, n_points=8)
        assert g.n_test_points == 8
        assert g.ndim == 1

    def test_domain_stored(self, domain_2d, square_map):
        g = GridMap(square_map, domain_2d)
        assert g.domain == domain_2d

    def test_wrong_tuple_length_raises(self, domain_2d, square_map):
        with pytest.raises(ValueError, match="length"):
            GridMap(square_map, domain_2d, n_points=(4, 4, 4))

    def test_zero_n_points_raises(self, domain_2d, square_map):
        with pytest.raises(ValueError, match=">= 1"):
            GridMap(square_map, domain_2d, n_points=0)

    def test_zero_in_tuple_raises(self, domain_2d, square_map):
        with pytest.raises(ValueError, match=">= 1"):
            GridMap(square_map, domain_2d, n_points=(4, 0))


# ── TestGridMapGeometry ────────────────────────────────────────────────────────

class TestGridMapGeometry:
    """
    Verify that GridMap produces exactly the same grid as GAIO.jl GridBoxMap.

    GAIO.jl layout (boxmap_sampled.jl):
        Δp = 2 / n_points
        points[i] = Δp * (i - 1) - 1   for i = 1 … n_points

    => u[k] = -1 + k * (2 / n_points)   for k = 0 … n_points-1
    """

    def test_n4_1d_exact_values(self, domain_1d, square_map):
        """n_points=4: grid = [-1.0, -0.5, 0.0, 0.5]."""
        g = GridMap(square_map, domain_1d, n_points=4)
        expected = np.array([[-1.0], [-0.5], [0.0], [0.5]])
        np.testing.assert_allclose(g._unit_points, expected)

    def test_n1_gives_minus_one(self, domain_2d, square_map):
        """n_points=1: single test point at [-1, -1] (corner)."""
        g = GridMap(square_map, domain_2d, n_points=1)
        assert g.n_test_points == 1
        np.testing.assert_allclose(g._unit_points, [[-1.0, -1.0]])

    def test_n2_values_2d(self, domain_2d, square_map):
        """n_points=2: grid is {-1, 0}² = 4 points."""
        g = GridMap(square_map, domain_2d, n_points=2)
        assert g.n_test_points == 4
        assert set(g._unit_points.ravel().tolist()) == {-1.0, 0.0}

    def test_grid_starts_at_minus_one(self, domain_2d, square_map):
        """First coordinate of the grid must be -1.0 in every dimension."""
        for n in [2, 4, 8, 16]:
            g = GridMap(square_map, domain_2d, n_points=n)
            # The first row of unit_points is the (0,0,...) multi-index → [-1,-1]
            assert np.allclose(g._unit_points[0], -1.0), f"failed for n={n}"

    def test_grid_ends_below_one(self, domain_2d, square_map):
        """All grid points must be < 1.0 (grid does NOT include +1)."""
        for n in [2, 4, 8]:
            g = GridMap(square_map, domain_2d, n_points=n)
            assert np.all(g._unit_points < 1.0)

    def test_grid_spacing_matches_julia(self, domain_1d, square_map):
        """Δp = 2/n; consecutive points must be exactly Δp apart."""
        for n in [3, 5, 7]:
            g = GridMap(square_map, domain_1d, n_points=n)
            pts = np.sort(g._unit_points[:, 0])
            diffs = np.diff(pts)
            np.testing.assert_allclose(diffs, 2.0 / n, rtol=1e-12)

    def test_all_unit_points_in_valid_range(self, domain_2d, square_map):
        """All unit points must be in [-1, 1)."""
        g = GridMap(square_map, domain_2d, n_points=8)
        assert np.all(g._unit_points >= -1.0)
        assert np.all(g._unit_points < 1.0)

    def test_unit_points_dtype(self, domain_2d, square_map):
        g = GridMap(square_map, domain_2d, n_points=4)
        assert g._unit_points.dtype == F64

    def test_unit_points_c_contiguous(self, domain_2d, square_map):
        g = GridMap(square_map, domain_2d, n_points=4)
        assert g._unit_points.flags["C_CONTIGUOUS"]

    def test_asymmetric_grid_shape(self, domain_2d, square_map):
        """n_points=(3, 5) → 15 points, shape (15, 2)."""
        g = GridMap(square_map, domain_2d, n_points=(3, 5))
        assert g._unit_points.shape == (15, 2)

    def test_grid_covers_expected_x_values(self, domain_1d, square_map):
        """n_points=3: x values = [-1, -1/3, 1/3] approximately."""
        g = GridMap(square_map, domain_1d, n_points=3)
        expected = np.array([[-1.0], [-1.0 + 2.0/3], [-1.0 + 4.0/3]])
        np.testing.assert_allclose(g._unit_points, expected, rtol=1e-12)


# ── TestMonteCarloMapConstruction ─────────────────────────────────────────────

class TestMonteCarloMapConstruction:
    """MonteCarloMap factory validation."""

    def test_returns_sampled_box_map(self, domain_2d, square_map):
        assert isinstance(MonteCarloMap(square_map, domain_2d, seed=0), SampledBoxMap)

    def test_default_n_points_2d(self, domain_2d, square_map):
        g = MonteCarloMap(square_map, domain_2d)
        assert g.n_test_points == 16 * 2

    def test_default_n_points_3d(self, domain_3d, square_map):
        g = MonteCarloMap(square_map, domain_3d)
        assert g.n_test_points == 16 * 3

    def test_default_n_points_1d(self, domain_1d, square_map):
        g = MonteCarloMap(square_map, domain_1d)
        assert g.n_test_points == 16 * 1

    def test_custom_n_points(self, domain_2d, square_map):
        g = MonteCarloMap(square_map, domain_2d, n_points=100, seed=1)
        assert g.n_test_points == 100

    def test_unit_points_shape(self, domain_2d, square_map):
        g = MonteCarloMap(square_map, domain_2d, n_points=50, seed=0)
        assert g._unit_points.shape == (50, 2)

    def test_unit_points_dtype(self, domain_2d, square_map):
        g = MonteCarloMap(square_map, domain_2d, seed=0)
        assert g._unit_points.dtype == F64

    def test_unit_points_c_contiguous(self, domain_2d, square_map):
        g = MonteCarloMap(square_map, domain_2d, seed=0)
        assert g._unit_points.flags["C_CONTIGUOUS"]

    def test_seed_reproducible(self, domain_2d, square_map):
        g1 = MonteCarloMap(square_map, domain_2d, n_points=50, seed=42)
        g2 = MonteCarloMap(square_map, domain_2d, n_points=50, seed=42)
        np.testing.assert_array_equal(g1._unit_points, g2._unit_points)

    def test_different_seeds_differ(self, domain_2d, square_map):
        g1 = MonteCarloMap(square_map, domain_2d, n_points=50, seed=0)
        g2 = MonteCarloMap(square_map, domain_2d, n_points=50, seed=1)
        assert not np.array_equal(g1._unit_points, g2._unit_points)

    def test_points_in_unit_cube(self, domain_2d, square_map):
        g = MonteCarloMap(square_map, domain_2d, n_points=1000, seed=7)
        assert np.all(g._unit_points >= -1.0)
        assert np.all(g._unit_points <= 1.0)

    def test_domain_stored(self, domain_2d, square_map):
        g = MonteCarloMap(square_map, domain_2d, seed=0)
        assert g.domain == domain_2d

    def test_3d_domain(self, domain_3d, square_map):
        g = MonteCarloMap(square_map, domain_3d, n_points=30, seed=0)
        assert g.ndim == 3
        assert g._unit_points.shape == (30, 3)


# ── TestMapBoxesBasic ─────────────────────────────────────────────────────────

class TestMapBoxesBasic:
    """Correctness of map_boxes on maps with known behaviour."""

    def test_empty_source_returns_empty(self, domain_2d, square_map, partition_4x4):
        source = BoxSet.empty(partition_4x4)
        g = GridMap(square_map, domain_2d)
        result = g(source)
        assert result.is_empty()

    def test_empty_preserves_partition(self, domain_2d, square_map, partition_4x4):
        source = BoxSet.empty(partition_4x4)
        result = GridMap(square_map, domain_2d)(source)
        assert result.partition == partition_4x4

    def test_returns_boxset(self, domain_2d, square_map, partition_4x4):
        source = BoxSet.full(partition_4x4)
        result = GridMap(square_map, domain_2d)(source)
        assert isinstance(result, BoxSet)

    def test_same_partition_as_source(self, domain_2d, square_map, partition_4x4):
        source = BoxSet.full(partition_4x4)
        result = GridMap(square_map, domain_2d)(source)
        assert result.partition == partition_4x4

    def test_result_keys_sorted(self, domain_2d, square_map, partition_4x4):
        source = BoxSet.full(partition_4x4)
        result = GridMap(square_map, domain_2d)(source)
        assert np.array_equal(result.keys, np.sort(result.keys))

    def test_result_keys_unique(self, domain_2d, square_map, partition_4x4):
        source = BoxSet.full(partition_4x4)
        result = GridMap(square_map, domain_2d)(source)
        assert len(result.keys) == len(np.unique(result.keys))

    def test_identity_map_full_set(self, domain_2d, identity_map, partition_4x4):
        """Identity on full set → image == full set."""
        source = BoxSet.full(partition_4x4)
        g = GridMap(identity_map, domain_2d)
        assert g(source) == source

    def test_identity_map_partial_set(self, domain_2d, identity_map, partition_4x4):
        """Identity on any subset → image == that subset."""
        source = BoxSet(partition_4x4, np.arange(6, dtype=I64))
        g = GridMap(identity_map, domain_2d)
        assert g(source) == source

    def test_out_of_domain_filtered(self, domain_2d, partition_4x4):
        """Map returning out-of-domain point → empty image."""
        def far_away(x):
            return np.array([999.0, 999.0])
        g = GridMap(far_away, domain_2d)
        result = g(BoxSet.full(partition_4x4))
        assert result.is_empty()

    def test_constant_map_single_cell(self, domain_2d, partition_4x4):
        """Map to fixed interior point → exactly one cell hit."""
        origin = domain_2d.lo.copy()
        g = GridMap(lambda x: origin, domain_2d)
        result = g(BoxSet.full(partition_4x4))
        assert len(result) == 1

    def test_callable_and_map_boxes_equivalent(self, domain_2d, square_map, partition_4x4):
        source = BoxSet(partition_4x4, np.arange(8, dtype=I64))
        g = GridMap(square_map, domain_2d)
        assert g(source) == g.map_boxes(source)

    def test_image_subset_of_full_partition(self, domain_2d, square_map, partition_4x4):
        source = BoxSet.full(partition_4x4)
        g = GridMap(square_map, domain_2d)
        assert g(source) <= BoxSet.full(partition_4x4)

    def test_single_cell_source(self, domain_2d, square_map, partition_4x4):
        """Single-cell source → valid (possibly 1-cell) image."""
        source = BoxSet(partition_4x4, np.array([7], dtype=I64))
        g = GridMap(square_map, domain_2d)
        result = g(source)
        assert isinstance(result, BoxSet)
        assert result <= BoxSet.full(partition_4x4)

    def test_1d_domain_map(self, domain_1d):
        """1-D map works correctly end-to-end."""
        p = BoxPartition(domain_1d, [8])
        source = BoxSet.full(p)
        g = GridMap(lambda x: x ** 2, domain_1d, n_points=4)
        result = g(source)
        assert isinstance(result, BoxSet)
        assert result.partition == p

    def test_3d_domain_map(self, domain_3d):
        """3-D map works correctly end-to-end."""
        p = BoxPartition(domain_3d, [4, 4, 4])
        source = BoxSet.full(p)
        g = GridMap(lambda x: 0.5 * x, domain_3d, n_points=2)
        result = g(source)
        assert isinstance(result, BoxSet)
        assert result <= BoxSet.full(p)

    def test_contraction_reduces_image(self, domain_2d, contraction_map, partition_8x8):
        """A contractive map pulls the full set image inward."""
        source = BoxSet.full(partition_8x8)
        g = GridMap(contraction_map, domain_2d, n_points=4)
        result = g(source)
        # Image must be non-empty and strictly smaller than full
        assert not result.is_empty()
        assert len(result) < len(source)

    def test_neg_map_symmetric_domain(self, domain_2d, neg_map, partition_4x4):
        """f(x) = -x on a domain symmetric about origin maps full set to itself."""
        source = BoxSet.full(partition_4x4)
        g = GridMap(neg_map, domain_2d, n_points=4)
        result = g(source)
        assert result == source

    def test_image_nonempty_for_full_set(self, domain_2d, square_map, partition_8x8):
        source = BoxSet.full(partition_8x8)
        g = GridMap(square_map, domain_2d)
        assert not g(source).is_empty()

    def test_montecarlo_image_subset_of_full(self, domain_2d, square_map, partition_4x4):
        source = BoxSet.full(partition_4x4)
        g = MonteCarloMap(square_map, domain_2d, n_points=200, seed=0)
        assert g(source) <= BoxSet.full(partition_4x4)

    def test_outer_approximation_contains_true_image(self):
        """
        For f(x) = 0.5*x and a single cell, the cell containing f(center)
        must appear in the image (outer approximation guarantee).
        """
        domain = Box([0.0], [1.0])
        p = BoxPartition(domain, [8])
        # Key 4 → center ≈ 0.125 (in the 5th cell from -1)
        key = 4
        box = p.key_to_box(key)
        center = box.center.copy()
        f_center = 0.5 * center
        expected_key = p.point_to_key(f_center)
        source = BoxSet(p, np.array([key], dtype=I64))
        g = GridMap(lambda x: 0.5 * x, domain, n_points=8)
        result = g(source)
        assert expected_key in result


# ── TestMapBoxesProperties ────────────────────────────────────────────────────

class TestMapBoxesProperties:
    """
    Mathematical laws that any outer-approximation BoxMap must satisfy.

    These hold regardless of the specific map or partition (as long as the
    map stays within the domain).
    """

    def test_monotonicity(self, domain_2d, contraction_map, partition_4x4):
        """
        Monotonicity:  A ⊆ B  ⟹  f(A) ⊆ f(B).

        Proof: every test point generated for a cell in A is also generated
        when processing B (which contains all of A's cells), so every hit key
        from A also appears in f(B).
        """
        g = GridMap(contraction_map, domain_2d)
        full = BoxSet.full(partition_4x4)
        half = BoxSet(partition_4x4, np.arange(8, dtype=I64))
        img_half = g(half)
        img_full = g(full)
        # Since half ⊆ full, we need img_half ⊆ img_full
        assert img_half <= img_full

    def test_image_of_union_equals_union_of_images(
            self, domain_2d, contraction_map, partition_4x4):
        """
        f(A ∪ B) = f(A) ∪ f(B).

        Proof: f(A ∪ B) processes every cell in A and every cell in B;
        the hits are exactly the union of the individual hits.
        """
        g = GridMap(contraction_map, domain_2d)
        setA = BoxSet(partition_4x4, np.arange(0, 8, dtype=I64))
        setB = BoxSet(partition_4x4, np.arange(6, 14, dtype=I64))
        img_union = g(setA | setB)
        img_A = g(setA)
        img_B = g(setB)
        assert img_union == (img_A | img_B)

    def test_empty_source_gives_empty_image(self, domain_2d, square_map, partition_4x4):
        """f(∅) = ∅."""
        g = GridMap(square_map, domain_2d)
        assert g(BoxSet.empty(partition_4x4)).is_empty()

    def test_more_test_points_gives_superset_image(self, domain_2d, square_map):
        """
        More test points → image is at least as large (superset).

        GridMap grid structure: n_points=4 includes all points of n_points=2
        (since {-1, 0} ⊆ {-1, -0.5, 0, 0.5}).
        Therefore image(n=4) ⊇ image(n=2).
        """
        p = BoxPartition(domain_2d, [8, 8])
        source = BoxSet.full(p)
        g2 = GridMap(square_map, domain_2d, n_points=2)
        g4 = GridMap(square_map, domain_2d, n_points=4)
        img2 = g2(source)
        img4 = g4(source)
        # n_points=4 hits a superset of n_points=2 cells
        assert img2 <= img4

    def test_image_always_subset_of_full(
            self, domain_2d, square_map, partition_4x4):
        """Image is always contained in the full partition."""
        g = GridMap(square_map, domain_2d)
        for keys in [np.arange(4), np.arange(8, 16), np.arange(16)]:
            source = BoxSet(partition_4x4, np.array(keys, dtype=I64))
            assert g(source) <= BoxSet.full(partition_4x4)

    def test_image_of_single_cell_subset_of_full(self, domain_2d, contraction_map):
        """Image of any single cell is always a valid subset."""
        p = BoxPartition(domain_2d, [8, 8])
        g = GridMap(contraction_map, domain_2d, n_points=4)
        full = BoxSet.full(p)
        for key in [0, 10, 32, 63]:
            source = BoxSet(p, np.array([key], dtype=I64))
            assert g(source) <= full

    def test_image_stable_under_repeated_application(self, domain_2d, contraction_map):
        """
        For a contractive map on a sub-region, repeated application
        shrinks (or stabilises) the image.
        """
        p = BoxPartition(domain_2d, [16, 16])
        g = GridMap(contraction_map, domain_2d, n_points=4)
        current = BoxSet.full(p)
        prev_len = len(current)
        for _ in range(5):
            current = g(current)
            assert len(current) <= prev_len
            prev_len = len(current)

    def test_grid_map_image_subset_of_montecarlo_image(self, domain_2d, square_map):
        """
        With enough Monte-Carlo points, the MC image covers at least the
        grid image (probabilistically guaranteed with large n_points + fixed seed).
        """
        p = BoxPartition(domain_2d, [8, 8])
        source = BoxSet.full(p)
        g_grid = GridMap(square_map, domain_2d, n_points=4)
        g_mc = MonteCarloMap(square_map, domain_2d, n_points=10_000, seed=0)
        img_grid = g_grid(source)
        img_mc = g_mc(source)
        # With 10k points, MC should cover all 4×4 = 16 grid-point images
        assert img_grid <= img_mc


# ── TestGAIOJL ────────────────────────────────────────────────────────────────

class TestGAIOJL:
    """
    Exact port of GAIO.jl/test/boxmap.jl → @testset "exported functionality".

    Julia source
    ------------
    f(x) = x .^ 2
    test_points = [(-1,-1),(-1,1),(1,-1),(1,1),(-1,0),(1,0),(0,-1),(0,1)]
    domain  = Box(SVector(0.0,0.0), SVector(1.0,1.0))
    g       = BoxMap(:pointdiscretized, f, domain, test_points)
    partition = BoxGrid(domain, (32,32))
    p1,p2,p3  = SVector(0,0), SVector(0.5,0), SVector(0,-0.5)
    boxset    = cover(partition, (p1,p2,p3))
    # ground-truth: map the 4 corners of each box in boxset
    image   = cover(partition, mapped_corners)
    mapped1 = g(boxset)
    @test !isempty(boxset) && !isempty(image)
    @test length(union(image,mapped1)) == length(intersect(image,mapped1))

    Python translation notes
    ------------------------
    * :pointdiscretized BoxMap takes unit-cube test points → SampledBoxMap
    * BoxGrid(domain,(32,32)) → BoxPartition(domain,[32,32])
    * cover(partition,(p1,p2,p3)) → BoxSet.cover(partition, array([p1,p2,p3]))
    * length(union(A,B))==length(intersect(A,B)) ↔ A == B  (set equality)
    """

    @pytest.fixture
    def gaio_setup(self, domain_2d, partition_32):
        def f(x):
            return x ** 2

        # 8 unit-cube points: 4 corners + 4 edge midpoints (Julia test_points)
        test_points = np.array([
            [-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0],
            [-1.0,  0.0], [1.0,  0.0], [0.0, -1.0], [0.0,  1.0],
        ])

        g = SampledBoxMap(f, domain_2d, test_points)

        p1 = np.array([0.0, 0.0])
        p2 = np.array([0.5, 0.0])
        p3 = np.array([0.0, -0.5])
        boxset = BoxSet.cover(partition_32, np.array([p1, p2, p3]))

        return f, g, partition_32, boxset

    @pytest.mark.gaio_jl
    def test_type_is_sampled_box_map(self, gaio_setup):
        """Julia: typeof(g) <: SampledBoxMap"""
        _, g, _, _ = gaio_setup
        assert isinstance(g, SampledBoxMap)

    @pytest.mark.gaio_jl
    def test_test_points_count(self, gaio_setup):
        """Julia: 8 test points specified."""
        _, g, _, _ = gaio_setup
        assert g.n_test_points == 8

    @pytest.mark.gaio_jl
    def test_boxset_nonempty(self, gaio_setup):
        """Julia: @test !isempty(boxset)"""
        _, _, _, boxset = gaio_setup
        assert not boxset.is_empty()

    @pytest.mark.gaio_jl
    def test_three_points_in_distinct_cells(self, gaio_setup):
        """
        p1=(0,0), p2=(0.5,0), p3=(0,-0.5) fall in 3 distinct cells of the
        32×32 partition.  The Julia test implicitly requires boxset to be
        non-trivial (covering multiple cells).
        """
        _, _, _, boxset = gaio_setup
        assert len(boxset) == 3

    @pytest.mark.gaio_jl
    def test_image_nonempty(self, gaio_setup):
        """Julia: @test !isempty(image)"""
        _, g, _, boxset = gaio_setup
        image = g(boxset)
        assert not image.is_empty()

    @pytest.mark.gaio_jl
    def test_image_equals_corner_ground_truth(self, gaio_setup):
        """
        Exact port of:
            @test length(union(image,mapped1)) == length(intersect(image,mapped1))

        The SampledBoxMap with 8 unit-cube points (4 corners + 4 edge midpoints)
        must produce the same image as the naive 4-corner ground truth because
        for f(x)=x² the midpoint images land in cells already covered by corners.
        """
        f, g, partition_32, boxset = gaio_setup

        # Ground truth: map the 4 actual corners of each box (Julia's loop)
        mapped_pts = []
        for _, box in boxset.boxes():
            lo = box.lo
            hi = box.hi
            mapped_pts.append(f(lo))
            mapped_pts.append(f(hi))
            mapped_pts.append(f(np.array([lo[0], hi[1]])))
            mapped_pts.append(f(np.array([hi[0], lo[1]])))

        image = BoxSet.cover(partition_32, np.array(mapped_pts))
        mapped1 = g(boxset)

        # Julia assertion: length(union)==length(intersect) ↔ set equality
        assert len(image | mapped1) == len(image & mapped1)

    @pytest.mark.gaio_jl
    def test_mapped1_subset_of_full(self, gaio_setup):
        """Image is always a subset of the full partition."""
        _, g, partition_32, boxset = gaio_setup
        mapped1 = g(boxset)
        assert mapped1 <= BoxSet.full(partition_32)

    @pytest.mark.gaio_jl
    def test_image_keys_dtype(self, gaio_setup):
        """Image keys must be int64 (required for Numba / MPI compatibility)."""
        _, g, _, boxset = gaio_setup
        image = g(boxset)
        assert image.keys.dtype == I64

    @pytest.mark.gaio_jl
    def test_image_keys_sorted(self, gaio_setup):
        """Image keys must be sorted (BoxSet invariant)."""
        _, g, _, boxset = gaio_setup
        image = g(boxset)
        assert np.array_equal(image.keys, np.sort(image.keys))
