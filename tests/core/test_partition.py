"""
Tests for gaio.core.BoxPartition.

Focus areas:
  - Construction and property invariants
  - Key ↔ multi-index ↔ spatial round-trips (the core correctness guarantee)
  - point_to_key scalar and batch forms
  - keys_in_box range query
  - Refinement (subdivide)
"""
import numpy as np
import pytest

from gaio import Box, BoxPartition, F64, I64


# ── Construction ──────────────────────────────────────────────────────────────

class TestConstruction:
    def test_basic(self, domain_2d):
        p = BoxPartition(domain_2d, [4, 4])
        assert p.ndim == 2
        assert p.size == 16

    def test_dims_dtype(self, partition_2d):
        assert partition_2d.dims.dtype == I64

    def test_cell_radius_dtype(self, partition_2d):
        assert partition_2d.cell_radius.dtype == F64

    def test_cell_radius_value(self, partition_2d):
        # domain radius = [1,1], dims = [4,4]  =>  cell_radius = [0.25, 0.25]
        assert np.allclose(partition_2d.cell_radius, [0.25, 0.25])

    def test_wrong_dims_length_raises(self, domain_2d):
        with pytest.raises(ValueError):
            BoxPartition(domain_2d, [4])

    def test_zero_dim_raises(self, domain_2d):
        with pytest.raises(ValueError, match="≥ 1"):
            BoxPartition(domain_2d, [4, 0])

    def test_asymmetric_partition(self, partition_2d_asym):
        assert partition_2d_asym.size == 15  # 3 × 5

    def test_3d_partition_size(self, partition_3d):
        assert partition_3d.size == 64  # 4³


# ── Key ↔ multi-index ─────────────────────────────────────────────────────────

class TestKeyMultiIndex:
    def test_first_key_multi(self, partition_2d):
        # flat key 0  →  multi-index (0, 0)
        assert np.array_equal(partition_2d.key_to_multi(0), [0, 0])

    def test_last_key_multi(self, partition_2d):
        # flat key 15  →  multi-index (3, 3)  for a 4×4 grid
        assert np.array_equal(partition_2d.key_to_multi(15), [3, 3])

    def test_middle_key_multi(self, partition_2d):
        # key 5 in a 4×4 grid (C-order): row 1, col 1  →  (1, 1)
        assert np.array_equal(partition_2d.key_to_multi(5), [1, 1])

    def test_multi_to_key_roundtrip(self, partition_2d):
        for k in range(partition_2d.size):
            multi = partition_2d.key_to_multi(k)
            assert partition_2d.multi_to_key(multi) == k

    def test_batch_key_to_multi(self, partition_2d):
        keys = np.arange(16, dtype=I64)
        multi = partition_2d.key_to_multi(keys)
        assert multi.shape == (16, 2)
        assert multi.dtype == I64


# ── Key ↔ spatial geometry ────────────────────────────────────────────────────

class TestKeyToBox:
    def test_first_cell_geometry(self, partition_2d):
        # cell (0,0): lo corner of domain + half a cell in each direction
        cell = partition_2d.key_to_box(0)
        assert np.allclose(cell.radius, [0.25, 0.25])
        assert np.allclose(cell.center, [-0.75, -0.75])

    def test_cell_radius_uniform(self, partition_2d):
        # every cell has the same radius
        for k in range(partition_2d.size):
            cell = partition_2d.key_to_box(k)
            assert np.allclose(cell.radius, partition_2d.cell_radius)

    def test_cells_tile_domain(self, partition_2d):
        # union of all cell volumes equals domain volume
        total = sum(
            partition_2d.key_to_box(k).volume
            for k in range(partition_2d.size)
        )
        assert total == pytest.approx(partition_2d.domain.volume)

    def test_cell_centre_in_domain(self, partition_2d):
        for k in range(partition_2d.size):
            c = partition_2d.key_to_box(k).center
            assert partition_2d.domain.contains_point(c)

    def test_cells_disjoint(self, partition_2d):
        # Sample a few pairs and verify non-intersection
        pairs = [(0, 1), (0, 4), (3, 12), (0, 15)]
        for i, j in pairs:
            b_i = partition_2d.key_to_box(i)
            b_j = partition_2d.key_to_box(j)
            assert not b_i.intersects(b_j), f"cells {i} and {j} should not intersect"


# ── point_to_key ──────────────────────────────────────────────────────────────

class TestPointToKey:
    def test_roundtrip_all_cells(self, partition_2d):
        # For every cell, its centre should map back to that key
        for k in range(partition_2d.size):
            centre = partition_2d.key_to_box(k).center
            recovered = partition_2d.point_to_key(centre)
            assert recovered == k, f"Round-trip failed for key {k}"

    def test_lower_domain_boundary_maps_to_key_0(self, partition_2d):
        lo = partition_2d.domain.lo
        assert partition_2d.point_to_key(lo) == 0

    def test_outside_domain_returns_none(self, partition_2d):
        outside = np.array([5.0, 5.0])
        assert partition_2d.point_to_key(outside) is None

    def test_upper_boundary_returns_none(self, partition_2d):
        # hi is outside the half-open domain
        hi = partition_2d.domain.hi
        assert partition_2d.point_to_key(hi) is None

    def test_3d_roundtrip(self, partition_3d):
        for k in range(partition_3d.size):
            c = partition_3d.key_to_box(k).center
            assert partition_3d.point_to_key(c) == k


class TestPointToKeyBatch:
    def test_batch_shape(self, partition_2d):
        pts = np.random.default_rng(0).uniform(-1, 1, (100, 2))
        keys = partition_2d.point_to_key_batch(pts)
        assert keys.shape == (100,)
        assert keys.dtype == I64

    def test_batch_out_of_domain_is_minus_one(self, partition_2d):
        pts = np.array([[5.0, 5.0], [-5.0, 0.0]])
        keys = partition_2d.point_to_key_batch(pts)
        assert np.all(keys == -1)

    def test_batch_consistent_with_scalar(self, partition_2d):
        rng = np.random.default_rng(42)
        pts = rng.uniform(-0.999, 0.999, (50, 2))
        batch_keys = partition_2d.point_to_key_batch(pts)
        for i, p in enumerate(pts):
            assert partition_2d.point_to_key(p) == batch_keys[i]

    def test_batch_all_cell_centres(self, partition_2d):
        centres = np.array([
            partition_2d.key_to_box(k).center
            for k in range(partition_2d.size)
        ])
        keys = partition_2d.point_to_key_batch(centres)
        expected = np.arange(partition_2d.size, dtype=I64)
        assert np.array_equal(keys, expected)


# ── keys_in_box range query ───────────────────────────────────────────────────

class TestKeysInBox:
    def test_full_domain_returns_all_keys(self, partition_2d):
        keys = partition_2d.keys_in_box(partition_2d.domain)
        assert set(keys.tolist()) == set(range(partition_2d.size))

    def test_single_cell_query(self, partition_2d):
        # A query box exactly covering cell 5 should return at least key 5
        cell5 = partition_2d.key_to_box(5)
        keys = partition_2d.keys_in_box(cell5)
        assert 5 in keys

    def test_disjoint_query_returns_empty(self, partition_2d):
        far_away = Box([10.0, 10.0], [0.5, 0.5])
        keys = partition_2d.keys_in_box(far_away)
        assert len(keys) == 0

    def test_half_domain_query(self, partition_2d):
        # Left half: x ∈ [-1, 0)
        left_half = Box([-0.5, 0.0], [0.5, 1.0])
        keys = partition_2d.keys_in_box(left_half)
        # All returned centres should be in the left half of the domain
        for k in keys:
            cx = partition_2d.key_to_box(int(k)).center[0]
            assert cx < 0.0, f"key {k} has centre.x = {cx} which is not in left half"


# ── Refinement ────────────────────────────────────────────────────────────────

class TestSubdivide:
    def test_subdivide_dim0_doubles_dim(self, partition_2d):
        refined = partition_2d.subdivide(0)
        assert refined.dims[0] == partition_2d.dims[0] * 2
        assert refined.dims[1] == partition_2d.dims[1]  # unchanged

    def test_subdivide_size_doubles(self, partition_2d):
        refined = partition_2d.subdivide(0)
        assert refined.size == partition_2d.size * 2

    def test_subdivide_domain_unchanged(self, partition_2d):
        refined = partition_2d.subdivide(0)
        assert refined.domain == partition_2d.domain

    def test_subdivide_cell_radius_halved(self, partition_2d):
        refined = partition_2d.subdivide(0)
        assert refined.cell_radius[0] == pytest.approx(partition_2d.cell_radius[0] / 2)
        assert refined.cell_radius[1] == pytest.approx(partition_2d.cell_radius[1])

    def test_subdivide_all(self, partition_2d):
        refined = partition_2d.subdivide_all()
        assert refined.size == partition_2d.size * 4  # 2² factor in 2D

    def test_equality(self, domain_2d):
        p1 = BoxPartition(domain_2d, [4, 4])
        p2 = BoxPartition(domain_2d, [4, 4])
        assert p1 == p2

    def test_inequality_dims(self, domain_2d):
        p1 = BoxPartition(domain_2d, [4, 4])
        p2 = BoxPartition(domain_2d, [8, 4])
        assert p1 != p2


# ── Direct ports of GAIO.jl partition_regular.jl tests ───────────────────────

class TestGAIOJL:
    """
    Exact Python equivalents of every assertion in
    GAIO.jl/test/partition_regular.jl.

    Notes on translation
    --------------------
    * Julia uses 1-based CartesianIndex keys; Python uses 0-based flat int64.
    * ``checkbounds(Bool, partition, key)`` → ``0 <= key < partition.size``.
    * ``GAIO.point_to_key`` returns ``nothing`` for out-of-domain points;
      our ``point_to_key`` returns ``None``.
    * Wrong-dimension points raise ``DimensionMismatch`` in Julia;
      NumPy raises ``ValueError`` for shape mismatches.
    * Out-of-range flat keys raise ``IndexError`` (we added this validation).
    """

    # -- exported functionality / basics --------------------------------------

    @pytest.mark.gaio_jl
    def test_basics_ndims_4d(self, domain_4d):
        """partition_regular.jl @testset "basics" — ndims(partition) == 4"""
        partition = BoxPartition(domain_4d, [1, 1, 1, 1])
        assert partition.ndim == 4

    # -- exported functionality / subdivision ---------------------------------

    @pytest.mark.gaio_jl
    def test_subdivision_loop_preserves_ndim(self, domain_4d):
        """
        partition_regular.jl @testset "subdivision"

        Julia: subdivide 10 times cycling through dims (k%3)+1 (1-indexed).
        Python: cycle through dims k%3 (0-indexed) — same 3 dims out of 4.
        After any sequence of subdivisions, ndim must remain unchanged.
        """
        partition = BoxPartition(domain_4d, np.ones(4, dtype=I64))
        for k in range(10):
            partition = partition.subdivide(k % 3)
        assert partition.ndim == 4

    # -- exported functionality / size ----------------------------------------

    @pytest.mark.gaio_jl
    def test_size_asymmetric_partition(self):
        """partition_regular.jl @testset "size" — size(partition) == (4, 2)"""
        domain = Box(np.array([0.0, 1.0]), np.array([1.0, 1.0]))
        partition = BoxPartition(domain, [4, 2])
        assert tuple(partition.dims) == (4, 2)
        assert partition.size == 8

    # -- internal functionality / keys ----------------------------------------

    @pytest.mark.gaio_jl
    def test_keys_length_matches_partition_length(self):
        """partition_regular.jl @testset "keys" — length(keys) == length(partition)"""
        domain = Box(np.zeros(3), np.ones(3))
        partition = BoxPartition(domain, [4, 2, 2])
        all_keys = partition.all_keys()
        assert len(all_keys) == partition.size

    # -- internal functionality / point_to_key --------------------------------

    @pytest.mark.gaio_jl
    def test_point_to_key_exact_cases(self):
        """
        partition_regular.jl @testset "point to key" — verbatim point list.

        Partition: 3-D domain centre=(0,0,0) radius=(1,1,1), dims=(4,2,2).
        Points mirror exactly those used in the Julia test.
        """
        domain    = Box(np.zeros(3), np.ones(3))
        partition = BoxPartition(domain, [4, 2, 2])

        inside            = np.array([ 0.5,  0.5,  0.5])
        left              = np.array([-1.0, -1.0, -1.0])   # lo corner
        right             = np.array([ 1.0,  1.0,  1.0])   # hi corner (outside)
        on_boundary_left  = np.array([ 0.0,  0.0, -1.0])   # z at lo
        on_boundary_right = np.array([ 0.0,  1.0,  0.0])   # y at hi (outside)
        outside_left      = np.array([ 0.0,  0.0, -2.0])
        outside_right     = np.array([ 0.0,  2.0,  0.0])
        eps_val = np.finfo(float).eps

        # !isnothing(point_to_key(partition, inside))
        assert partition.point_to_key(inside) is not None
        # !isnothing(point_to_key(partition, left))  — lo boundary included
        assert partition.point_to_key(left) is not None
        # isnothing(point_to_key(partition, right))  — hi boundary excluded
        assert partition.point_to_key(right) is None
        # on_boundary_left: z=-1 is the lo → included
        key_obl = partition.point_to_key(on_boundary_left)
        assert key_obl is not None
        assert 0 <= key_obl < partition.size       # checkbounds passes
        # on_boundary_left + eps: still inside
        key_obl_eps = partition.point_to_key(on_boundary_left + eps_val)
        assert key_obl_eps is not None
        assert 0 <= key_obl_eps < partition.size
        # on_boundary_right: y=1 is the hi → excluded
        assert partition.point_to_key(on_boundary_right) is None
        # on_boundary_right - eps: just inside
        key_obr_eps = partition.point_to_key(on_boundary_right - eps_val)
        assert key_obr_eps is not None
        assert 0 <= key_obr_eps < partition.size
        key_obr_5eps = partition.point_to_key(on_boundary_right - 5 * eps_val)
        assert key_obr_5eps is not None
        assert 0 <= key_obr_5eps < partition.size
        key_obr_10eps = partition.point_to_key(on_boundary_right - 10 * eps_val)
        assert key_obr_10eps is not None
        assert 0 <= key_obr_10eps < partition.size
        # truly outside
        assert partition.point_to_key(outside_left)  is None
        assert partition.point_to_key(outside_right) is None
        # key type for domain centre
        key_centre = partition.point_to_key(partition.domain.center)
        assert isinstance(key_centre, (int, np.integer))

    # -- internal functionality / key_to_box ----------------------------------

    @pytest.mark.gaio_jl
    def test_key_to_box_returns_box_type(self):
        """partition_regular.jl @testset "key to point" — typeof == Box"""
        domain    = Box(np.zeros(3), np.ones(3))
        partition = BoxPartition(domain, [4, 2, 2])
        inside    = np.array([0.5, 0.5, 0.5])
        left      = np.array([-1.0, -1.0, -1.0])
        key_inside = partition.point_to_key(inside)
        key_left   = partition.point_to_key(left)
        assert isinstance(partition.key_to_box(key_inside), Box)
        assert partition.key_to_box(key_inside) != partition.key_to_box(key_left)

    # -- internal functionality / roundtrip -----------------------------------

    @pytest.mark.gaio_jl
    def test_roundtrip_point_in_box_and_centre_maps_back(self):
        """partition_regular.jl @testset "roundtrip" """
        domain    = Box(np.zeros(3), np.ones(3))
        partition = BoxPartition(domain, [4, 2, 2])
        point  = np.array([0.3, 0.3, 0.3])
        key    = partition.point_to_key(point)
        box    = partition.key_to_box(key)
        # point ∈ box
        assert point in box
        # key is stable: box.center maps back to the same key
        key2 = partition.point_to_key(box.center)
        assert key == key2

    # -- internal functionality / points with wrong dimension -----------------

    @pytest.mark.gaio_jl
    def test_wrong_dimension_point_raises(self):
        """partition_regular.jl @testset "points with wrong dimension" """
        domain    = Box(np.zeros(3), np.ones(3))
        partition = BoxPartition(domain, [4, 2, 2])
        point_2d = np.array([0.0, 0.0])
        point_4d = np.array([0.0, 0.0, 0.0, 0.0])
        with pytest.raises((ValueError, Exception)):
            partition.point_to_key(point_2d)
        with pytest.raises((ValueError, Exception)):
            partition.point_to_key(point_4d)

    # -- internal functionality / non existing keys ---------------------------

    @pytest.mark.gaio_jl
    def test_key_to_box_out_of_range_raises(self):
        """
        partition_regular.jl @testset "non existing keys"

        Julia tests out-of-range CartesianIndex multi-indices.
        We expose key_to_box via flat keys; an out-of-range flat key raises
        IndexError (added in this phase to mirror the Julia BoundsError).
        """
        domain    = Box(np.zeros(3), np.ones(3))
        partition = BoxPartition(domain, [4, 2, 2])
        # negative flat key
        with pytest.raises(IndexError):
            partition.key_to_box(-1)
        # key == size (one past the end)
        with pytest.raises(IndexError):
            partition.key_to_box(partition.size)
        # key way out of range
        with pytest.raises(IndexError):
            partition.key_to_box(partition.size + 100)
