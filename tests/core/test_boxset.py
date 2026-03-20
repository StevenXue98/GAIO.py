"""
Tests for gaio.core.BoxSet.

Focus areas:
  - Construction invariants (sorted, unique, bounds-checked)
  - Membership and iteration
  - Set algebra correctness and commutativity/associativity laws
  - Vectorised geometry accessors (centers, cell_radius, bounds)
  - Convenience constructors (full, empty, cover, from_box)
  - Subdivision and repartitioning
  - Compatibility guard (cross-partition operations raise)
"""
import numpy as np
import pytest

from gaio import Box, BoxPartition, BoxSet, F64, I64


# ── Construction ──────────────────────────────────────────────────────────────

class TestConstruction:
    def test_keys_are_sorted_unique(self, partition_2d):
        s = BoxSet(partition_2d, np.array([5, 3, 3, 1], dtype=I64))
        assert np.array_equal(s.keys, [1, 3, 5])

    def test_keys_dtype(self, partition_2d):
        s = BoxSet(partition_2d, [0, 1, 2])
        assert s.keys.dtype == I64

    def test_keys_c_contiguous(self, partition_2d):
        s = BoxSet(partition_2d, [0, 1, 2])
        assert s.keys.flags["C_CONTIGUOUS"]

    def test_out_of_range_key_raises(self, partition_2d):
        with pytest.raises(ValueError, match="outside"):
            BoxSet(partition_2d, [0, 100])

    def test_negative_key_raises(self, partition_2d):
        with pytest.raises(ValueError, match="outside"):
            BoxSet(partition_2d, [-1, 0])

    def test_empty_construction(self, partition_2d):
        s = BoxSet(partition_2d, [])
        assert len(s) == 0

    def test_full_construction(self, partition_2d):
        s = BoxSet(partition_2d, np.arange(partition_2d.size))
        assert len(s) == partition_2d.size


# ── Convenience constructors ──────────────────────────────────────────────────

class TestConvenienceConstructors:
    def test_full(self, partition_2d):
        s = BoxSet.full(partition_2d)
        assert len(s) == 16
        assert np.array_equal(s.keys, np.arange(16, dtype=I64))

    def test_empty(self, partition_2d):
        s = BoxSet.empty(partition_2d)
        assert len(s) == 0
        assert s.is_empty()

    def test_cover_all_centres(self, partition_2d):
        centres = np.array([
            partition_2d.key_to_box(k).center
            for k in range(partition_2d.size)
        ])
        s = BoxSet.cover(partition_2d, centres)
        assert len(s) == partition_2d.size

    def test_cover_ignores_out_of_domain(self, partition_2d):
        pts = np.array([[0.0, 0.0], [99.0, 99.0]])
        s = BoxSet.cover(partition_2d, pts)
        assert len(s) == 1  # only the in-domain point

    def test_cover_empty_points(self, partition_2d):
        pts = np.empty((0, 2), dtype=F64)
        s = BoxSet.cover(partition_2d, pts)
        assert len(s) == 0

    def test_from_box_full_domain(self, partition_2d):
        s = BoxSet.from_box(partition_2d, partition_2d.domain)
        assert len(s) == partition_2d.size

    def test_from_box_single_cell(self, partition_2d):
        cell_box = partition_2d.key_to_box(7)
        s = BoxSet.from_box(partition_2d, cell_box)
        assert 7 in s

    def test_from_box_outside_domain(self, partition_2d):
        outside = Box([10.0, 10.0], [0.5, 0.5])
        s = BoxSet.from_box(partition_2d, outside)
        assert len(s) == 0


# ── Membership ────────────────────────────────────────────────────────────────

class TestMembership:
    def test_key_in_set(self, half_set_2d):
        for k in range(8):
            assert k in half_set_2d

    def test_key_not_in_set(self, half_set_2d):
        for k in range(8, 16):
            assert k not in half_set_2d

    def test_len(self, half_set_2d):
        assert len(half_set_2d) == 8

    def test_iter_sorted(self, half_set_2d):
        keys = list(half_set_2d)
        assert keys == sorted(keys)

    def test_iter_yields_ints(self, half_set_2d):
        for k in half_set_2d:
            assert isinstance(int(k), int)


# ── Set algebra ───────────────────────────────────────────────────────────────

class TestSetAlgebra:
    """
    half_set_2d  = {0,1,2,3,4,5,6,7}
    overlap_set_2d = {4,5,6,7,8,9,10,11}
    """

    def test_union(self, half_set_2d, overlap_set_2d):
        result = half_set_2d | overlap_set_2d
        assert np.array_equal(result.keys, np.arange(12, dtype=I64))

    def test_intersection(self, half_set_2d, overlap_set_2d):
        result = half_set_2d & overlap_set_2d
        assert np.array_equal(result.keys, np.arange(4, 8, dtype=I64))

    def test_difference(self, half_set_2d, overlap_set_2d):
        result = half_set_2d - overlap_set_2d
        assert np.array_equal(result.keys, np.arange(4, dtype=I64))

    def test_symmetric_difference(self, half_set_2d, overlap_set_2d):
        result = half_set_2d ^ overlap_set_2d
        expected = np.array([0, 1, 2, 3, 8, 9, 10, 11], dtype=I64)
        assert np.array_equal(result.keys, expected)

    def test_union_with_self_is_self(self, half_set_2d):
        result = half_set_2d | half_set_2d
        assert np.array_equal(result.keys, half_set_2d.keys)

    def test_intersection_with_self_is_self(self, half_set_2d):
        result = half_set_2d & half_set_2d
        assert np.array_equal(result.keys, half_set_2d.keys)

    def test_difference_with_self_is_empty(self, half_set_2d):
        result = half_set_2d - half_set_2d
        assert len(result) == 0

    def test_union_with_empty(self, half_set_2d, partition_2d):
        empty = BoxSet.empty(partition_2d)
        assert np.array_equal((half_set_2d | empty).keys, half_set_2d.keys)

    def test_intersection_with_empty(self, half_set_2d, partition_2d):
        empty = BoxSet.empty(partition_2d)
        assert len(half_set_2d & empty) == 0

    def test_union_commutativity(self, half_set_2d, overlap_set_2d):
        assert np.array_equal(
            (half_set_2d | overlap_set_2d).keys,
            (overlap_set_2d | half_set_2d).keys,
        )

    def test_intersection_commutativity(self, half_set_2d, overlap_set_2d):
        assert np.array_equal(
            (half_set_2d & overlap_set_2d).keys,
            (overlap_set_2d & half_set_2d).keys,
        )

    def test_subset_operator(self, half_set_2d, full_set_2d):
        assert half_set_2d <= full_set_2d
        assert not full_set_2d <= half_set_2d

    def test_superset_operator(self, half_set_2d, full_set_2d):
        assert full_set_2d >= half_set_2d
        assert not half_set_2d >= full_set_2d

    def test_cross_partition_raises(self, partition_2d, domain_2d):
        other_partition = BoxPartition(domain_2d, [8, 8])
        s1 = BoxSet.full(partition_2d)
        s2 = BoxSet.full(other_partition)
        with pytest.raises(ValueError, match="partition"):
            _ = s1 | s2


# ── Geometry accessors ────────────────────────────────────────────────────────

class TestGeometryAccessors:
    def test_centers_shape(self, half_set_2d):
        c = half_set_2d.centers()
        assert c.shape == (8, 2)

    def test_centers_dtype(self, half_set_2d):
        assert half_set_2d.centers().dtype == F64

    def test_centers_c_contiguous(self, half_set_2d):
        assert half_set_2d.centers().flags["C_CONTIGUOUS"]

    def test_centers_all_inside_domain(self, full_set_2d):
        centres = full_set_2d.centers()
        domain = full_set_2d.partition.domain
        for c in centres:
            assert domain.contains_point(c), f"centre {c} not in domain"

    def test_centers_consistent_with_key_to_box(self, half_set_2d):
        batch_centres = half_set_2d.centers()
        for i, k in enumerate(half_set_2d.keys):
            expected = half_set_2d.partition.key_to_box(int(k)).center
            assert np.allclose(batch_centres[i], expected)

    def test_centers_empty_set(self, partition_2d):
        s = BoxSet.empty(partition_2d)
        c = s.centers()
        assert c.shape == (0, 2)

    def test_cell_radius_value(self, partition_2d):
        s = BoxSet.full(partition_2d)
        assert np.allclose(s.cell_radius(), [0.25, 0.25])

    def test_bounds_full_set_equals_domain(self, full_set_2d):
        lo, hi = full_set_2d.bounds()
        assert np.allclose(lo, full_set_2d.partition.domain.lo)
        assert np.allclose(hi, full_set_2d.partition.domain.hi)

    def test_bounds_empty_set(self, partition_2d):
        s = BoxSet.empty(partition_2d)
        lo, hi = s.bounds()
        assert np.all(lo == np.inf)
        assert np.all(hi == -np.inf)


# ── Subdivision ───────────────────────────────────────────────────────────────

class TestSubdivide:
    def test_subdivide_doubles_partition_dim(self, half_set_2d):
        refined = half_set_2d.subdivide(0)
        assert refined.partition.dims[0] == half_set_2d.partition.dims[0] * 2

    def test_subdivide_doubles_cell_count(self, half_set_2d):
        refined = half_set_2d.subdivide(0)
        assert len(refined) == len(half_set_2d) * 2

    def test_subdivide_spatial_coverage_preserved(self, half_set_2d):
        original_centres = set(map(tuple, half_set_2d.centers().tolist()))
        refined = half_set_2d.subdivide(0)
        refined_bounds_lo, refined_bounds_hi = refined.bounds()
        orig_lo, orig_hi = half_set_2d.bounds()
        assert np.allclose(refined_bounds_lo, orig_lo)
        assert np.allclose(refined_bounds_hi, orig_hi)

    def test_subdivide_all_dims(self, half_set_2d):
        refined = half_set_2d.subdivide_all()
        # 2D: each cell → 4 children
        assert len(refined) == len(half_set_2d) * 4

    def test_subdivide_empty_set(self, partition_2d):
        empty = BoxSet.empty(partition_2d)
        refined = empty.subdivide(0)
        assert len(refined) == 0


# ── Equality ──────────────────────────────────────────────────────────────────

class TestEquality:
    def test_equal_sets(self, partition_2d):
        s1 = BoxSet(partition_2d, [0, 1, 2])
        s2 = BoxSet(partition_2d, [2, 1, 0])  # same keys, different order
        assert s1 == s2

    def test_unequal_keys(self, partition_2d):
        s1 = BoxSet(partition_2d, [0, 1, 2])
        s2 = BoxSet(partition_2d, [0, 1, 3])
        assert s1 != s2


# ── Direct ports of GAIO.jl boxset.jl tests ──────────────────────────────────

class TestGAIOJL:
    """
    Exact Python equivalents of every assertion in GAIO.jl/test/boxset.jl.

    GAIO.jl tests both BoxGrid (regular) and BoxTree partitions with identical
    assertions; we only have the regular partition so far — these tests cover
    that branch.  A BoxTree port is a Phase 2 stretch goal.

    API translation
    ---------------
    * ``BoxSet(partition)``         → ``BoxSet.empty(partition)``
    * ``cover(partition, :)``       → ``BoxSet.full(partition)``
    * ``cover(partition, points)``  → ``BoxSet.cover(partition, points)``
    * ``cover(partition, box)``     → ``BoxSet.from_box(partition, box)``
    * ``subdivide(B, dim)``         → ``B.subdivide(dim - 1)``  (0-indexed)
    * ``union/intersect/setdiff``   → ``|  &  -``  (all return new BoxSet)
    * In-place ``union!/intersect!/setdiff!`` have no equivalent — BoxSets
      are immutable by design (see ARCHITECTURE.md §D3).

    Partition used in GAIO.jl: BoxGrid with domain centre=(0,0,0)
    radius=(1,1,1), dims=(16,8,8).  Total cells = 16×8×8 = 1024 = 2^10.
    """

    @pytest.fixture
    def partition_3d_1688(self):
        """The exact partition used in GAIO.jl boxset.jl."""
        domain = Box(np.zeros(3), np.ones(3))
        return BoxPartition(domain, np.array([16, 8, 8], dtype=I64))

    # Named test points (mirror GAIO.jl)
    p1 = np.array([ 0.5,  0.5,  0.5])
    p2 = np.array([-0.5,  0.5,  0.5])
    p3 = np.array([ 0.5, -0.5,  0.5])
    p4 = np.array([ 0.51, 0.51, 0.51])  # same cell as p1 on the 16×8×8 grid

    # -- exported functionality / empty ---------------------------------------

    @pytest.mark.gaio_jl
    def test_empty_is_empty(self, partition_3d_1688):
        """boxset.jl @testset "empty" """
        empty = BoxSet.empty(partition_3d_1688)
        assert empty.is_empty()
        assert len(empty) == 0

    @pytest.mark.gaio_jl
    def test_empty_subdivide_stays_empty(self, partition_3d_1688):
        """boxset.jl @testset "empty" — subdivide(empty_boxset, 1)"""
        empty = BoxSet.empty(partition_3d_1688)
        refined = empty.subdivide(0)          # dim 0 ≡ Julia dim 1
        assert refined.is_empty()
        assert len(refined) == 0

    # -- exported functionality / full ----------------------------------------

    @pytest.mark.gaio_jl
    def test_full_length_is_2_pow_10(self, partition_3d_1688):
        """boxset.jl @testset "full" — length(full_boxset) == 2^10"""
        full = BoxSet.full(partition_3d_1688)
        assert not full.is_empty()
        assert len(full) == 2**10        # 16*8*8 = 1024

    @pytest.mark.gaio_jl
    def test_full_subdivide_length_is_2_pow_11(self, partition_3d_1688):
        """boxset.jl @testset "full" — length after subdivide == 2^11"""
        full    = BoxSet.full(partition_3d_1688)
        refined = full.subdivide(0)      # dim 0 ≡ Julia dim 1
        assert not refined.is_empty()
        assert len(refined) == 2**11

    # -- exported functionality / boxsets created on points ------------------

    @pytest.mark.gaio_jl
    def test_cover_four_points_gives_three_cells(self, partition_3d_1688):
        """
        boxset.jl @testset "boxsets created on points"

        p4 = (0.51, 0.51, 0.51) falls in the same 16×8×8 cell as
        p1 = (0.5, 0.5, 0.5), so covering all four points yields 3 cells.
        """
        pts = np.array([self.p1, self.p2, self.p3, self.p4])
        box_set = BoxSet.cover(partition_3d_1688, pts)
        assert len(box_set) == 3

    @pytest.mark.gaio_jl
    def test_cover_points_each_point_in_some_box(self, partition_3d_1688):
        """boxset.jl — any(box -> pᵢ ∈ box, box_set) for each point"""
        pts = np.array([self.p1, self.p2, self.p3, self.p4])
        box_set = BoxSet.cover(partition_3d_1688, pts)
        for p in (self.p1, self.p2, self.p3, self.p4):
            assert any(p in box for _, box in box_set.boxes()), \
                f"Point {p} not found in any box of the set"

    @pytest.mark.gaio_jl
    @pytest.mark.slow
    def test_cover_subdivide_10_times_length(self, partition_3d_1688):
        """
        boxset.jl — after n=10 cycled subdivisions, length == 3 * 2^n.

        Julia cycles (k%3)+1 (1-indexed dims 1,2,3).
        Python cycles k%3   (0-indexed dims 0,1,2) — same 3 of the 3 dims.
        """
        pts = np.array([self.p1, self.p2, self.p3, self.p4])
        box_set = BoxSet.cover(partition_3d_1688, pts)
        n = 10
        for k in range(n):
            box_set = box_set.subdivide(k % 3)
        assert len(box_set) == 3 * (2**n)

    @pytest.mark.gaio_jl
    @pytest.mark.slow
    def test_cover_subdivide_10_times_points_still_in_boxes(self, partition_3d_1688):
        """
        boxset.jl — any(box -> pᵢ ∈ box, box_set) still true after subdivision.
        """
        pts = np.array([self.p1, self.p2, self.p3, self.p4])
        box_set = BoxSet.cover(partition_3d_1688, pts)
        for k in range(10):
            box_set = box_set.subdivide(k % 3)
        for p in (self.p1, self.p2, self.p3, self.p4):
            assert any(p in box for _, box in box_set.boxes()), \
                f"Point {p} not in any box after 10 subdivisions"

    # -- exported functionality / boxsets created on boxes -------------------

    @pytest.mark.gaio_jl
    def test_cover_from_box_equals_cover_from_point(self, partition_3d_1688):
        """
        boxset.jl — cover(partition, point_to_box(partition, p1)) == cover(partition, p1)
        """
        key    = partition_3d_1688.point_to_key(self.p1)
        cell   = partition_3d_1688.key_to_box(key)
        from_box   = BoxSet.from_box(partition_3d_1688, cell)
        from_point = BoxSet.cover(partition_3d_1688, self.p1[None, :])
        assert from_box == from_point

    @pytest.mark.gaio_jl
    def test_cover_from_boxes_equals_cover_from_points(self, partition_3d_1688):
        """
        boxset.jl — cover(partition, [boxes...]) == cover(partition, (p1,p2,p3,p4))
        """
        pts = np.array([self.p1, self.p2, self.p3, self.p4])
        from_points = BoxSet.cover(partition_3d_1688, pts)

        # Build cover from the individual cell-boxes of each point
        keys = [partition_3d_1688.point_to_key(p) for p in pts]
        cells = [partition_3d_1688.key_to_box(k) for k in keys]
        from_boxes = BoxSet.empty(partition_3d_1688)
        for cell in cells:
            from_boxes = from_boxes | BoxSet.from_box(partition_3d_1688, cell)

        assert from_boxes == from_points

    # -- exported functionality / set operations ------------------------------

    @pytest.mark.gaio_jl
    def test_set_operations_exact_counts(self, partition_3d_1688):
        """
        boxset.jl @testset "set operations"

        All length assertions from the Julia test, translated 1-to-1.
        Note: Julia's in-place union!/intersect!/setdiff! are not ported
        because BoxSets are immutable by design (see ARCHITECTURE.md §D3).
        """
        p1_set    = BoxSet.cover(partition_3d_1688, self.p1[None, :])
        p1p2_set  = BoxSet.cover(partition_3d_1688,
                                  np.array([self.p1, self.p2]))
        p3_set    = BoxSet.cover(partition_3d_1688, self.p3[None, :])
        p2p3_set  = BoxSet.cover(partition_3d_1688,
                                  np.array([self.p2, self.p3]))

        assert len(p1_set)   == 1
        assert len(p1p2_set) == 2
        assert len(p3_set)   == 1
        assert len(p2p3_set) == 2

        # union
        assert len(p1_set   | p3_set)   == 2
        assert len(p1_set   | p1p2_set) == 2
        assert len(p1p2_set | p2p3_set) == 3

        # intersect
        assert len(p1_set   & p3_set)   == 0
        assert len(p1_set   & p1p2_set) == 1
        assert len(p2p3_set & p2p3_set) == 2

        # setdiff
        assert len(p1_set   - p3_set)   == 1
        assert len(p1_set   - p1p2_set) == 0
        assert len(p1p2_set - p1_set)   == 1

    # -- exported functionality / accessing boxes ----------------------------

    @pytest.mark.gaio_jl
    def test_accessing_boxes_types_and_shapes(self, partition_3d_1688):
        """
        boxset.jl @testset "accessing boxes"

        Julia: collect(B) isa Vector{Box{3,Float64}} with length 3.
               centers matrix has shape (3, 3).
        Python: list(box_set.boxes()) gives (key, Box) pairs.
                centers() returns (N, ndim) = (3, 3) float64 array.
        """
        pts     = np.array([self.p1, self.p2, self.p3, self.p4])
        box_set = BoxSet.cover(partition_3d_1688, pts)

        # collect boxes
        boxes = [box for _, box in box_set.boxes()]
        assert len(boxes) == 3
        assert all(isinstance(b, Box) for b in boxes)

        # centres matrix shape: (3 cells, 3 dimensions)
        centres = box_set.centers()
        assert centres.shape == (3, 3)
        assert centres.dtype == np.float64
