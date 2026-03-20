"""
Tests for gaio.core.Box.

Each test group maps to one responsibility of the Box class so that
a failing test immediately identifies which behaviour broke.
"""
import numpy as np
import pytest

from gaio import Box, F64


# ── Construction ──────────────────────────────────────────────────────────────

class TestConstruction:
    def test_basic_2d(self):
        b = Box([0.0, 0.0], [1.0, 1.0])
        assert b.ndim == 2
        assert b.center.dtype == F64
        assert b.radius.dtype == F64

    def test_arrays_are_c_contiguous(self):
        b = Box([1.0, 2.0], [3.0, 4.0])
        assert b.center.flags["C_CONTIGUOUS"]
        assert b.radius.flags["C_CONTIGUOUS"]

    def test_integer_inputs_promoted_to_float64(self):
        b = Box([0, 0], [1, 1])
        assert b.center.dtype == F64
        assert b.radius.dtype == F64

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            Box([0.0, 0.0], [1.0])

    def test_zero_radius_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            Box([0.0, 0.0], [1.0, 0.0])

    def test_negative_radius_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            Box([0.0], [-1.0])

    def test_2d_array_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            Box([[0.0, 0.0]], [[1.0, 1.0]])

    def test_high_dimensional(self):
        n = 10
        b = Box(np.zeros(n), np.ones(n))
        assert b.ndim == n


# ── Derived properties ────────────────────────────────────────────────────────

class TestProperties:
    def test_lo_hi(self, domain_2d):
        b = domain_2d  # center=[0,0], radius=[1,1]
        assert np.allclose(b.lo, [-1.0, -1.0])
        assert np.allclose(b.hi, [1.0, 1.0])

    def test_volume_2d(self, domain_2d):
        assert domain_2d.volume == pytest.approx(4.0)

    def test_volume_3d(self, domain_3d):
        assert domain_3d.volume == pytest.approx(8.0)

    def test_volume_scales_with_radius(self):
        b = Box([0.0, 0.0], [2.0, 3.0])
        assert b.volume == pytest.approx(24.0)   # 2*2 * 2*3

    def test_widths(self, domain_2d):
        assert np.allclose(domain_2d.widths, [2.0, 2.0])


# ── Point containment ─────────────────────────────────────────────────────────

class TestContainsPoint:
    def test_interior_point(self, domain_2d):
        assert domain_2d.contains_point(np.array([0.5, -0.5]))

    def test_centre_is_inside(self, domain_2d):
        assert domain_2d.contains_point(np.array([0.0, 0.0]))

    def test_lower_boundary_inclusive(self, domain_2d):
        # lo = [-1, -1] is inside (half-open: [lo, hi))
        assert domain_2d.contains_point(np.array([-1.0, -1.0]))

    def test_upper_boundary_exclusive(self, domain_2d):
        # hi = [1, 1] is outside
        assert not domain_2d.contains_point(np.array([1.0, 1.0]))

    def test_upper_boundary_one_component(self, domain_2d):
        # x=1.0 is on the right edge → outside
        assert not domain_2d.contains_point(np.array([1.0, 0.0]))

    def test_exterior_point(self, domain_2d):
        assert not domain_2d.contains_point(np.array([2.0, 0.0]))

    def test_in_operator(self, domain_2d):
        assert np.array([0.0, 0.0]) in domain_2d
        assert np.array([2.0, 0.0]) not in domain_2d


# ── Box containment ───────────────────────────────────────────────────────────

class TestContainsBox:
    def test_self_contains_self(self, domain_2d):
        assert domain_2d.contains_box(domain_2d)

    def test_larger_contains_smaller(self, domain_2d):
        small = Box([0.0, 0.0], [0.5, 0.5])
        assert domain_2d.contains_box(small)

    def test_smaller_does_not_contain_larger(self, domain_2d):
        small = Box([0.0, 0.0], [0.5, 0.5])
        assert not small.contains_box(domain_2d)

    def test_partial_overlap_not_contained(self, domain_2d):
        shifted = Box([0.5, 0.5], [1.0, 1.0])  # extends beyond domain_2d
        assert not domain_2d.contains_box(shifted)


# ── Intersection ──────────────────────────────────────────────────────────────

class TestIntersects:
    def test_overlapping_boxes_intersect(self, domain_2d):
        other = Box([0.5, 0.5], [1.0, 1.0])
        assert domain_2d.intersects(other)

    def test_touching_boxes_do_not_intersect(self):
        # b1 covers [-1, 0) and b2 covers [0, 1): they share only the boundary
        b1 = Box([-0.5, 0.0], [0.5, 1.0])
        b2 = Box([0.5, 0.0], [0.5, 1.0])
        assert not b1.intersects(b2)

    def test_disjoint_boxes_do_not_intersect(self):
        b1 = Box([-2.0, 0.0], [0.5, 1.0])
        b2 = Box([2.0, 0.0], [0.5, 1.0])
        assert not b1.intersects(b2)


class TestIntersection:
    def test_intersection_geometry(self):
        b1 = Box([0.0, 0.0], [2.0, 2.0])   # [-2,2] × [-2,2]
        b2 = Box([1.0, 1.0], [2.0, 2.0])   # [-1,3] × [-1,3]
        inter = b1.intersection(b2)
        assert np.allclose(inter.lo, [-1.0, -1.0])
        assert np.allclose(inter.hi, [2.0, 2.0])

    def test_non_intersecting_raises(self):
        b1 = Box([-2.0, 0.0], [0.5, 1.0])
        b2 = Box([2.0, 0.0], [0.5, 1.0])
        with pytest.raises(ValueError, match="do not intersect"):
            b1.intersection(b2)

    def test_and_operator(self, domain_2d):
        other = Box([0.5, 0.5], [1.0, 1.0])
        result = domain_2d & other
        assert isinstance(result, Box)


# ── Bounding box ──────────────────────────────────────────────────────────────

class TestBoundingBox:
    def test_self_bounding_box_is_self(self, domain_2d):
        bb = domain_2d.bounding_box(domain_2d)
        assert np.allclose(bb.center, domain_2d.center)
        assert np.allclose(bb.radius, domain_2d.radius)

    def test_bounding_box_of_two_boxes(self):
        b1 = Box([-1.0, 0.0], [1.0, 1.0])  # [-2,0] × [-1,1]
        b2 = Box([1.0, 0.0], [1.0, 1.0])   # [0,2]  × [-1,1]
        bb = b1.bounding_box(b2)
        assert np.allclose(bb.lo, [-2.0, -1.0])
        assert np.allclose(bb.hi, [2.0, 1.0])

    def test_or_operator(self, domain_2d):
        other = Box([2.0, 2.0], [1.0, 1.0])
        result = domain_2d | other
        assert isinstance(result, Box)


# ── Subdivision ───────────────────────────────────────────────────────────────

class TestSubdivide:
    def test_subdivide_dim0(self, domain_2d):
        lo, hi = domain_2d.subdivide(0)
        # left child: center at (-0.5, 0), radius (0.5, 1)
        assert np.allclose(lo.center, [-0.5, 0.0])
        assert np.allclose(lo.radius, [0.5, 1.0])
        # right child: center at (0.5, 0), radius (0.5, 1)
        assert np.allclose(hi.center, [0.5, 0.0])
        assert np.allclose(hi.radius, [0.5, 1.0])

    def test_subdivide_children_are_disjoint(self, domain_2d):
        lo, hi = domain_2d.subdivide(0)
        assert not lo.intersects(hi)

    def test_subdivide_children_cover_parent(self, domain_2d):
        lo, hi = domain_2d.subdivide(0)
        bb = lo.bounding_box(hi)
        assert np.allclose(bb.center, domain_2d.center)
        assert np.allclose(bb.radius, domain_2d.radius)

    def test_subdivide_preserves_volume(self, domain_2d):
        lo, hi = domain_2d.subdivide(0)
        assert (lo.volume + hi.volume) == pytest.approx(domain_2d.volume)

    def test_subdivide_all_count(self, domain_2d):
        children = domain_2d.subdivide_all()
        assert len(children) == 4  # 2² in 2D

    def test_subdivide_all_count_3d(self, domain_3d):
        children = domain_3d.subdivide_all()
        assert len(children) == 8  # 2³ in 3D

    def test_subdivide_all_preserves_total_volume(self, domain_2d):
        children = domain_2d.subdivide_all()
        total = sum(c.volume for c in children)
        assert total == pytest.approx(domain_2d.volume)


# ── Coordinate transforms ─────────────────────────────────────────────────────

class TestCoordinateTransforms:
    def test_rescale_origin(self, domain_2d):
        # rescale(0) should return the center
        result = domain_2d.rescale(np.zeros(2))
        assert np.allclose(result, domain_2d.center)

    def test_rescale_unit_corners(self, domain_2d):
        # rescale([1, 1]) should return hi - eps (upper-right corner)
        result = domain_2d.rescale(np.ones(2))
        assert np.allclose(result, domain_2d.hi)

    def test_rescale_normalize_roundtrip(self, domain_2d):
        pts = np.array([[0.3, -0.7], [-0.5, 0.9]])
        for p in pts:
            assert np.allclose(domain_2d.normalize(domain_2d.rescale(p)), p)

    def test_rescale_batch(self, domain_2d):
        unit_pts = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
        result = domain_2d.rescale(unit_pts)
        assert result.shape == (3, 2)


# ── Equality and hashing ──────────────────────────────────────────────────────

class TestEquality:
    def test_equal_boxes(self, domain_2d):
        b2 = Box(np.zeros(2), np.ones(2))
        assert domain_2d == b2

    def test_unequal_center(self, domain_2d):
        b2 = Box([0.1, 0.0], [1.0, 1.0])
        assert domain_2d != b2

    def test_unequal_radius(self, domain_2d):
        b2 = Box([0.0, 0.0], [1.0, 2.0])
        assert domain_2d != b2

    def test_hashable(self, domain_2d):
        s = {domain_2d, domain_2d}
        assert len(s) == 1


# ── Direct ports of GAIO.jl box.jl tests ─────────────────────────────────────
# Each test below has a comment citing the exact @testset name in GAIO.jl so
# the correspondence can always be verified against the Julia source.

class TestGAIOJL:
    """
    Exact Python equivalents of every assertion in GAIO.jl/test/box.jl.
    Marked with @pytest.mark.gaio_jl so they can be run in isolation:
        pytest -m gaio_jl
    """

    # -- exported functionality / basics --------------------------------------

    @pytest.mark.gaio_jl
    def test_basics_center_and_radius_stored(self):
        """box.jl @testset "basics" """
        center = np.array([0.0, 0.1])
        radius = np.array([10.0, 10.0])
        box = Box(center, radius)
        assert np.array_equal(box.center, center)
        assert np.array_equal(box.radius, radius)

    # -- exported functionality / types ---------------------------------------

    @pytest.mark.gaio_jl
    def test_types_mixed_int_float_promotes(self):
        """
        box.jl @testset "types"

        Julia: center=SVector(0,0,1) (Int), radius=SVector(1.0,0.1,1.0) (Float).
        The Box constructor should promote center to float so both fields share
        the same element type.  In our implementation both are always float64.
        """
        center = np.array([0, 0, 1])       # integer dtype
        radius = np.array([1.0, 0.1, 1.0]) # float dtype
        box = Box(center, radius)
        # Both must be the same dtype (float64) — center was promoted
        assert box.center.dtype == box.radius.dtype == np.float64
        # And center must no longer be integer
        assert box.center.dtype != np.array([0, 0, 1]).dtype

    # -- exported functionality / containment ---------------------------------

    @pytest.mark.gaio_jl
    def test_containment_exact_cases(self):
        """box.jl @testset "containment" — verbatim point list from GAIO.jl"""
        center = np.array([0.0, 0.0, 0.0])
        radius = np.array([1.0, 1.0, 1.0])
        box = Box(center, radius)

        inside             = np.array([ 0.5,  0.5,  0.5])
        left               = np.array([-1.0, -1.0, -1.0])   # lo corner
        right              = np.array([ 1.0,  1.0,  1.0])   # hi corner
        on_boundary_left   = np.array([ 0.0,  0.0, -1.0])   # z at lo
        on_boundary_right  = np.array([ 0.0,  1.0,  0.0])   # y at hi
        outside_left       = np.array([ 0.0,  0.0, -2.0])
        outside_right      = np.array([ 0.0,  2.0,  0.0])

        # Boxes are half-open: [lo, hi)
        assert inside           in box
        assert box.center       in box
        assert left             in box       # lo boundary is INCLUSIVE
        assert right        not in box       # hi boundary is EXCLUSIVE
        assert on_boundary_left     in box   # z=-1 is the lo boundary → in
        assert on_boundary_right not in box  # y=1 is the hi boundary → out
        assert outside_left     not in box
        assert outside_right    not in box

    # -- exported functionality / non matching dimensions --------------------

    @pytest.mark.gaio_jl
    def test_non_matching_dimensions_raises(self):
        """box.jl @testset "non matching dimensions" """
        center = np.array([0.0, 0.0, 0.0])
        radius = np.array([1.0, 1.0])
        with pytest.raises((ValueError, Exception)):
            Box(center, radius)

    # -- exported functionality / nonpositive radii ---------------------------

    @pytest.mark.gaio_jl
    def test_nonpositive_radius_negative_raises(self):
        """box.jl @testset "nonpositive radii" — negative component"""
        center = np.array([0.0, 0.0])
        radius = np.array([1.0, -1.0])
        with pytest.raises(ValueError):
            Box(center, radius)

    @pytest.mark.gaio_jl
    def test_nonpositive_radius_zero_raises(self):
        """box.jl @testset "nonpositive radii" — zero component"""
        center = np.array([0.0, 0.0])
        radius = np.array([1.0, 0.0])
        with pytest.raises(ValueError):
            Box(center, radius)

    # -- internal functionality / integer point in box ------------------------

    @pytest.mark.gaio_jl
    def test_integer_point_in_box(self):
        """box.jl @testset "integer point in box" """
        box = Box(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        point_int_outside = np.array([2, 2])
        point_int_inside  = np.array([0, 0])
        assert point_int_inside  in box
        assert point_int_outside not in box

    # -- internal functionality / DimensionMismatch ---------------------------

    @pytest.mark.gaio_jl
    def test_wrong_dimension_point_raises(self):
        """
        box.jl: @test_throws DimensionMismatch SVector(0.,0.,0.) ∈ box

        A 3-D point tested against a 2-D box must raise an error.
        NumPy raises ValueError when arrays of incompatible shapes are compared.
        """
        box = Box(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        with pytest.raises((ValueError, Exception)):
            _ = np.array([0.0, 0.0, 0.0]) in box
