"""
Tests for gaio/core/boxmeasure.py — BoxMeasure.

Two categories:
  A. Behavioural (randomized / property-based)
  B. Exact ports of Julia's test/boxmeasure.jl
"""
import numpy as np
import pytest

from gaio.core.box import Box
from gaio.core.partition import BoxPartition
from gaio.core.boxset import BoxSet
from gaio.core.boxmeasure import BoxMeasure


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def setup_measure():
    """
    Exact replica of Julia's boxmeasure.jl fixture:

        domain   = Box((0.0, 0.0), (1.0, 1.0))   # center (0,0), radius (1,1)
        partition = BoxGrid(domain, (16, 8))        # 16×8 = 128 cells
        left  = cells covering [-1, 0] × [-1, 1]   (64 cells)
        right = cells covering [ 0, 1] × [-1, 1]   (64 cells)
        full  = all 128 cells
        scale = volume(domain) / (2 * 64) = 4 / 128 = 1/32

    Each cell in left/right/full gets weight = scale.
    """
    domain = Box([0.0, 0.0], [1.0, 1.0])
    partition = BoxPartition(domain, [16, 8])
    # left half: x ∈ [-1, 0)
    left_box = Box([-0.5, 0.0], [0.5, 1.0])   # center (-0.5, 0), r (0.5, 1)
    right_box = Box([0.5, 0.0], [0.5, 1.0])   # center ( 0.5, 0), r (0.5, 1)

    left = BoxSet.from_box(partition, left_box)
    right = BoxSet.from_box(partition, right_box)
    full = BoxSet.full(partition)

    n = len(right)
    vol = float(np.prod(2.0 * domain.radius))   # 4.0
    scale = vol / (2 * n)

    mu_left = BoxMeasure.from_boxset(left, np.full(len(left), scale))
    mu_right = BoxMeasure.from_boxset(right, np.full(len(right), scale))
    mu_full = BoxMeasure.from_boxset(full, np.full(len(full), scale))

    return domain, partition, left, right, full, n, scale, mu_left, mu_right, mu_full


# ===========================================================================
# A. Behavioural tests
# ===========================================================================

class TestBoxMeasureConstruction:
    def test_uniform_constructor(self):
        domain = Box([0.0, 0.0], [1.0, 1.0])
        P = BoxPartition(domain, [4, 4])
        B = BoxSet.full(P)
        mu = BoxMeasure.uniform(B)
        assert len(mu) == 16
        np.testing.assert_array_equal(mu.weights, np.ones(16))

    def test_zeros_constructor(self):
        domain = Box([0.0, 0.0], [1.0, 1.0])
        P = BoxPartition(domain, [4, 4])
        B = BoxSet.full(P)
        mu = BoxMeasure.zeros(B)
        assert len(mu) == 16
        np.testing.assert_array_equal(mu.weights, np.zeros(16))

    def test_from_boxset_with_weights(self):
        domain = Box([0.0, 0.0], [1.0, 1.0])
        P = BoxPartition(domain, [4, 4])
        B = BoxSet.full(P)
        w = np.arange(16, dtype=float)
        mu = BoxMeasure.from_boxset(B, w)
        assert len(mu) == 16
        np.testing.assert_array_equal(mu.weights, w)

    def test_weight_length_mismatch_raises(self):
        domain = Box([0.0, 0.0], [1.0, 1.0])
        P = BoxPartition(domain, [4, 4])
        B = BoxSet.full(P)
        with pytest.raises(ValueError):
            BoxMeasure(P, B._keys, np.ones(5))

    def test_keys_are_sorted(self):
        domain = Box([0.0, 0.0], [1.0, 1.0])
        P = BoxPartition(domain, [4, 4])
        B = BoxSet.full(P)
        mu = BoxMeasure.uniform(B)
        assert np.all(mu.keys[:-1] <= mu.keys[1:])


class TestBoxMeasureAccess:
    def test_getitem_in_support(self):
        domain = Box([0.0, 0.0], [1.0, 1.0])
        P = BoxPartition(domain, [4, 4])
        B = BoxSet(P, [5, 7, 11])
        mu = BoxMeasure.from_boxset(B, [1.0, 2.0, 3.0])
        assert mu[5] == pytest.approx(1.0)
        assert mu[7] == pytest.approx(2.0)
        assert mu[11] == pytest.approx(3.0)

    def test_getitem_outside_support_returns_zero(self):
        domain = Box([0.0, 0.0], [1.0, 1.0])
        P = BoxPartition(domain, [4, 4])
        B = BoxSet(P, [5])
        mu = BoxMeasure.from_boxset(B, [1.0])
        assert mu[0] == pytest.approx(0.0)
        assert mu[3] == pytest.approx(0.0)

    def test_setitem_updates_weight(self):
        domain = Box([0.0, 0.0], [1.0, 1.0])
        P = BoxPartition(domain, [4, 4])
        B = BoxSet(P, [5])
        mu = BoxMeasure.from_boxset(B, [1.0])
        mu[5] = 99.0
        assert mu[5] == pytest.approx(99.0)

    def test_setitem_outside_support_raises(self):
        domain = Box([0.0, 0.0], [1.0, 1.0])
        P = BoxPartition(domain, [4, 4])
        B = BoxSet(P, [5])
        mu = BoxMeasure.from_boxset(B, [1.0])
        with pytest.raises(KeyError):
            mu[0] = 1.0

    def test_to_boxset_excludes_zeros(self):
        domain = Box([0.0, 0.0], [1.0, 1.0])
        P = BoxPartition(domain, [4, 4])
        B = BoxSet.full(P)
        # All zeros except keys 3 and 7
        w = np.zeros(16)
        w[3] = 1.0
        w[7] = 2.0
        mu = BoxMeasure(P, B._keys, w)
        bs = mu.to_boxset()
        assert set(bs._keys.tolist()) == {B._keys[3], B._keys[7]}


class TestBoxMeasureArithmetic:
    def test_negation(self):
        domain = Box([0.0, 0.0], [1.0, 1.0])
        P = BoxPartition(domain, [4, 4])
        B = BoxSet.full(P)
        mu = BoxMeasure.uniform(B, 2.0)
        neg = -mu
        np.testing.assert_array_equal(neg.weights, -2.0 * np.ones(16))

    def test_scalar_multiply(self):
        domain = Box([0.0, 0.0], [1.0, 1.0])
        P = BoxPartition(domain, [4, 4])
        B = BoxSet.full(P)
        mu = BoxMeasure.uniform(B, 3.0)
        result = 2.0 * mu
        np.testing.assert_array_equal(result.weights, 6.0 * np.ones(16))

    def test_scalar_divide(self):
        domain = Box([0.0, 0.0], [1.0, 1.0])
        P = BoxPartition(domain, [4, 4])
        B = BoxSet.full(P)
        mu = BoxMeasure.uniform(B, 6.0)
        result = mu / 3.0
        np.testing.assert_array_equal(result.weights, 2.0 * np.ones(16))

    def test_different_partition_add_raises(self):
        d = Box([0.0, 0.0], [1.0, 1.0])
        P1 = BoxPartition(d, [4, 4])
        P2 = BoxPartition(d, [2, 2])
        mu1 = BoxMeasure.uniform(BoxSet.full(P1))
        mu2 = BoxMeasure.uniform(BoxSet.full(P2))
        with pytest.raises(ValueError):
            _ = mu1 + mu2


class TestBoxMeasureIntegration:
    def test_total_mass(self):
        domain = Box([0.0, 0.0], [1.0, 1.0])
        P = BoxPartition(domain, [4, 4])
        B = BoxSet.full(P)
        mu = BoxMeasure.uniform(B, 2.0)
        assert mu.total() == pytest.approx(32.0)

    def test_integrate_constant(self):
        domain = Box([0.0, 0.0], [1.0, 1.0])
        P = BoxPartition(domain, [4, 4])
        B = BoxSet.full(P)
        mu = BoxMeasure.uniform(B, 1.0)
        # integrate(f=3) = sum over cells of 3 * 1 = 48
        result = mu.integrate(lambda x: 3.0)
        assert result == pytest.approx(48.0)

    def test_call_total_mass(self):
        domain = Box([0.0, 0.0], [1.0, 1.0])
        P = BoxPartition(domain, [4, 4])
        B = BoxSet.full(P)
        mu = BoxMeasure.uniform(B, 1.5)
        assert mu() == pytest.approx(24.0)

    def test_call_restricted_to_boxset(self):
        domain = Box([0.0, 0.0], [1.0, 1.0])
        P = BoxPartition(domain, [4, 4])
        full = BoxSet.full(P)
        mu = BoxMeasure.uniform(full, 1.0)
        # Restrict to first 4 cells
        sub = BoxSet(P, full._keys[:4])
        assert mu(sub) == pytest.approx(4.0)


class TestBoxMeasureNorm:
    def test_normalize_produces_unit_l2_norm(self):
        domain = Box([0.0, 0.0], [1.0, 1.0])
        P = BoxPartition(domain, [4, 4])
        B = BoxSet.full(P)
        mu = BoxMeasure.uniform(B, 2.0)
        normed = mu.normalize()
        assert normed.norm(2) == pytest.approx(1.0, rel=1e-10)

    def test_normalize_zero_measure_raises(self):
        domain = Box([0.0, 0.0], [1.0, 1.0])
        P = BoxPartition(domain, [4, 4])
        B = BoxSet.full(P)
        mu = BoxMeasure.zeros(B)
        with pytest.raises(ValueError):
            mu.normalize()

    def test_density_callable(self):
        domain = Box([0.0, 0.0], [1.0, 1.0])
        P = BoxPartition(domain, [4, 4])
        B = BoxSet.full(P)
        vol = P.cell_volume
        mu = BoxMeasure.uniform(B, vol)  # density = 1 everywhere
        g = mu.density()
        assert g(np.array([0.0, 0.0])) == pytest.approx(1.0)

    def test_density_outside_domain_is_zero(self):
        domain = Box([0.0, 0.0], [1.0, 1.0])
        P = BoxPartition(domain, [4, 4])
        B = BoxSet.full(P)
        mu = BoxMeasure.uniform(B)
        g = mu.density()
        # Outside the domain
        assert g(np.array([5.0, 5.0])) == pytest.approx(0.0)


# ===========================================================================
# B. Julia exact-port tests  (test/boxmeasure.jl)
# ===========================================================================

class TestBoxMeasureJuliaPort:
    """
    Port of Julia's test/boxmeasure.jl.

    domain   = Box((0.0,0.0),(1.0,1.0))  → [-1,1]² (vol=4)
    partition = BoxGrid(domain, (16,8))   → 128 cells
    left  = cover(partition, Box((-0.5,0.0),(0.5,1.0)))  → 64 cells
    right = cover(partition, Box(( 0.5,0.0),(0.5,1.0)))  → 64 cells
    scale = 4 / 128 = 1/32
    """

    def test_lengths(self, setup_measure):
        domain, P, left, right, full, n, scale, mu_l, mu_r, mu_f = setup_measure
        assert len(left) == n
        assert len(right) == n
        assert len(full) == 2 * n

    def test_addition_left_plus_right_equals_full(self, setup_measure):
        """μ_left + μ_right == μ_full"""
        domain, P, left, right, full, n, scale, mu_l, mu_r, mu_f = setup_measure
        result = mu_l + mu_r
        assert result == mu_f

    def test_subtraction(self, setup_measure):
        """μ_full - μ_left == μ_right"""
        domain, P, left, right, full, n, scale, mu_l, mu_r, mu_f = setup_measure
        result = mu_f - mu_l
        assert result == mu_r

    def test_subtraction_negative(self, setup_measure):
        """μ_left - μ_full == -μ_right"""
        domain, P, left, right, full, n, scale, mu_l, mu_r, mu_f = setup_measure
        result = mu_l - mu_f
        assert result == -mu_r

    def test_double_plus_equals_double_full(self, setup_measure):
        """2*μ_left + 2*μ_right == μ_full + μ_full"""
        domain, P, left, right, full, n, scale, mu_l, mu_r, mu_f = setup_measure
        lhs = 2 * mu_l + 2 * mu_r
        rhs = mu_f + mu_f
        assert lhs == rhs

    def test_half_sum_equals_half_full(self, setup_measure):
        """μ_left/2 + μ_right/2 == μ_full/2"""
        domain, P, left, right, full, n, scale, mu_l, mu_r, mu_f = setup_measure
        lhs = mu_l / 2 + mu_r / 2
        rhs = mu_f / 2
        assert lhs == rhs

    def test_integration_total_mass_equals_volume(self, setup_measure):
        """μ_full(domain) == volume(domain)"""
        domain, P, left, right, full, n, scale, mu_l, mu_r, mu_f = setup_measure
        vol = float(np.prod(2.0 * domain.radius))  # 4.0
        assert mu_f() == pytest.approx(vol)

    def test_integration_constant_function(self, setup_measure):
        """sum(x->2, μ_full) == 2 * volume(domain)"""
        domain, P, left, right, full, n, scale, mu_l, mu_r, mu_f = setup_measure
        vol = float(np.prod(2.0 * domain.radius))
        result = mu_f.integrate(lambda x: 2.0)
        assert result == pytest.approx(2.0 * vol)

    def test_scaled_measure_total_mass(self, setup_measure):
        """(2*μ_full)(domain) == 2*volume(domain)"""
        domain, P, left, right, full, n, scale, mu_l, mu_r, mu_f = setup_measure
        vol = float(np.prod(2.0 * domain.radius))
        assert (2 * mu_f)() == pytest.approx(2.0 * vol)
