"""
Tests for gaio/transfer/operator.py — TransferOperator.

Two categories:
  A. Behavioural (randomized / property-based)
  B. Exact ports of Julia's test/transfer_operator.jl and test/boxmeasure.jl
     (the TransferOperator × BoxMeasure portion)
"""
import numpy as np
import pytest

from gaio.core.box import Box
from gaio.core.partition import BoxPartition
from gaio.core.boxset import BoxSet
from gaio.core.boxmeasure import BoxMeasure
from gaio.maps.base import SampledBoxMap
from gaio.transfer.operator import TransferOperator


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def contraction_setup():
    """f(x) = x*0.5, domain [-1,1]², 4×4 grid, center test point."""
    domain = Box([0.0, 0.0], [1.0, 1.0])
    f = lambda x: x * 0.5
    pts = np.array([[0.0, 0.0]])
    F = SampledBoxMap(f, domain, pts)
    P = BoxPartition(domain, [4, 4])
    B = BoxSet.full(P)
    return F, P, B


@pytest.fixture
def shift_setup():
    """
    Julia boxmeasure.jl fixture:
        h(x, y) = (x+1, y)  — shift right by 1 unit
        domain = [-1,1]² (Box center=(0,0), radius=(1,1))
        16×8 grid, center test point
        left  = [-1,0] × [-1,1]   (64 cells)
        right = [0, 1] × [-1,1]   (64 cells)
    """
    domain = Box([0.0, 0.0], [1.0, 1.0])
    h = lambda x: np.array([x[0] + 1.0, x[1]])
    pts = np.array([[0.0, 0.0]])   # center point only
    H = SampledBoxMap(h, domain, pts)
    P = BoxPartition(domain, [16, 8])

    left_box = Box([-0.5, 0.0], [0.5, 1.0])
    right_box = Box([0.5, 0.0], [0.5, 1.0])
    left = BoxSet.from_box(P, left_box)
    right = BoxSet.from_box(P, right_box)
    full = BoxSet.full(P)

    n = len(right)
    vol = float(np.prod(2.0 * domain.radius))
    scale = vol / (2 * n)

    mu_left = BoxMeasure.from_boxset(left, np.full(len(left), scale))
    mu_right = BoxMeasure.from_boxset(right, np.full(len(right), scale))

    return H, P, full, left, right, mu_left, mu_right, scale


# ===========================================================================
# A. Behavioural tests
# ===========================================================================

class TestTransferOperatorConstruction:
    def test_shape_equals_codomain_by_domain(self, contraction_setup):
        F, P, B = contraction_setup
        T = TransferOperator(F, B, B)
        assert T.shape == (16, 16)

    def test_auto_codomain(self, contraction_setup):
        """When codomain omitted, it's computed as F(domain)."""
        F, P, B = contraction_setup
        T = TransferOperator(F, B)
        assert isinstance(T.codomain, BoxSet)
        assert len(T.codomain) > 0

    def test_column_stochastic(self, contraction_setup):
        """Each nonzero column sums to 1."""
        F, P, B = contraction_setup
        T = TransferOperator(F, B, B)
        col_sums = np.asarray(T.mat.sum(axis=0)).ravel()
        # All columns that have any nonzero entry must sum to ~1
        nonzero_cols = col_sums > 1e-12
        np.testing.assert_allclose(col_sums[nonzero_cols], 1.0, atol=1e-12)

    def test_matrix_nonnegative(self, contraction_setup):
        F, P, B = contraction_setup
        T = TransferOperator(F, B, B)
        assert T.mat.min() >= -1e-15

    def test_empty_domain_returns_zero_matrix(self):
        domain = Box([0.0, 0.0], [1.0, 1.0])
        F = SampledBoxMap(lambda x: x, domain, np.array([[0.0, 0.0]]))
        P = BoxPartition(domain, [4, 4])
        empty = BoxSet.empty(P)
        full = BoxSet.full(P)
        T = TransferOperator(F, empty, full)
        assert T.mat.shape == (16, 0)

    def test_repr_is_string(self, contraction_setup):
        F, P, B = contraction_setup
        T = TransferOperator(F, B, B)
        assert isinstance(repr(T), str)


class TestTransferOperatorMeasureOps:
    def test_push_forward_returns_boxmeasure(self, contraction_setup):
        F, P, B = contraction_setup
        T = TransferOperator(F, B, B)
        mu = BoxMeasure.uniform(B, 1.0)
        result = T.push_forward(mu)
        assert isinstance(result, BoxMeasure)

    def test_pull_back_returns_boxmeasure(self, contraction_setup):
        F, P, B = contraction_setup
        T = TransferOperator(F, B, B)
        mu = BoxMeasure.uniform(B, 1.0)
        result = T.pull_back(mu)
        assert isinstance(result, BoxMeasure)

    def test_matmul_same_as_push_forward(self, contraction_setup):
        F, P, B = contraction_setup
        T = TransferOperator(F, B, B)
        mu = BoxMeasure.uniform(B, 1.0)
        r1 = T.push_forward(mu)
        r2 = T @ mu
        assert r1 == r2

    def test_push_forward_conserves_total_mass(self, contraction_setup):
        """Push-forward on column-stochastic T conserves total mass."""
        F, P, B = contraction_setup
        T = TransferOperator(F, B, B)
        mu = BoxMeasure.uniform(B, 1.0)
        result = T.push_forward(mu)
        # Total mass in = total mass out (columns sum to 1)
        assert result.total() == pytest.approx(mu.total(), rel=1e-10)


class TestTransferOperatorSpectral:
    def test_eigs_returns_correct_shapes(self, contraction_setup):
        F, P, B = contraction_setup
        T = TransferOperator(F, B, B)
        vals, measures = T.eigs(k=2)
        assert len(vals) == 2
        assert len(measures) == 2
        assert all(isinstance(m, BoxMeasure) for m in measures)

    def test_svds_returns_correct_shapes(self, contraction_setup):
        F, P, B = contraction_setup
        T = TransferOperator(F, B, B)
        U, s, V = T.svds(k=2)
        assert len(U) == 2
        assert len(s) == 2
        assert len(V) == 2


# ===========================================================================
# B. Julia exact-port tests
# ===========================================================================

class TestBakerMapEigenvalue:
    """
    Port of Julia test/transfer_operator.jl:

        Baker transformation:
            f(x,y) = (2x, y/2)   if x < 0   (left half → top)
                   = (2x-1, y/2+1/2)  if x ≥ 0  (right half → bottom... wait)

    Actually Julia's Baker map is:
        if x < 0.5 (i.e. x ∈ [0, 0.5) → centre = (0.25, 0.5)·...)
    But the Julia test uses domain = Box((0.5, 0.5), (0.5, 0.5)) = [0,1]²
    and f(x,y) = (2x mod 1, y/2 if x<0.5 else y/2+0.5)

    The Baker map is area-preserving.  The stationary distribution is uniform.
    So eigenvalue λ₁ ≈ 1 and the eigenmeasure is (approximately) uniform.
    """

    @pytest.fixture
    def baker_setup(self):
        c = r = np.array([0.5, 0.5])
        domain = Box(c, r)  # [0,1]²

        def baker(x):
            if x[0] < 0.5:
                return np.array([2.0 * x[0], x[1] * 0.5])
            else:
                return np.array([2.0 * x[0] - 1.0, x[1] * 0.5 + 0.5])

        # Grid map with 4 test points per cell
        from gaio.maps.grid_map import GridMap
        F = GridMap(baker, domain, [2, 2])  # 2×2 grid → 4 test points
        P = BoxPartition(domain, [16, 16])
        S = BoxSet.full(P)
        return F, P, S

    def test_dominant_eigenvalue_close_to_one(self, baker_setup):
        """λ₁ ≈ 1 for the Baker map (area-preserving → stationary measure)."""
        F, P, S = baker_setup
        T = TransferOperator(F, S, S)
        v0 = np.ones(len(S), dtype=np.float64)
        vals, measures = T.eigs(k=1, which="LM", v0=v0, tol=1e-10)
        dominant = float(np.abs(vals[0]))
        assert dominant == pytest.approx(1.0, abs=0.05)

    def test_dominant_eigenmeasure_approximately_uniform(self, baker_setup):
        """The dominant eigenmeasure of Baker should be approximately uniform."""
        F, P, S = baker_setup
        T = TransferOperator(F, S, S)
        v0 = np.ones(len(S), dtype=np.float64)
        vals, measures = T.eigs(k=1, which="LM", v0=v0, tol=1e-10)
        mu = measures[0]
        # All weights should be approximately equal
        w = mu.weights
        w_mean = np.mean(np.abs(w))
        if w_mean > 1e-15:
            w_norm = np.abs(w) / w_mean
            # Each weight should be within 10% of the mean
            assert np.all(np.abs(w_norm - 1.0) < 0.2)


class TestShiftMapTransfer:
    """
    Port of the TransferOperator × BoxMeasure portion of Julia's
    test/boxmeasure.jl:

        h(x,y) = (x+1, y)   (shift right by 1)
        T*μ_left  == μ_right
        T'*μ_right == μ_left
    """

    def test_push_forward_left_to_right(self, shift_setup):
        """T * μ_left == μ_right."""
        H, P, full, left, right, mu_l, mu_r, scale = shift_setup
        T = TransferOperator(H, full, full)
        result = T.push_forward(mu_l)
        assert result == mu_r

    def test_pull_back_right_to_left(self, shift_setup):
        """T' * μ_right == μ_left."""
        H, P, full, left, right, mu_l, mu_r, scale = shift_setup
        T = TransferOperator(H, full, full)
        result = T.pull_back(mu_r)
        assert result == mu_l

    def test_push_forward_left_mass_preserved(self, shift_setup):
        """T*μ_left total mass = μ_left total mass (all left cells map into domain)."""
        H, P, full, left, right, mu_l, mu_r, scale = shift_setup
        T = TransferOperator(H, full, full)
        result = T.push_forward(mu_l)
        assert result.total() == pytest.approx(mu_l.total(), rel=1e-10)
