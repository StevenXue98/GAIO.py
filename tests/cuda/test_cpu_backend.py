"""
tests/cuda/test_cpu_backend.py
================================
Tests for the Numba @njit(parallel=True) CPU backend.

These tests run on any machine with Numba installed — no GPU required.

Test categories
---------------
Class TestMapParallelCorrectness:
    Behavioural correctness of map_parallel against the Python baseline.

Class TestMapParallelEdgeCases:
    Single-point, single-dimension, large-N, identity, and zero-map cases.

Class TestCPUBackendProperties:
    Output dtype, shape, and memory-layout guarantees.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not NUMBA_AVAILABLE,
    reason="Numba not installed — skipping CPU backend tests",
)

from gaio.cuda.cpu_backend import map_parallel


# ---------------------------------------------------------------------------
# Shared Numba-compiled test functions
# ---------------------------------------------------------------------------

@njit
def _f_jit_scale_half(x):
    """f(x) = x * 0.5"""
    return x * 0.5


@njit
def _f_jit_harmonic(x):
    """f(x) = [x[1], -x[0]]  (harmonic oscillator RHS)"""
    out = np.empty(2, dtype=np.float64)
    out[0] = x[1]
    out[1] = -x[0]
    return out


@njit
def _f_jit_identity(x):
    return x.copy()


@njit
def _f_jit_zero(x):
    return np.zeros_like(x)


@njit
def _f_jit_square(x):
    return x * x


# ---------------------------------------------------------------------------
# Class 1: Correctness against Python baseline
# ---------------------------------------------------------------------------

class TestMapParallelCorrectness:

    def test_scale_matches_python_loop(self):
        """map_parallel(scale 0.5) must match a Python for-loop exactly."""
        rng = np.random.default_rng(0)
        pts = rng.standard_normal((500, 2)).astype(np.float64)

        # Python baseline
        expected = np.empty_like(pts)
        for i, p in enumerate(pts):
            expected[i] = p * 0.5

        result = map_parallel(_f_jit_scale_half, pts)
        np.testing.assert_allclose(result, expected, rtol=0, atol=0)

    def test_harmonic_matches_python_loop(self):
        """map_parallel(harmonic) must give [y, -x] for each input."""
        rng = np.random.default_rng(1)
        pts = rng.standard_normal((300, 2)).astype(np.float64)

        expected = np.column_stack([pts[:, 1], -pts[:, 0]])

        result = map_parallel(_f_jit_harmonic, pts)
        np.testing.assert_allclose(result, expected, rtol=1e-15, atol=1e-15)

    def test_identity_returns_copy_of_input(self):
        """Identity map: result must equal input point-wise."""
        rng = np.random.default_rng(2)
        pts = rng.standard_normal((200, 3)).astype(np.float64)
        result = map_parallel(_f_jit_identity, pts)
        np.testing.assert_array_equal(result, pts)

    def test_zero_map_returns_zeros(self):
        """Zero map must produce an all-zero output of same shape."""
        pts = np.ones((100, 4), dtype=np.float64)
        result = map_parallel(_f_jit_zero, pts)
        np.testing.assert_array_equal(result, np.zeros_like(pts))

    def test_element_wise_square(self):
        """x → x² element-wise."""
        pts = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, -2.0]])
        result = map_parallel(_f_jit_square, pts)
        expected = pts ** 2
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_large_batch(self):
        """Apply harmonic to 65536 points — exercises prange parallelism."""
        rng = np.random.default_rng(3)
        pts = rng.standard_normal((65536, 2)).astype(np.float64)
        expected = np.column_stack([pts[:, 1], -pts[:, 0]])
        result = map_parallel(_f_jit_harmonic, pts)
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)

    def test_deterministic_across_repeated_calls(self):
        """Two calls with the same input must return byte-identical results."""
        rng = np.random.default_rng(4)
        pts = rng.standard_normal((1000, 2)).astype(np.float64)
        r1 = map_parallel(_f_jit_scale_half, pts)
        r2 = map_parallel(_f_jit_scale_half, pts)
        np.testing.assert_array_equal(r1, r2)

    def test_3d_input(self):
        """Verify correctness for 3-dimensional state space."""
        @njit
        def f3d(x):
            out = np.empty(3, dtype=np.float64)
            out[0] = x[1]
            out[1] = x[2]
            out[2] = -x[0]
            return out

        rng = np.random.default_rng(5)
        pts = rng.standard_normal((200, 3)).astype(np.float64)
        result = map_parallel(f3d, pts)
        expected = np.column_stack([pts[:, 1], pts[:, 2], -pts[:, 0]])
        np.testing.assert_allclose(result, expected, rtol=1e-14)


# ---------------------------------------------------------------------------
# Class 2: Edge cases
# ---------------------------------------------------------------------------

class TestMapParallelEdgeCases:

    def test_single_point(self):
        """N=1: single row input."""
        pts = np.array([[2.0, 3.0]])
        result = map_parallel(_f_jit_harmonic, pts)
        np.testing.assert_allclose(result, [[3.0, -2.0]])

    def test_two_points(self):
        """N=2: minimal multi-row input (exercises parallel path with prange)."""
        pts = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = map_parallel(_f_jit_harmonic, pts)
        np.testing.assert_allclose(result, [[0.0, -1.0], [1.0, 0.0]])

    def test_non_contiguous_input_is_accepted(self):
        """map_parallel should handle non-C-contiguous arrays by copying."""
        pts = np.random.rand(200, 4).astype(np.float64)
        pts_f = np.asfortranarray(pts)     # Fortran order
        # map_parallel internally casts to C-contiguous
        result = map_parallel(_f_jit_scale_half, pts_f)
        expected = map_parallel(_f_jit_scale_half, pts)
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_float32_input_cast_to_float64(self):
        """Float32 input must be upcast to float64 before processing."""
        pts32 = np.random.rand(100, 2).astype(np.float32)
        result = map_parallel(_f_jit_scale_half, pts32)
        assert result.dtype == np.float64, "Output must be float64"

    def test_input_not_mutated(self):
        """map_parallel must not modify the input array."""
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        pts_copy = pts.copy()
        map_parallel(_f_jit_scale_half, pts)
        np.testing.assert_array_equal(pts, pts_copy)


# ---------------------------------------------------------------------------
# Class 3: Output properties
# ---------------------------------------------------------------------------

class TestCPUBackendProperties:

    def test_output_dtype_is_float64(self):
        pts = np.random.rand(50, 2).astype(np.float64)
        result = map_parallel(_f_jit_harmonic, pts)
        assert result.dtype == np.float64

    def test_output_shape_matches_input(self):
        pts = np.random.rand(128, 3).astype(np.float64)

        @njit
        def f3(x):
            return x * 2.0

        result = map_parallel(f3, pts)
        assert result.shape == pts.shape

    def test_output_is_c_contiguous(self):
        pts = np.random.rand(64, 2).astype(np.float64)
        result = map_parallel(_f_jit_harmonic, pts)
        assert result.flags["C_CONTIGUOUS"], "Output must be C-contiguous"

    def test_output_is_independent_from_input(self):
        """Mutating the output must not affect the input."""
        pts = np.ones((10, 2), dtype=np.float64)
        result = map_parallel(_f_jit_identity, pts)
        result[:] = 999.0
        np.testing.assert_array_equal(pts, np.ones((10, 2)))
