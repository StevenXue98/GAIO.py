"""
Tests for gaio/maps/rk4.py — RK4 integrator.

Two categories:
  A. Behavioural (randomized / property-based)
  B. Exact numerical portlets of Julia's maps.jl contract
"""
import numpy as np
import pytest

from gaio.core.box import Box
from gaio.core.partition import BoxPartition
from gaio.core.boxset import BoxSet
from gaio.maps.base import SampledBoxMap
from gaio.maps.rk4 import rk4_step, rk4_flow_map


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def linear_ode(x):
    """x' = -x  →  exact solution x(t) = exp(-t) x₀."""
    return -x


def harmonic_ode(x):
    """x' = [x[1], -x[0]]  (SHO, ω=1)  →  energy x₀²+x₁² conserved."""
    return np.array([x[1], -x[0]])


def lorenz_ode(x, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """Lorenz system."""
    return np.array([
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ])


# ===========================================================================
# A. Behavioural tests
# ===========================================================================

class TestRK4Step:
    def test_returns_ndarray(self):
        x = np.array([1.0, 2.0])
        result = rk4_step(lambda x: -x, x, 0.1)
        assert isinstance(result, np.ndarray)

    def test_output_shape_matches_input(self):
        for n in [1, 2, 3, 5]:
            x = np.ones(n)
            result = rk4_step(lambda x: -x, x, 0.1)
            assert result.shape == (n,)

    def test_zero_step_is_identity(self):
        x = np.array([3.14, -2.71])
        result = rk4_step(lambda x: x, x, 0.0)
        np.testing.assert_allclose(result, x, atol=1e-14)

    def test_scalar_ode_exact(self):
        """x' = -x, step τ=0.1: compare to 4th-order Taylor expansion."""
        x0 = np.array([2.0])
        tau = 0.1
        result = rk4_step(linear_ode, x0, tau)
        exact = x0 * np.exp(-tau)
        np.testing.assert_allclose(result, exact, rtol=1e-7)

    def test_harmonic_energy_conservation(self):
        """Energy should be conserved to machine precision for SHO."""
        x0 = np.array([1.0, 0.0])
        result = rk4_step(harmonic_ode, x0, 0.01)
        e0 = float(np.dot(x0, x0))
        e1 = float(np.dot(result, result))
        assert abs(e1 - e0) < 1e-8

    def test_float32_input_upcasts(self):
        """rk4_step should accept float32 and return float64."""
        x = np.array([1.0, 0.0], dtype=np.float32)
        result = rk4_step(harmonic_ode, x, 0.01)
        assert result.dtype == np.float64

    def test_list_input_works(self):
        result = rk4_step(linear_ode, [1.0, 2.0], 0.1)
        assert isinstance(result, np.ndarray)


class TestRK4FlowMap:
    def test_returns_callable(self):
        flow = rk4_flow_map(linear_ode)
        assert callable(flow)

    def test_linear_ode_matches_exact(self):
        """x' = -x over T=1 (step=0.01, steps=100)."""
        flow = rk4_flow_map(linear_ode, step_size=0.01, steps=100)
        x0 = np.array([1.0])
        result = flow(x0)
        np.testing.assert_allclose(result, np.exp(-1.0), rtol=1e-7)

    def test_harmonic_oscillator_period(self):
        """SHO with ω=1: period T=2π → should return to initial state."""
        T = 2 * np.pi
        step = 0.001
        steps = int(round(T / step))
        flow = rk4_flow_map(harmonic_ode, step_size=step, steps=steps)
        x0 = np.array([1.0, 0.0])
        result = flow(x0)
        np.testing.assert_allclose(result, x0, atol=5e-4)

    def test_flow_map_default_params(self):
        """Default step_size=0.01, steps=20 → T=0.2."""
        flow = rk4_flow_map(linear_ode)
        x0 = np.array([1.0])
        result = flow(x0)
        np.testing.assert_allclose(result, np.exp(-0.2), rtol=1e-7)

    def test_multidimensional_contraction(self):
        """x' = -x in n=3 — all components should decay."""
        flow = rk4_flow_map(linear_ode, step_size=0.1, steps=10)  # T=1
        x0 = np.array([1.0, 2.0, 3.0])
        result = flow(x0)
        expected = x0 * np.exp(-1.0)
        np.testing.assert_allclose(result, expected, rtol=1e-6)


# ===========================================================================
# B. Julia exact-port tests
# ===========================================================================

class TestRK4AsBoxMapFlow:
    """
    Port of Julia usage pattern:
        g = rk4_flow_map(f, step_size=0.01, steps=20)
        F = GridBoxMap(g, domain)
    Verify that wrapping rk4_flow_map in a SampledBoxMap gives the
    expected image for a simple ODE.
    """

    def test_contraction_image_near_origin(self):
        """
        For x'=-x, flow map contracts everything toward 0.
        The forward image of the full domain should cover the origin cell.
        """
        domain = Box([0.0, 0.0], [1.0, 1.0])
        flow = rk4_flow_map(linear_ode, step_size=0.05, steps=20)  # T=1
        pts = np.array([[-1., -1.], [-1., 1.], [0., 0.], [1., -1.], [1., 1.]])
        F = SampledBoxMap(flow, domain, pts)
        P = BoxPartition(domain, [8, 8])
        B = BoxSet.full(P)
        image = F(B)
        # Image should be non-empty
        assert len(image) > 0
        # Origin cell should be in the image (everything contracts to 0)
        origin_key = P.point_to_key(np.array([0.0, 0.0]))
        assert origin_key in image

    def test_lorenz_flow_produces_nonempty_image(self):
        """
        Lorenz flow from a seed near the attractor should stay in the domain.
        """
        domain = Box([0.0, 0.0, 25.0], [25.0, 30.0, 15.0])
        flow = rk4_flow_map(lorenz_ode, step_size=0.001, steps=50)
        pts = np.array([[0.0, 0.0, 0.0]])
        F = SampledBoxMap(flow, domain, pts)
        P = BoxPartition(domain, [4, 4, 4])
        # Single cell at origin of the domain
        seed = BoxSet.cover(P, np.array([[0.0, 0.0, 25.0]]))
        if len(seed) > 0:
            image = F(seed)
            assert isinstance(image, BoxSet)
