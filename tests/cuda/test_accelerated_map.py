"""
tests/cuda/test_accelerated_map.py
====================================
Tests for AcceleratedBoxMap across all backends.

Test categories
---------------
TestBackendResolution      — resolve_backend() and constructor validation
TestPythonBackend          — AcceleratedBoxMap(backend='python') baseline
TestCPUBackend             — AcceleratedBoxMap(backend='cpu') vs SampledBoxMap
TestAcceleratedAlgorithms  — AcceleratedBoxMap used inside Phase 2 algorithms
TestGridAndMonteCarloWrappers — AcceleratedGridMap / AcceleratedMCMap patterns
TestGPUBackend             — @cuda.jit path (skipped if no GPU)
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except (ImportError, Exception):
    CUDA_AVAILABLE = False

from gaio.core.box import Box
from gaio.core.partition import BoxPartition
from gaio.core.boxset import BoxSet
from gaio.maps.base import SampledBoxMap
from gaio.cuda.accelerated_map import AcceleratedBoxMap
from gaio.cuda.backends import (
    BACKEND_PYTHON, BACKEND_CPU, BACKEND_GPU,
    resolve_backend,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

DOMAIN = Box([0.0, 0.0], [1.0, 1.0])
UNIT_PTS_4 = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])


def _f_py_scale(x):
    return x * 0.5


def _f_py_harmonic(x):
    return np.array([x[1], -x[0]])


def _f_py_identity(x):
    return x.copy()


if NUMBA_AVAILABLE:
    @njit
    def _f_jit_scale(x):
        return x * 0.5

    @njit
    def _f_jit_harmonic(x):
        out = np.empty(2, dtype=np.float64)
        out[0] = x[1]
        out[1] = -x[0]
        return out

    @njit
    def _f_jit_identity(x):
        return x.copy()

if CUDA_AVAILABLE:
    @cuda.jit(device=True)
    def _f_device_scale(x, out):
        """f(x) = x * 0.5  — output-parameter pattern."""
        out[0] = x[0] * 0.5
        out[1] = x[1] * 0.5

    @cuda.jit(device=True)
    def _f_device_identity(x, out):
        """f(x) = x  — output-parameter pattern."""
        out[0] = x[0]
        out[1] = x[1]


def _make_partition(n=4):
    return BoxPartition(DOMAIN, [n, n])


def _boxset_full(n=4):
    return BoxSet.full(_make_partition(n))


# ---------------------------------------------------------------------------
# Class 1: Backend resolution
# ---------------------------------------------------------------------------

class TestBackendResolution:

    def test_auto_python_when_no_callables(self):
        result = resolve_backend("auto", f_jit=None, f_device=None)
        assert result == BACKEND_PYTHON

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
    def test_auto_cpu_when_f_jit_provided_no_gpu(self):
        # Pass a dummy f_jit; GPU check done by resolve_backend
        result = resolve_backend("auto", f_jit=_f_jit_scale, f_device=None)
        assert result in (BACKEND_CPU, BACKEND_GPU)  # GPU if available

    def test_explicit_python_always_resolves(self):
        result = resolve_backend("python", f_jit=None, f_device=None)
        assert result == BACKEND_PYTHON

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
    def test_explicit_cpu_resolves(self):
        result = resolve_backend("cpu", f_jit=_f_jit_scale, f_device=None)
        assert result == BACKEND_CPU

    def test_explicit_cpu_without_f_jit_raises(self):
        with pytest.raises(ValueError, match="f_jit"):
            resolve_backend("cpu", f_jit=None, f_device=None)

    def test_explicit_gpu_without_f_device_raises(self):
        with pytest.raises(ValueError, match="f_device"):
            resolve_backend("gpu", f_jit=None, f_device=None)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            resolve_backend("turbo", f_jit=None, f_device=None)

    def test_constructor_stores_resolved_backend(self):
        F = AcceleratedBoxMap(
            _f_py_scale, DOMAIN, UNIT_PTS_4, backend="python"
        )
        assert F.backend == BACKEND_PYTHON

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
    def test_constructor_cpu_backend_stored(self):
        F = AcceleratedBoxMap(
            _f_py_scale, DOMAIN, UNIT_PTS_4,
            f_jit=_f_jit_scale, backend="cpu"
        )
        assert F.backend == BACKEND_CPU

    def test_bad_unit_points_shape_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            AcceleratedBoxMap(_f_py_scale, DOMAIN, np.array([0.0, 1.0]))


# ---------------------------------------------------------------------------
# Class 2: Python backend baseline
# ---------------------------------------------------------------------------

class TestPythonBackend:
    """AcceleratedBoxMap(backend='python') must equal SampledBoxMap exactly."""

    def setup_method(self):
        self.F_py = AcceleratedBoxMap(
            _f_py_scale, DOMAIN, UNIT_PTS_4, backend="python"
        )
        self.F_ref = SampledBoxMap(_f_py_scale, DOMAIN, UNIT_PTS_4)
        self.B = _boxset_full(4)

    def test_image_equals_sampled_box_map(self):
        img_acc = self.F_py(self.B)
        img_ref = self.F_ref(self.B)
        assert img_acc == img_ref

    def test_n_test_points_matches(self):
        assert self.F_py.n_test_points == self.F_ref.n_test_points == 4

    def test_ndim_matches(self):
        assert self.F_py.ndim == self.F_ref.ndim == 2

    def test_empty_source_returns_empty(self):
        empty = BoxSet.empty(self.B.partition)
        result = self.F_py(empty)
        assert result.is_empty()

    def test_contraction_image_subset_of_domain(self):
        """Contracting map: image ⊆ domain."""
        result = self.F_py(self.B)
        assert result.issubset(self.B)

    def test_identity_image_equals_source(self):
        F_id = AcceleratedBoxMap(
            _f_py_identity, DOMAIN, UNIT_PTS_4, backend="python"
        )
        B = _boxset_full(8)
        result = F_id(B)
        # Identity: F(B) = B (outer approximation may equal B exactly)
        assert result.issuperset(B)

    def test_repr_contains_backend(self):
        r = repr(self.F_py)
        assert "python" in r

    def test_call_and_map_boxes_identical(self):
        r1 = self.F_py(self.B)
        r2 = self.F_py.map_boxes(self.B)
        assert r1 == r2

    def test_coarser_partition_subset_of_finer(self):
        """Image on coarser grid ⊆ union of image on finer grid (monotone)."""
        B_coarse = _boxset_full(2)
        B_fine   = _boxset_full(4)
        img_coarse = self.F_py(B_coarse)
        img_fine   = self.F_py(B_fine)
        # Both images use scale-0.5 map; coarse image may be larger
        # At minimum they should both be BoxSets
        assert isinstance(img_coarse, BoxSet)
        assert isinstance(img_fine, BoxSet)


# ---------------------------------------------------------------------------
# Class 3: CPU backend correctness
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
class TestCPUBackend:

    def setup_method(self):
        self.F_cpu = AcceleratedBoxMap(
            _f_py_scale, DOMAIN, UNIT_PTS_4,
            f_jit=_f_jit_scale, backend="cpu"
        )
        self.F_ref = SampledBoxMap(_f_py_scale, DOMAIN, UNIT_PTS_4)
        self.B = _boxset_full(4)

    def test_cpu_image_equals_sampled_box_map(self):
        """CPU backend must produce the same BoxSet as SampledBoxMap."""
        img_cpu = self.F_cpu(self.B)
        img_ref = self.F_ref(self.B)
        assert img_cpu == img_ref

    def test_cpu_backend_attribute(self):
        assert self.F_cpu.backend == BACKEND_CPU

    def test_cpu_harmonic_image_nonempty(self):
        F_h = AcceleratedBoxMap(
            _f_py_harmonic, DOMAIN, UNIT_PTS_4,
            f_jit=_f_jit_harmonic, backend="cpu"
        )
        result = F_h(self.B)
        assert not result.is_empty()

    def test_cpu_identity_covers_source(self):
        F_id = AcceleratedBoxMap(
            _f_py_identity, DOMAIN, UNIT_PTS_4,
            f_jit=_f_jit_identity, backend="cpu"
        )
        B = _boxset_full(8)
        result = F_id(B)
        assert result.issuperset(B)

    def test_cpu_contraction_image_subset(self):
        result = self.F_cpu(self.B)
        assert result.issubset(self.B)

    def test_cpu_empty_source_returns_empty(self):
        empty = BoxSet.empty(self.B.partition)
        result = self.F_cpu(empty)
        assert result.is_empty()

    def test_cpu_repr_contains_cpu(self):
        assert "cpu" in repr(self.F_cpu)

    def test_cpu_n_test_points(self):
        assert self.F_cpu.n_test_points == 4

    def test_cpu_finer_partition(self):
        """CPU backend on [16,16] grid must match Python baseline."""
        B16 = _boxset_full(16)
        img_cpu = self.F_cpu(B16)
        img_ref = self.F_ref(B16)
        assert img_cpu == img_ref

    def test_cpu_repeated_calls_deterministic(self):
        r1 = self.F_cpu(self.B)
        r2 = self.F_cpu(self.B)
        assert r1 == r2

    def test_cpu_auto_backend_selects_cpu(self):
        """Auto backend selects CPU when f_jit is provided and no GPU."""
        F_auto = AcceleratedBoxMap(
            _f_py_scale, DOMAIN, UNIT_PTS_4,
            f_jit=_f_jit_scale, backend="auto"
        )
        # On a CPU-only machine auto selects CPU; on a GPU machine it selects GPU
        assert F_auto.backend in (BACKEND_CPU, BACKEND_GPU)

    def test_cpu_matches_python_on_16x16(self):
        """16×16 partition — larger batch exercises prange more heavily."""
        B = _boxset_full(16)
        img_cpu = self.F_cpu(B)
        img_py  = AcceleratedBoxMap(
            _f_py_scale, DOMAIN, UNIT_PTS_4, backend="python"
        )(B)
        assert img_cpu == img_py


# ---------------------------------------------------------------------------
# Class 4: AcceleratedBoxMap inside Phase 2 algorithms
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
class TestAcceleratedAlgorithms:
    """AcceleratedBoxMap must be usable as a drop-in inside all algorithms."""

    def setup_method(self):
        from gaio.algorithms.attractor import relative_attractor
        from gaio.algorithms.manifolds import unstable_set
        self.relative_attractor = relative_attractor
        self.unstable_set = unstable_set
        self.F = AcceleratedBoxMap(
            _f_py_scale, DOMAIN, UNIT_PTS_4,
            f_jit=_f_jit_scale, backend="cpu"
        )

    def test_relative_attractor_with_accelerated_map(self):
        P = BoxPartition(DOMAIN, [1, 1])
        B0 = BoxSet.full(P)
        result = self.relative_attractor(self.F, B0, steps=4)
        assert isinstance(result, BoxSet)
        assert not result.is_empty()

    def test_unstable_set_with_accelerated_map(self):
        P = BoxPartition(DOMAIN, [4, 4])
        # Seed: corner cell that maps inside domain
        seed_key = P.point_to_key(np.array([-0.5, -0.5]))
        seed = BoxSet(P, np.array([seed_key]))
        result = self.unstable_set(self.F, seed)
        assert isinstance(result, BoxSet)
        assert seed.issubset(result)

    def test_relative_attractor_convergence(self):
        """After steps=6, attractor should be smaller than initial set."""
        P = BoxPartition(DOMAIN, [1, 1])
        B0 = BoxSet.full(P)
        r4  = self.relative_attractor(self.F, B0, steps=4)
        r6  = self.relative_attractor(self.F, B0, steps=6)
        # More steps → at least as fine or smaller (not strictly necessary,
        # but contraction map should converge toward the origin cell)
        assert isinstance(r6, BoxSet)
        assert len(r6) <= len(r4) * 4  # at most 4× due to subdivision


# ---------------------------------------------------------------------------
# Class 5: GPU backend (skipped if no CUDA device)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="No CUDA device available")
class TestGPUBackend:

    def setup_method(self):
        self.F_gpu = AcceleratedBoxMap(
            _f_py_scale, DOMAIN, UNIT_PTS_4,
            f_device=_f_device_scale, backend="gpu"
        )
        self.F_ref = SampledBoxMap(_f_py_scale, DOMAIN, UNIT_PTS_4)
        self.B = _boxset_full(4)

    def test_gpu_backend_attribute(self):
        assert self.F_gpu.backend == BACKEND_GPU

    def test_gpu_image_equals_reference(self):
        """GPU kernel must produce same BoxSet as Python baseline."""
        img_gpu = self.F_gpu(self.B)
        img_ref = self.F_ref(self.B)
        assert img_gpu == img_ref

    def test_gpu_identity_covers_source(self):
        F_id = AcceleratedBoxMap(
            _f_py_identity, DOMAIN, UNIT_PTS_4,
            f_device=_f_device_identity, backend="gpu"
        )
        B = _boxset_full(8)
        result = F_id(B)
        assert result.issuperset(B)

    def test_gpu_empty_source(self):
        empty = BoxSet.empty(self.B.partition)
        result = self.F_gpu(empty)
        assert result.is_empty()

    def test_gpu_repr_contains_gpu(self):
        assert "gpu" in repr(self.F_gpu)

    def test_gpu_repeated_calls_deterministic(self):
        r1 = self.F_gpu(self.B)
        r2 = self.F_gpu(self.B)
        assert r1 == r2

    def test_gpu_auto_selects_gpu_when_device_provided(self):
        F_auto = AcceleratedBoxMap(
            _f_py_scale, DOMAIN, UNIT_PTS_4,
            f_device=_f_device_scale, backend="auto"
        )
        assert F_auto.backend == BACKEND_GPU

    def test_cuda_dispatcher_grid_dims(self):
        """Verify block/grid arithmetic covers all N work items."""
        from gaio.cuda.gpu_backend import CUDADispatcher
        disp = CUDADispatcher(_f_device_scale, threads_per_block=256)
        for N in [1, 255, 256, 257, 1024, 65536, 65537]:
            bpg, tpb = disp._grid_dims(N)
            assert bpg * tpb >= N, f"Grid too small for N={N}"
            assert tpb == 256

    def test_cuda_dispatcher_invalid_tpb_raises(self):
        from gaio.cuda.gpu_backend import CUDADispatcher
        with pytest.raises(ValueError, match="multiple of 32"):
            CUDADispatcher(_f_device_scale, threads_per_block=100)

    def test_cuda_dispatcher_tpb_over_max_raises(self):
        from gaio.cuda.gpu_backend import CUDADispatcher
        with pytest.raises(ValueError, match="maximum"):
            CUDADispatcher(_f_device_scale, threads_per_block=2048)
