"""
Shared pytest fixtures for the gaio test suite.

All fixtures are built from the Phase 1 public API so that later phases
can reuse them without modification.
"""
import numpy as np
import pytest

from gaio import Box, BoxPartition, BoxSet, F64, I64


# ── Domain Boxes ──────────────────────────────────────────────────────────────

@pytest.fixture
def domain_1d():
    """1-D domain  [-1, 1)."""
    return Box(np.array([0.0]), np.array([1.0]))


@pytest.fixture
def domain_2d():
    """2-D domain  [-1, 1)²  centred at the origin."""
    return Box(np.zeros(2), np.ones(2))


@pytest.fixture
def domain_3d():
    """3-D domain  [-1, 1)³."""
    return Box(np.zeros(3), np.ones(3))


@pytest.fixture
def domain_4d():
    """4-D domain  [-1, 1)⁴  — used to stress-test high-dimensional paths."""
    return Box(np.zeros(4), np.ones(4))


# ── BoxPartitions ─────────────────────────────────────────────────────────────

@pytest.fixture
def partition_2d(domain_2d):
    """4 × 4 = 16 cell partition of the 2-D unit domain."""
    return BoxPartition(domain_2d, np.array([4, 4], dtype=I64))


@pytest.fixture
def partition_2d_asym(domain_2d):
    """3 × 5 = 15 cell partition — non-square to catch dimension-ordering bugs."""
    return BoxPartition(domain_2d, np.array([3, 5], dtype=I64))


@pytest.fixture
def partition_3d(domain_3d):
    """4 × 4 × 4 = 64 cell partition of the 3-D unit domain."""
    return BoxPartition(domain_3d, np.array([4, 4, 4], dtype=I64))


@pytest.fixture
def partition_4d(domain_4d):
    """2 × 2 × 2 × 2 = 16 cell partition of the 4-D unit domain."""
    return BoxPartition(domain_4d, np.array([2, 2, 2, 2], dtype=I64))


# ── BoxSets ───────────────────────────────────────────────────────────────────

@pytest.fixture
def full_set_2d(partition_2d):
    """All 16 cells of the 4 × 4 partition."""
    return BoxSet.full(partition_2d)


@pytest.fixture
def half_set_2d(partition_2d):
    """First 8 cells (keys 0–7)."""
    return BoxSet(partition_2d, np.arange(8, dtype=I64))


@pytest.fixture
def overlap_set_2d(partition_2d):
    """Cells 4–11 — overlaps with half_set_2d on keys 4–7."""
    return BoxSet(partition_2d, np.arange(4, 12, dtype=I64))
