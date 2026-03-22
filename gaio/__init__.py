"""
gaio — Global Analysis of Invariant Objects
============================================
Python port of GAIO.jl (https://github.com/gaioguys/GAIO.jl)
using NumPy, Numba CUDA, and mpi4py.

Install (from the GAIO.py repo root):
    pip install -e .

Then from anywhere:
    from gaio import Box, BoxPartition, BoxSet

Phase 1 public API (foundational data structures):
    Box           — axis-aligned hyperrectangle  [c-r, c+r)
    BoxPartition  — uniform Cartesian grid; cells indexed by flat int64 keys
    BoxSet        — sparse sorted set of active cells; full set algebra

Phase 2 public API (BoxMap discretisation):
    SampledBoxMap — maps a continuous f onto a BoxPartition via test points
    GridMap       — SampledBoxMap with uniform Cartesian grid test points
    MonteCarloMap — SampledBoxMap with random Monte-Carlo test points
"""
from .core import Box, BoxPartition, BoxSet, BoxMeasure, F64, I64
from .maps import SampledBoxMap, GridMap, MonteCarloMap, rk4_step, rk4_flow_map, rk4_flow_map_tspan, NonautonomousBoxMap
from .algorithms import (
    relative_attractor, unstable_set,
    preimage, alpha_limit_set, maximal_invariant_set,
    morse_sets, morse_tiles, recurrent_set,
    finite_time_lyapunov_exponents,
)
from .transfer import TransferOperator
from .graph import BoxGraph
from .cuda import (
    AcceleratedBoxMap,
    CUDADispatcher,
    make_map_kernel,
    map_parallel,
    BACKEND_PYTHON, BACKEND_CPU, BACKEND_GPU,
    cuda_available, numba_available,
)

__all__ = [
    # core
    "Box", "BoxPartition", "BoxSet", "BoxMeasure", "F64", "I64",
    # maps
    "SampledBoxMap", "GridMap", "MonteCarloMap",
    "rk4_step", "rk4_flow_map", "rk4_flow_map_tspan", "NonautonomousBoxMap",
    # transfer & graph
    "TransferOperator", "BoxGraph",
    # algorithms
    "relative_attractor", "unstable_set",
    "preimage", "alpha_limit_set", "maximal_invariant_set",
    "morse_sets", "morse_tiles", "recurrent_set",
    "finite_time_lyapunov_exponents",
    # Phase 3 — heterogeneous acceleration
    "AcceleratedBoxMap",
    "CUDADispatcher", "make_map_kernel", "map_parallel",
    "BACKEND_PYTHON", "BACKEND_CPU", "BACKEND_GPU",
    "cuda_available", "numba_available",
]
__version__ = "0.1.0"
