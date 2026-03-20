"""
gaio.cuda — Phase 3: Heterogeneous Acceleration (CPU & GPU)

Public API
----------
AcceleratedBoxMap  — drop-in SampledBoxMap replacement with Numba backends
CUDADispatcher     — GPU memory manager + kernel launcher (advanced use)
make_map_kernel    — CUDA kernel factory (advanced use)
map_parallel       — CPU parallel map loop (advanced use)
BACKEND_PYTHON     — string constant 'python'
BACKEND_CPU        — string constant 'cpu'
BACKEND_GPU        — string constant 'gpu'
cuda_available     — runtime CUDA device check
"""
from .accelerated_map import AcceleratedBoxMap
from .gpu_backend import CUDADispatcher, make_map_kernel
from .cpu_backend import map_parallel
from .backends import (
    BACKEND_PYTHON,
    BACKEND_CPU,
    BACKEND_GPU,
    _cuda_available as cuda_available,
    _numba_available as numba_available,
    detect_gpu_dtype,
    _query_fp64_ratio as query_fp64_ratio,
)

__all__ = [
    "AcceleratedBoxMap",
    "CUDADispatcher",
    "make_map_kernel",
    "map_parallel",
    "BACKEND_PYTHON",
    "BACKEND_CPU",
    "BACKEND_GPU",
    "cuda_available",
    "numba_available",
]
