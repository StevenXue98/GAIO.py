"""
gaio.mpi — Phase 4: Multi-GPU scaling via MPI
==============================================
Distributed domain decomposition and COO-triplet gather for the
GAIO transfer operator.

Quick-start
-----------
No user code changes are required.  Run any existing GAIO script under
``mpirun`` and the library activates the MPI path automatically:

    mpirun -n 4 python my_gaio_script.py
    mpirun -n 4 --bind-to socket python my_gaio_script.py   # socket-pinned

Each rank receives the full ``TransferOperator`` at the end (Allgatherv),
so all downstream spectral analysis (``eigs``, ``svds``) works unchanged
on every rank.

Multi-GPU usage
---------------
Pair with ``AcceleratedBoxMap(backend='gpu')`` and one MPI rank per GPU:

    # Launch 2 ranks, one per GPU
    mpirun -n 2 python -c "
    from gaio.mpi import rank
    import numba.cuda as cuda
    cuda.select_device(rank())   # bind rank i to GPU i
    # ... rest of script ...
    "

GPUDirect / CUDA-aware MPI check
---------------------------------
    from gaio.mpi import check_cuda_aware_mpi
    check_cuda_aware_mpi()

Public API
----------
    get_comm()               — active MPI communicator (or serial stub)
    rank()                   — this process's rank
    size()                   — total number of ranks
    is_mpi_active()          — True iff size > 1

    morton_encode(multi_idx) — (N,d) int → (N,) uint64 Morton codes
    morton_sort_keys(keys, partition) — sort flat keys by Morton code
    local_keys(keys, rank, size)      — contiguous slice for one rank
    decompose(keys, partition, rank, size) — Morton-sort + slice

    allgather_coo(comm, rows, cols, vals) — Allgatherv COO triplets
    gather_sizes(comm, local_n)           — Allgather entry counts

    check_cuda_aware_mpi()   — probe for GPUDirect RDMA support
    mpi_info()               — print rank / size / GPU info
"""
from __future__ import annotations

from .comm import get_comm, rank, size, is_mpi_active
from .decompose import morton_encode, morton_sort_keys, local_keys, decompose
from .gather import allgather_coo, gather_sizes
from .rdma import is_rdma_capable, rdma_allgatherv, stage_to_host
from .distributed_attractor import distributed_relative_attractor
from .distributed_eigs import (
    slepc_available, distributed_eigs, distributed_svds,
)
from .load_balance import (
    compute_imbalance,
    should_rebalance,
    weighted_local_keys,
    compute_partition_weights,
)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def check_cuda_aware_mpi() -> bool:
    """
    Probe whether the active MPI build supports CUDA-aware (GPUDirect) mode.

    CUDA-aware MPI allows GPU device buffers to be passed directly to MPI
    send/recv calls without staging through CPU memory (GPUDirect RDMA).
    This is strictly optional — GAIO always stages through CPU NumPy arrays
    so it works correctly regardless — but CUDA-aware MPI eliminates a
    device-to-host copy per ``Allgatherv`` call and improves bandwidth on
    InfiniBand clusters.

    Detection strategy
    ------------------
    1. Check the ``OMPI_MCA_opal_cuda_support`` environment variable
       (Open MPI sets this to ``"1"`` when CUDA support is compiled in).
    2. Call ``mpi4py.MPI.query_thread()`` and inspect the MPI library
       version string for known CUDA-aware builds.
    3. Fall back to ``False`` if neither heuristic fires.

    Returns
    -------
    bool
        ``True`` if CUDA-aware MPI is detected, ``False`` otherwise.

    Prints a human-readable summary to stdout.
    """
    import os

    cuda_aware = False
    reason = "no positive indicator found"

    # Heuristic 1: Open MPI environment variable
    if os.environ.get("OMPI_MCA_opal_cuda_support", "0") == "1":
        cuda_aware = True
        reason = "OMPI_MCA_opal_cuda_support=1"

    # Heuristic 2: mpi4py version string
    if not cuda_aware:
        try:
            from mpi4py import MPI  # type: ignore[import]
            lib_ver = MPI.Get_library_version()
            if "cuda" in lib_ver.lower() or "gpudirect" in lib_ver.lower():
                cuda_aware = True
                reason = f"MPI library version string contains CUDA: {lib_ver[:80]}"
        except ImportError:
            reason = "mpi4py not installed"

    # Heuristic 3: MPICH / MVAPICH env
    if not cuda_aware:
        if os.environ.get("MV2_USE_CUDA", "0") == "1":
            cuda_aware = True
            reason = "MV2_USE_CUDA=1 (MVAPICH)"

    status = "YES (GPUDirect RDMA likely available)" if cuda_aware else "NO (CPU staging will be used)"
    print(f"[gaio.mpi] CUDA-aware MPI: {status}")
    print(f"[gaio.mpi]   reason : {reason}")
    print(f"[gaio.mpi]   rank={rank()}, size={size()}")
    return cuda_aware


def mpi_info() -> None:
    """
    Print a one-line summary of MPI rank, size, and GPU assignment.

    Each rank prints independently; output may be interleaved when running
    under ``mpirun``.
    """
    r, s = rank(), size()
    gpu_str = "(no GPU / CPU mode)"
    try:
        from numba import cuda as numba_cuda  # type: ignore[import]
        if numba_cuda.is_available():
            dev = numba_cuda.get_current_device()
            gpu_str = f"GPU {dev.id}: {dev.name.decode()}"
    except Exception:
        pass
    print(f"[gaio.mpi] rank {r}/{s}  {gpu_str}")


__all__ = [
    # comm
    "get_comm", "rank", "size", "is_mpi_active",
    # decompose
    "morton_encode", "morton_sort_keys", "local_keys", "decompose",
    # gather
    "allgather_coo", "gather_sizes",
    # rdma
    "is_rdma_capable", "rdma_allgatherv", "stage_to_host",
    # distributed algorithms
    "distributed_relative_attractor",
    "slepc_available", "distributed_eigs", "distributed_svds",
    # load balancing (Phase 5)
    "compute_imbalance", "should_rebalance",
    "weighted_local_keys", "compute_partition_weights",
    # diagnostics
    "check_cuda_aware_mpi", "mpi_info",
]
