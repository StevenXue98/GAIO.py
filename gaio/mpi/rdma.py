"""
gaio/mpi/rdma.py
=================
GPUDirect RDMA utilities for CUDA-aware MPI communication.

What GPUDirect RDMA does
------------------------
In the default (non-RDMA) pipeline, the Allgatherv that assembles the
global COO matrix requires two copies per rank:

    GPU VRAM → host RAM  (device-to-host copy, PCIe)
    host RAM → NIC       (DMA to network card)

With CUDA-aware MPI (OpenMPI ≥ 4.0 + UCX + GPUDirect RDMA driver), the
NIC can read directly from GPU VRAM over PCIe:

    GPU VRAM → NIC       (single DMA, no CPU staging)

For a 100K-cell, 27-test-point, 3-D problem, the COO values array is
100K × 27 × 8 bytes ≈ 21 MB per rank per frame.  With GPUDirect the
device-to-host copy (typically 8–12 GB/s on PCIe 4.0) is eliminated,
saving ~2 ms per frame — significant in nonautonomous animations that
compute 24–60 frames.

Current pipeline stage where RDMA applies
------------------------------------------
Currently, stages 1–3 of ``_build_transitions`` produce numpy (CPU) arrays.
GPUDirect only helps when the COO data lives on the device at the point of
the Allgatherv.  Activating full RDMA requires:

    Stage 2 (map): F._apply_map returns a CUDA device array
    Stage 3 (key lookup): point_to_key_batch runs on GPU → device COO
    Allgatherv receives device arrays directly

The infrastructure in this module is ready for that pipeline.  The
``rdma_allgatherv`` function tries a device-direct Allgatherv and falls
back to CPU staging if RDMA is unavailable or the arrays are on host.

Detection
---------
Three-level probe (in order of reliability):
1. ``OMPI_MCA_opal_cuda_support=1`` environment variable (Open MPI).
2. ``MV2_USE_CUDA=1`` environment variable (MVAPICH2).
3. Runtime probe: try a device-direct Allgather of a tiny test array;
   if MPI raises or returns wrong data, RDMA is not available.

The probe result is cached after the first call.

Public API
----------
    is_rdma_capable(comm)                        — bool (cached)
    rdma_allgatherv(comm, arr, counts, displs)   — device-or-host gather
    stage_to_host(arr)                           — copy device→host if needed
"""
from __future__ import annotations

import os
from typing import Union

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Device array detection
# ---------------------------------------------------------------------------

def _is_device_array(arr) -> bool:
    """Return True if *arr* is a Numba CUDA DeviceNDArray or exposes
    ``__cuda_array_interface__`` (CuPy, RAPIDS, etc.)."""
    # Numba device array
    try:
        from numba.cuda.cudadrv.devicearray import DeviceNDArray
        if isinstance(arr, DeviceNDArray):
            return True
    except ImportError:
        pass
    # Generic CUDA array interface (CuPy, etc.)
    return hasattr(arr, "__cuda_array_interface__")


def stage_to_host(arr) -> NDArray:
    """
    Copy *arr* to a numpy host array if it is on a CUDA device.
    Returns *arr* unchanged if it is already a numpy array.
    """
    if not _is_device_array(arr):
        return arr
    # Numba DeviceNDArray
    try:
        return arr.copy_to_host()
    except AttributeError:
        pass
    # CuPy or other __cuda_array_interface__ arrays
    try:
        import cupy as cp
        return cp.asnumpy(arr)
    except Exception:
        pass
    # Last resort: numpy frombuffer via __cuda_array_interface__
    return np.array(arr)


# ---------------------------------------------------------------------------
# RDMA capability probe
# ---------------------------------------------------------------------------

_rdma_cache: dict[int, bool] = {}   # keyed by id(comm)


def is_rdma_capable(comm) -> bool:
    """
    Return True iff the MPI build supports CUDA-aware (GPUDirect) transfers.

    The result is cached per communicator object.

    Detection strategy
    ------------------
    1. Open MPI: ``OMPI_MCA_opal_cuda_support`` env var.
    2. MVAPICH2: ``MV2_USE_CUDA`` env var.
    3. Runtime probe: Allgather a 4-byte CUDA device array and verify the
       result.  If the MPI library does not support device arrays it will
       either raise or corrupt the data.

    Parameters
    ----------
    comm : mpi4py communicator or _SerialComm

    Returns
    -------
    bool
    """
    cid = id(comm)
    if cid in _rdma_cache:
        return _rdma_cache[cid]

    capable = False

    # ── Env-var heuristics (fast, no CUDA needed) ────────────────────────────
    if os.environ.get("OMPI_MCA_opal_cuda_support", "0") == "1":
        capable = True
    elif os.environ.get("MV2_USE_CUDA", "0") == "1":
        capable = True
    else:
        # ── Runtime probe ────────────────────────────────────────────────────
        try:
            from numba import cuda as numba_cuda
            if numba_cuda.is_available():
                # Allocate a 1-element device array with a known value
                d_send = numba_cuda.to_device(np.array([42], dtype=np.int32))
                d_recv = numba_cuda.device_array(
                    comm.Get_size(), dtype=np.int32
                )
                comm.Allgather(d_send, d_recv)
                h_recv = d_recv.copy_to_host()
                if np.all(h_recv == 42):
                    capable = True
        except Exception:
            capable = False   # MPI rejected device pointer or CUDA unavailable

    _rdma_cache[cid] = capable
    return capable


# ---------------------------------------------------------------------------
# RDMA-aware Allgatherv
# ---------------------------------------------------------------------------

def rdma_allgatherv(
    comm,
    arr: Union[NDArray, "numba.cuda.cudadrv.devicearray.DeviceNDArray"],
    counts: NDArray[np.int64],
    displs: NDArray[np.int64],
    out: NDArray,
) -> None:
    """
    Allgatherv with GPUDirect RDMA when available, CPU staging otherwise.

    Parameters
    ----------
    comm   : mpi4py communicator
    arr    : send buffer — numpy array or CUDA device array
    counts : ndarray, int64, shape (size,) — elements per rank
    displs : ndarray, int64, shape (size,) — start offsets per rank
    out    : receive buffer — numpy array (always; output is on host)

    Notes
    -----
    If *arr* is a device array and RDMA is available, the device pointer is
    passed directly to ``comm.Allgatherv``.  The MPI library (Open MPI/UCX)
    handles the DMA read and writes to the CPU receive buffer ``out``.

    If *arr* is a device array but RDMA is unavailable, it is first copied
    to host via ``stage_to_host``, then a normal Allgatherv is called.

    If *arr* is already a numpy array, a normal Allgatherv is called
    regardless of RDMA capability.
    """
    on_device = _is_device_array(arr)

    if on_device and is_rdma_capable(comm):
        # Device-direct path: MPI reads from GPU VRAM
        try:
            comm.Allgatherv(arr, [out, (counts, displs)])
            return
        except Exception:
            # Fall through to CPU staging if device Allgatherv fails
            pass

    # CPU staging path (always safe)
    host_arr = stage_to_host(arr) if on_device else arr
    comm.Allgatherv(host_arr, [out, (counts, displs)])
