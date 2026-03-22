"""
gaio/mpi/comm.py
================
Lazy MPI communicator singleton with a serial fallback stub.

``mpi4py`` is imported only on the first call to ``get_comm()``.  If it is
not installed, or if the process is running outside of ``mpirun``, the stub
``_SerialComm`` is returned instead.  All downstream code (decompose, gather,
transfer operator) works identically in both cases.

Public API
----------
    get_comm()        — return the active communicator (real or stub)
    rank()            — this process's rank (0 in serial mode)
    size()            — number of ranks (1 in serial mode)
    is_mpi_active()   — True iff mpi4py is present AND size > 1
"""
from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Serial stub — mirrors the mpi4py.MPI.Comm interface we actually call
# ---------------------------------------------------------------------------

class _SerialComm:
    """
    Drop-in stub for mpi4py.MPI.COMM_WORLD when running single-process.

    Implements only the methods used by gaio.mpi:
        Get_rank, Get_size, Barrier, Allgather, Allgatherv, Bcast, bcast.
    """

    def Get_rank(self) -> int:
        return 0

    def Get_size(self) -> int:
        return 1

    def Barrier(self) -> None:
        pass

    def Allgather(self, sendbuf: Any, recvbuf: Any) -> None:
        # size=1: recvbuf has exactly len(sendbuf) elements
        np.copyto(recvbuf, sendbuf)

    def Allgatherv(self, sendbuf: Any, recvbuf: Any) -> None:
        # recvbuf is [buffer, (counts, displs)] — size=1, so just copy
        buf, (counts, displs) = recvbuf
        n = int(counts[0])
        buf[int(displs[0]) : int(displs[0]) + n] = sendbuf[:n]

    def Bcast(self, buf: Any, root: int = 0) -> None:
        pass  # already on rank 0, nothing to broadcast

    def bcast(self, obj: Any, root: int = 0) -> Any:
        return obj


# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_cache: dict[str, Any] = {}


def get_comm():
    """
    Return the MPI communicator for this process.

    First call: imports mpi4py if available **and** the process is running
    under ``mpiexec`` / ``mpirun`` (detected via standard env-var probes).
    Subsequent calls return the cached object with no import overhead.

    Returns ``_SerialComm`` when:
    - ``mpi4py`` is not installed, OR
    - The process is *not* running under ``mpiexec`` (no MPI env vars set).

    This prevents mpi4py from being imported in standalone scripts and
    tests, where some MPI implementations (e.g. Open MPI on WSL) can
    segfault during initialisation.

    Returns
    -------
    mpi4py.MPI.Intracomm or _SerialComm
    """
    if "comm" not in _cache:
        import os
        # Standard env-var probes for common MPI launchers:
        #   OMPI_COMM_WORLD_SIZE  — Open MPI
        #   PMI_SIZE              — MPICH / Intel MPI
        #   PMIX_RANK             — PMIx (Open MPI 4+, Slurm)
        #   MPI_LOCALNRANKS       — MVAPICH2
        in_mpi_env = bool(
            os.environ.get("OMPI_COMM_WORLD_SIZE")
            or os.environ.get("PMI_SIZE")
            or os.environ.get("PMIX_RANK")
            or os.environ.get("MPI_LOCALNRANKS")
        )
        if in_mpi_env:
            try:
                from mpi4py import MPI  # type: ignore[import]
                _cache["comm"] = MPI.COMM_WORLD
                _cache["MPI"]  = MPI
            except ImportError:
                _cache["comm"] = _SerialComm()
                _cache["MPI"]  = None
        else:
            _cache["comm"] = _SerialComm()
            _cache["MPI"]  = None
    return _cache["comm"]


def rank() -> int:
    """Return this process's MPI rank (0 when running single-process)."""
    return get_comm().Get_rank()


def size() -> int:
    """Return the number of MPI ranks (1 when running single-process)."""
    return get_comm().Get_size()


def is_mpi_active() -> bool:
    """Return True iff mpi4py is available and more than one rank is running."""
    return size() > 1
