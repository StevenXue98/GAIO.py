"""
gaio/mpi/gather.py
==================
Allgatherv pipeline that collects COO (row, col, val) transfer-operator
triplets from all MPI ranks into one global array on every rank.

Design
------
Each rank independently computes the COO entries for its local shard of
domain cells (chaotic attractor sparsity means shard sizes will vary
significantly — the double-gyre can have 10× density differences between
spatial regions).  After all ranks finish, three ``Allgatherv`` calls
assemble the global triplet arrays.  Every rank receives the complete set,
so the subsequent ``scipy.sparse.coo_matrix`` assembly and column
normalisation can be performed locally without further communication.

Dynamic buffer allocation
-------------------------
Because each rank maps a different number of test points to valid codomain
cells (attractor sparsity + sparse regions near domain boundaries), the
receive buffers cannot be pre-sized statically.  The pipeline is:

    1. Each rank computes ``local_n = len(local_rows)``  (cheap).
    2. ``gather_sizes`` does one ``Allgather`` of the scalar ``local_n``
       from every rank — O(P) communication where P is the number of ranks.
    3. Rank r allocates ``np.empty(total, dtype)`` receive buffers exactly
       sized to the sum of all per-rank counts.
    4. Three ``Allgatherv`` calls fill rows, cols, and vals in one pass.

No rank ever over-allocates or under-allocates; buffer sizes are determined
at runtime from the actual computation results.

Return values
-------------
``allgather_coo`` returns a 4-tuple ``(rows, cols, vals, counts)`` where
``counts[r]`` is the number of COO entries contributed by rank ``r``.
This allows callers to build per-rank diagnostics without a second
collective call.

Public API
----------
    allgather_coo(comm, rows, cols, vals)
        → (global_rows, global_cols, global_vals, per_rank_counts)

    gather_sizes(comm, local_n)
        → (counts, displs, total)   (helper, also exported for testing)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gaio.core.box import F64, I64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gather_sizes(
    comm,
    local_n: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], int]:
    """
    Allgather the local COO count from every rank.

    This is a single ``Allgather`` of one int64 scalar per rank — the
    cheapest possible collective call.  Its result is used to size the
    receive buffers for the subsequent ``Allgatherv`` calls.

    Parameters
    ----------
    comm    : mpi4py communicator or _SerialComm
    local_n : int — number of COO triplets computed on this rank

    Returns
    -------
    counts : ndarray, shape (size,), int64 — entries per rank
    displs : ndarray, shape (size,), int64 — byte offset (in elements) per rank
    total  : int — sum of all counts (= size of each receive buffer)
    """
    sz   = comm.Get_size()
    send = np.array([local_n], dtype=np.int64)
    recv = np.empty(sz, dtype=np.int64)
    comm.Allgather(send, recv)
    displs = np.concatenate([[0], np.cumsum(recv[:-1])]).astype(np.int64)
    return recv, displs, int(recv.sum())


# ---------------------------------------------------------------------------
# Main gather
# ---------------------------------------------------------------------------

def allgather_coo(
    comm,
    rows: NDArray[I64],
    cols: NDArray[I64],
    vals: NDArray[F64],
) -> tuple[NDArray[I64], NDArray[I64], NDArray[F64], NDArray[np.int64]]:
    """
    Gather COO triplets from all MPI ranks onto every rank.

    Each rank contributes its local *rows*, *cols*, *vals* arrays.  The
    function returns the concatenation of all contributions in rank order
    on **every** rank (Allgatherv, not Gatherv), along with the per-rank
    entry counts so callers can build diagnostics without a second collective.

    Dynamic buffer allocation
    -------------------------
    Buffer sizes are determined at runtime:

    1. ``gather_sizes`` does one ``Allgather`` to collect ``local_n`` from
       every rank → produces ``counts[r]`` and ``total = sum(counts)``.
    2. Each rank allocates ``np.empty(total, dtype)`` for rows, cols, vals.
    3. Three ``Allgatherv`` calls fill the buffers in rank order.

    Serial fast-path
    ----------------
    When ``comm.Get_size() == 1`` the function returns the input arrays
    unchanged with no copies, and ``counts = np.array([len(rows)])``.

    Parameters
    ----------
    comm       : mpi4py.MPI.Intracomm or _SerialComm
    rows, cols : ndarray, dtype int64  — local row / column indices
    vals       : ndarray, dtype float64 — local values (all 1.0 for raw
                 hit counts; normalisation happens after matrix assembly)

    Returns
    -------
    global_rows   : ndarray, shape (total,), int64
    global_cols   : ndarray, shape (total,), int64
    global_vals   : ndarray, shape (total,), float64
    per_rank_counts : ndarray, shape (size,), int64
        ``per_rank_counts[r]`` is the number of COO entries from rank ``r``.
        Identical on every rank (all ranks receive the same data).

    Notes
    -----
    The returned arrays (except in serial fast-path) are freshly allocated
    buffers owned by the caller.
    """
    if comm.Get_size() == 1:
        # Serial fast-path: no communication, no copies
        return rows, cols, vals, np.array([len(rows)], dtype=np.int64)

    # ── Step 1: Allgather per-rank COO counts (1 collective, O(P) traffic) ──
    counts, displs, total = gather_sizes(comm, len(rows))

    # ── Step 2: Dynamically allocate exact-sized receive buffers ─────────────
    g_rows = np.empty(total, dtype=I64)
    g_cols = np.empty(total, dtype=I64)
    g_vals = np.empty(total, dtype=F64)

    # ── Step 3: Allgatherv — fill buffers from all ranks in one pass each ────
    # Use RDMA-aware helper: tries GPUDirect if arrays are on device and
    # CUDA-aware MPI is available; falls back to CPU staging otherwise.
    from gaio.mpi.rdma import rdma_allgatherv
    rdma_allgatherv(comm, rows, counts, displs, g_rows)
    rdma_allgatherv(comm, cols, counts, displs, g_cols)
    rdma_allgatherv(comm, vals, counts, displs, g_vals)

    return g_rows, g_cols, g_vals, counts
