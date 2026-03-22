"""
gaio/mpi/load_balance.py
========================
Phase 5: adaptive load-balancing for MPI-distributed GAIO computation.

Problem
-------
Phase 4 splits ``domain._keys`` into equal-sized K/P chunks (one per rank)
after Morton sorting.  For non-uniform attractors such as the Hénon
horseshoe or the Ikeda spiral, the per-cell hit density varies widely:
cells near the attractor core map almost all M test points back into the
codomain, while cells at the periphery map few.  Because every rank
processes the same number of cells (K/P), the per-rank COO contribution
(``per_rank_nnz``) can differ by an order of magnitude — the fast ranks
sit idle at the Allgatherv barrier while the overloaded rank finishes.

Solution
--------
After one ``TransferOperator`` construction we know the per-cell hit
counts.  On the next call we use a **weighted prefix-sum split**: instead
of giving each rank K/P keys, we give each rank a contiguous Morton range
whose total hit weight is ≈ total_weight / P.  Heavily loaded regions
receive fewer keys per rank; sparse regions receive more.

Usage pattern (nonautonomous systems)
--------------------------------------
::

    weights = None                          # frame 0: Phase 4 behavior
    for t in range(n_frames):
        A = relative_attractor(F_t, S, steps=steps, comm=comm)
        T = TransferOperator(F_t, A, A, comm=comm, partition_weights=weights)
        weights = T.partition_weights       # reuse on next frame

For a single-frame computation pass ``partition_weights=None`` (default) —
this is identical to Phase 4 and incurs no extra overhead.

Public API
----------
    compute_imbalance(per_rank_counts)           — max/min imbalance ratio
    should_rebalance(per_rank_counts, ...)       — threshold gate
    weighted_local_keys(keys, weights, r, P)    — load-balanced shard
    compute_partition_weights(local_hits, comm)  — gather per-cell weights
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gaio.core.box import I64


# ---------------------------------------------------------------------------
# Imbalance metrics
# ---------------------------------------------------------------------------

def compute_imbalance(per_rank_counts: NDArray) -> float:
    """
    Max/min load imbalance ratio from per-rank COO contribution counts.

    Parameters
    ----------
    per_rank_counts : array-like, shape (P,)
        Number of COO entries contributed by each rank
        (``TransferOperator.mpi_stats["per_rank_nnz"]``).

    Returns
    -------
    float
        Imbalance ≥ 1.0.  Returns ``float('inf')`` when any rank has
        zero entries; returns ``1.0`` for a single rank.
    """
    counts = np.asarray(per_rank_counts, dtype=np.int64)
    if len(counts) <= 1:
        return 1.0
    mn = int(counts.min())
    if mn == 0:
        return float("inf")
    return float(counts.max()) / mn


def should_rebalance(
    per_rank_counts: NDArray,
    threshold: float = 2.0,
    min_total_nnz: int = 500,
) -> bool:
    """
    Return ``True`` when load rebalancing is likely to improve performance.

    Rebalancing adds a one-time ``Allgatherv`` of per-cell hit counts (O(K)
    int32 values) plus a weighted partition step (O(K)).  This overhead is
    amortised over subsequent frames in nonautonomous systems.  The gate
    activates only when:

    1. Total work exceeds *min_total_nnz* — below this, communication
       overhead dominates any compute saving.
    2. Imbalance ratio exceeds *threshold* — below this, ranks are already
       near-uniform and rebalancing yields diminishing returns.

    Parameters
    ----------
    per_rank_counts : array-like, shape (P,)
    threshold : float
        Imbalance trigger.  Default: 2.0×.
    min_total_nnz : int
        Minimum total COO entries.  Default: 500.

    Returns
    -------
    bool
    """
    counts = np.asarray(per_rank_counts, dtype=np.int64)
    if int(counts.sum()) < min_total_nnz:
        return False
    return compute_imbalance(counts) > threshold


# ---------------------------------------------------------------------------
# Weighted domain decomposition
# ---------------------------------------------------------------------------

def weighted_local_keys(
    morton_sorted_keys: NDArray[I64],
    weights: NDArray,
    rank: int,
    size: int,
) -> NDArray[I64]:
    """
    Return the contiguous Morton shard for *rank* with approximately equal
    total weight to every other rank.

    Phase 4 uniform split: every rank gets K // P consecutive Morton keys.
    Phase 5 weighted split: every rank gets a range of consecutive Morton
    keys with total weight ≈ ``sum(weights) / size``, reducing idle time at
    the Allgatherv barrier for non-uniform attractors.

    Parameters
    ----------
    morton_sorted_keys : ndarray, shape (K,), int64
        Keys in Morton order (output of :func:`gaio.mpi.decompose.morton_sort_keys`).
    weights : ndarray, shape (K,), float32 or float64
        Non-negative per-cell estimated work from the previous frame
        (``TransferOperator.partition_weights``).  Must satisfy
        ``len(weights) == K``.
    rank : int  — 0 ≤ rank < size
    size : int  — total number of MPI ranks

    Returns
    -------
    ndarray, shape (local_K,), int64
        Contiguous slice of *morton_sorted_keys* for this rank.

    Notes
    -----
    *   Computes the cumulative weight prefix sum in float64 to prevent
        precision loss when summing many float32 values.
    *   The last rank always captures the full tail (``ends[-1] = K``) to
        absorb any floating-point rounding in ``linspace``.
    *   Falls back to the uniform Phase 4 split when all weights are zero.
    """
    K = len(morton_sorted_keys)
    if size == 1 or K == 0:
        return morton_sorted_keys

    w = np.asarray(weights, dtype=np.float64)
    total = float(w.sum())

    if total == 0.0:
        # All weights zero — uniform Phase 4 fallback
        from gaio.mpi.decompose import local_keys
        return local_keys(morton_sorted_keys, rank, size)

    # Prefix-sum boundary search
    cumw = np.cumsum(w)                                       # (K,) float64
    boundaries = np.linspace(0.0, total, size + 1)            # (size+1,)

    # searchsorted('left'): first position where cumw >= boundary
    starts = np.searchsorted(cumw, boundaries[:-1], side="left")   # (size,)
    ends   = np.searchsorted(cumw, boundaries[1:],  side="left")   # (size,)

    # Clamp and guarantee last rank gets the full tail
    starts = np.clip(starts, 0, K)
    ends   = np.clip(ends,   0, K)
    ends[-1] = K

    return morton_sorted_keys[int(starts[rank]) : int(ends[rank])]


# ---------------------------------------------------------------------------
# Per-cell weight gathering
# ---------------------------------------------------------------------------

def compute_partition_weights(
    local_hits: NDArray[np.int32],
    comm,
) -> NDArray[np.float32]:
    """
    Allgatherv per-cell hit counts from all ranks into a global weight array.

    After ``TransferOperator`` Stage 3, each rank holds the hit counts for
    its local Morton shard.  This function gathers them into a global array
    (indexed by Morton position, length = K) cast to float32.

    Serial fast-path: when ``comm.Get_size() == 1``, returns
    ``local_hits.astype(float32)`` with no communication.

    Parameters
    ----------
    local_hits : ndarray, shape (local_K,), int32
        Hit count per source cell in Morton order (accumulated during
        Stage 3 of :func:`gaio.transfer.operator._build_transitions`).
    comm : mpi4py communicator or _SerialComm

    Returns
    -------
    ndarray, shape (K,), float32
        Global per-cell hit counts in Morton order, suitable as the
        *weights* argument to :func:`weighted_local_keys` on the next frame.
    """
    if comm.Get_size() == 1:
        return local_hits.astype(np.float32)

    from gaio.mpi.gather import gather_sizes

    hits_i32 = np.asarray(local_hits, dtype=np.int32)
    counts, displs, total = gather_sizes(comm, len(hits_i32))

    g_hits = np.empty(total, dtype=np.int32)
    # Allgatherv with int32 counts and displs (mpi4py requires matching dtypes)
    comm.Allgatherv(
        hits_i32,
        [g_hits, (counts.astype(np.int32), displs.astype(np.int32))],
    )
    return g_hits.astype(np.float32)
