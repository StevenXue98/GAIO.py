"""
gaio/mpi/decompose.py
=====================
Morton (Z-order) curve domain decomposition for MPI rank assignment.

Why Morton order?
-----------------
The flat int64 keys used by ``BoxPartition`` are assigned in C row-major
order (the last dimension varies fastest).  A simple round-robin split of
these keys gives each rank a set of thin horizontal slices — long-range
neighbours in all but the last dimension.

Morton codes interleave the bits of each dimension's cell index, mapping
the N-dimensional grid onto a 1-D curve that visits nearby cells in nearby
positions.  Splitting by contiguous Morton ranges gives each rank a
spatially compact subdomain, which:

1. Minimises the number of "boundary" cells whose images land on another
   rank's subdomain (reduces ``Allgatherv`` traffic from O(K) to O(K^{(d-1)/d})).
2. Improves GPU memory-access locality (coalesced loads from nearby cells).

Algorithm
---------
1. Decode each flat key to its d-dimensional multi-index via
   ``np.unravel_index(keys, partition.dims)``.
2. Interleave the bits of the multi-index components to produce a uint64
   Morton code per cell (up to 21 bits per dimension → 63-bit code for 3-D).
3. Sort keys by their Morton code (stable sort preserves tie ordering).
4. Split the sorted array into ``size`` contiguous chunks of nearly equal
   length and return the chunk belonging to ``rank``.

Public API
----------
    morton_encode(multi_idx)             — (N, d) int → (N,) uint64 codes
    morton_sort_keys(keys, partition)    — sort flat keys by Morton code
    local_keys(keys, rank, size)         — contiguous slice for one rank
    decompose(keys, partition, rank, size) — convenience: sort then slice
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gaio.core.box import I64


# ---------------------------------------------------------------------------
# Morton bit-interleaving
# ---------------------------------------------------------------------------

def _spread_bits(x: NDArray[np.uint64], stride: int) -> NDArray[np.uint64]:
    """
    Spread the bits of integer array *x* by inserting *(stride-1)* zero bits
    between each original bit.

    For stride=2 (2-D): bit k of x  →  bit 2k of result.
    For stride=3 (3-D): bit k of x  →  bit 3k of result.

    Handles up to 21 input bits per element (safe for typical partition dims
    ≤ 2,097,152; beyond that, higher bits are silently discarded).

    Parameters
    ----------
    x      : ndarray, uint64, any shape
    stride : int ≥ 1

    Returns
    -------
    ndarray, uint64, same shape as *x*
    """
    x = np.asarray(x, dtype=np.uint64)
    result = np.zeros_like(x, dtype=np.uint64)
    for bit in range(21):
        mask = np.uint64(1) << np.uint64(bit)
        result |= (x & mask) << np.uint64(bit * (stride - 1))
    return result


def morton_encode(multi_idx: NDArray) -> NDArray[np.uint64]:
    """
    Compute Morton (Z-order) codes for an array of grid multi-indices.

    Parameters
    ----------
    multi_idx : ndarray, shape (N, d), non-negative integers
        One row per cell; each column is the index along one dimension.

    Returns
    -------
    codes : ndarray, shape (N,), dtype uint64

    Examples
    --------
    >>> morton_encode(np.array([[0, 0], [1, 0], [0, 1], [1, 1]]))
    array([0, 1, 2, 3], dtype=uint64)
    """
    multi_idx = np.asarray(multi_idx, dtype=np.uint64)
    if multi_idx.ndim == 1:
        multi_idx = multi_idx[:, np.newaxis]
    N, d = multi_idx.shape
    codes = np.zeros(N, dtype=np.uint64)
    for dim in range(d):
        # dim-th component occupies bit positions dim, dim+d, dim+2d, ...
        codes |= _spread_bits(multi_idx[:, dim], stride=d) << np.uint64(dim)
    return codes


# ---------------------------------------------------------------------------
# Key sorting and range slicing
# ---------------------------------------------------------------------------

def morton_sort_keys(
    keys: NDArray[I64],
    partition,            # gaio.core.partition.BoxPartition
) -> NDArray[I64]:
    """
    Return *keys* re-ordered by their Morton code within *partition*.

    Parameters
    ----------
    keys      : ndarray, shape (K,), int64 — flat partition keys, any order
    partition : BoxPartition

    Returns
    -------
    sorted_keys : ndarray, shape (K,), int64
    """
    dims = np.asarray(partition.dims, dtype=np.int64)   # shape (d,)
    # Decode flat keys → multi-indices; unravel_index returns d arrays of len K
    multi_idx = np.stack(
        np.unravel_index(keys, dims), axis=1
    ).astype(np.uint64)                                  # (K, d)
    codes = morton_encode(multi_idx)                     # (K,) uint64
    order = np.argsort(codes, kind="stable")
    return keys[order]


def local_keys(
    keys: NDArray[I64],
    rank: int,
    size: int,
) -> NDArray[I64]:
    """
    Return the contiguous slice of Morton-sorted *keys* owned by *rank*.

    The ``K`` keys are split as evenly as possible: the first ``K % size``
    ranks each own one extra key (``numpy.array_split`` semantics).

    Parameters
    ----------
    keys : ndarray, shape (K,), int64 — should be Morton-sorted already
    rank : int   — 0 ≤ rank < size
    size : int   — total number of ranks

    Returns
    -------
    ndarray, shape (local_K,), int64
    """
    K = len(keys)
    if size == 1:
        return keys
    base, extra = divmod(K, size)
    start = rank * base + min(rank, extra)
    end   = start + base + (1 if rank < extra else 0)
    return keys[start:end]


def decompose(
    keys: NDArray[I64],
    partition,
    rank: int,
    size: int,
) -> NDArray[I64]:
    """
    Convenience wrapper: Morton-sort *keys* then return the local shard.

    Parameters
    ----------
    keys      : ndarray, shape (K,), int64
    partition : BoxPartition
    rank      : int
    size      : int

    Returns
    -------
    ndarray, shape (local_K,), int64 — Morton-sorted local shard
    """
    sorted_keys = morton_sort_keys(keys, partition)
    return local_keys(sorted_keys, rank, size)
