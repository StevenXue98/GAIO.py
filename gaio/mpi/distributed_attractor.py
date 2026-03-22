"""
gaio/mpi/distributed_attractor.py
===================================
Distributed ``relative_attractor`` — each MPI rank processes only its
Morton-order shard of the active BoxSet at each subdivision step.

Memory model
------------
The full BoxSet ``S`` (a sorted int64 key array) is replicated on every
rank.  At K=1M cells the key array is only 8 MB — negligible.  The memory
saving comes from the **test-point array**:

    Serial  : K × M × d × 8 bytes  (e.g. 1M × 27 × 3 × 8 = 648 MB)
    MPI/P   : (K/P) × M × d × 8 bytes  (e.g. 162 MB with P=4)

Each rank builds and maps only its ``K/P`` test points, so the GPU VRAM
requirement scales as 1/P.

Algorithm (one subdivision step)
----------------------------------
1. ``S = S.subdivide(dim)``            — deterministic, all ranks do it
2. Distribute S by Morton order        — local_S has K/P keys
3. ``local_image = F(local_S)``        — GPU kernel on K/P × M test points
4. Allgatherv local image keys         — O(K_image × 8 bytes) traffic
5. ``global_image = union(all keys)``  — np.unique, in memory
6. ``S = S & global_image``            — sorted-array intersection, O(K)

Convergence guarantee
---------------------
The outer approximation guarantee is preserved: a cell survives iff *any*
test point from it lands in S.  Because test points are disjoint between
ranks (each cell is owned by exactly one rank), the union of all image sets
equals the image of the full S.  No cell is incorrectly included or excluded.

Public API
----------
    distributed_relative_attractor(F, B0, steps, comm=None)
    _allgather_keys(comm, local_keys)   — reusable key-gather helper
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gaio.core.box import I64
from gaio.core.boxset import BoxSet
from gaio.maps.base import SampledBoxMap


# ---------------------------------------------------------------------------
# Key-gather helper (reused by other algorithms)
# ---------------------------------------------------------------------------

def _allgather_keys(comm, local_keys: NDArray[I64]) -> NDArray[I64]:
    """
    Allgatherv a variable-length int64 key array from all ranks.

    Returns the concatenated array (not de-duplicated) on every rank.
    Use ``np.unique`` afterwards to get the union.

    Parameters
    ----------
    comm       : mpi4py communicator or _SerialComm
    local_keys : ndarray, int64 — keys owned by this rank

    Returns
    -------
    all_keys : ndarray, int64, shape (total,)
    """
    if comm.Get_size() == 1:
        return local_keys

    from gaio.mpi.gather import gather_sizes
    counts, displs, total = gather_sizes(comm, len(local_keys))
    g_keys = np.empty(total, dtype=I64)
    comm.Allgatherv(local_keys, [g_keys, (counts, displs)])
    return g_keys


# ---------------------------------------------------------------------------
# Distributed relative attractor
# ---------------------------------------------------------------------------

def distributed_relative_attractor(
    F: SampledBoxMap,
    B0: BoxSet,
    steps: int = 12,
    comm=None,
) -> BoxSet:
    """
    Compute an outer approximation of the relative attractor with MPI.

    Distributes the map-application step (``F(S)``) across MPI ranks while
    keeping the full BoxSet S replicated on every rank.  All ranks return
    the same BoxSet result.

    Parameters
    ----------
    F : SampledBoxMap or AcceleratedBoxMap
        The box map.  GPU acceleration is used automatically if ``F`` is an
        ``AcceleratedBoxMap(backend='gpu')``.
    B0 : BoxSet
        Initial covering set (typically ``BoxSet.full(P)``).
    steps : int
        Subdivision iterations.  Default: 12.
    comm : mpi4py communicator, _SerialComm, or None
        ``None`` (default) auto-detects via :func:`gaio.mpi.comm.get_comm`.
        Pass ``False`` to force serial mode.

    Returns
    -------
    BoxSet
        Outer approximation of the attractor on the refined partition.
        Identical on every rank.

    Notes
    -----
    Each rank selects its shard by Morton-order decomposition (see
    :mod:`gaio.mpi.decompose`).  The shard is recomputed each step since
    the partition changes after every subdivision; Morton sorting is O(K log K)
    and cheap compared to the map evaluation.
    """
    # ── Resolve communicator ─────────────────────────────────────────────────
    if comm is False:
        from gaio.mpi.comm import _SerialComm
        comm = _SerialComm()
    elif comm is None:
        from gaio.mpi.comm import get_comm
        comm = get_comm()

    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    # Serial fast-path — identical to the non-MPI implementation
    if mpi_size == 1:
        B = B0
        for _ in range(steps):
            dim = int(np.argmin(B.partition.dims))
            B = B.subdivide(dim)
            B = B & F(B)
        return B

    from gaio.mpi.decompose import decompose as morton_decompose

    B = B0
    for step in range(steps):
        # ── 1. Subdivide (deterministic — same on all ranks) ─────────────────
        dim = int(np.argmin(B.partition.dims))
        B = B.subdivide(dim)

        if len(B) == 0:
            break   # no cells left — attractor is empty

        # ── 2. Distribute by Morton order — each rank owns K/P cells ─────────
        local_keys = morton_decompose(B._keys, B.partition, mpi_rank, mpi_size)

        if len(local_keys) == 0:
            # This rank has no cells this step; contribute empty image
            local_image_keys = np.empty(0, dtype=I64)
        else:
            local_S    = BoxSet(B.partition, local_keys)

            # ── 3. Apply map to local shard (GPU kernel if available) ─────────
            local_image = F(local_S)
            local_image_keys = local_image._keys   # sorted int64

        # ── 4. Allgatherv image keys from all ranks ───────────────────────────
        all_image_keys = _allgather_keys(comm, local_image_keys)

        # ── 5. Union → global image BoxSet ────────────────────────────────────
        union_keys = np.unique(all_image_keys)

        # ── 6. Intersect: B = B ∩ global_image ───────────────────────────────
        image_set = BoxSet(B.partition, union_keys)
        B = B & image_set

    return B
