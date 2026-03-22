"""
gaio/algorithms/attractor.py
==============================
relative_attractor — outer approximation of the maximal backward-invariant
set (ω-limit set) via iterated subdivision and intersection.

Correspondence with GAIO.jl
----------------------------
``relative_attractor`` ↔ ``ω`` / ``relative_attractor`` in
``src/algorithms/invariant_sets.jl`` with ``subdivision=true``.

    const relative_attractor = ω

    function ω(F, B::BoxSet; subdivision=true, steps=12)
        iter = (S -> S ∩ F(S)) ∘ subdivide
        return iterate_until_equal(iter, B; max_iterations=steps)
    end

Algorithm
---------
Each iteration:

1. Subdivide B along dimension ``argmin(dims)`` — matches Julia's
   ``subdivide(B)`` which calls ``subdivide(B, argmin(B.partition.dims))``.
2. Restrict to cells whose forward image re-enters B:  B ← B ∩ F(B).

Since the partition changes at every step (dims grow), the Julia
``iterate_until_equal`` fixed-point check never triggers when
``subdivision=true``; all ``steps`` iterations are always performed.

Phase 4 MPI
-----------
When running under ``mpirun`` (or when ``comm`` is passed explicitly),
the map-application step ``F(B)`` is distributed across ranks via
:func:`gaio.mpi.distributed_attractor.distributed_relative_attractor`.
Each rank processes only its Morton-order shard (K/P cells), reducing
per-rank test-point memory by a factor of P.
"""
from __future__ import annotations

import numpy as np

from gaio.core.boxset import BoxSet
from gaio.maps.base import SampledBoxMap


def relative_attractor(
    F: SampledBoxMap,
    B0: BoxSet,
    steps: int = 12,
    comm=None,
) -> BoxSet:
    """
    Compute an outer approximation of the relative attractor (ω-limit set)
    of *F* within *B0*.

    Parameters
    ----------
    F : SampledBoxMap
        The box map to iterate.
    B0 : BoxSet
        Initial covering set (typically ``BoxSet.full(partition)`` for a
        coarse 1×1…×1 partition covering the domain of interest).
    steps : int, optional
        Number of subdivision–intersection iterations.  Default: 12.
        Matches GAIO.jl ``ω(...; subdivision=true, steps=12)``.
    comm : mpi4py communicator, _SerialComm, or None
        MPI communicator.  ``None`` (default) auto-detects via
        :func:`gaio.mpi.comm.get_comm`; pass ``False`` to force serial
        mode even when running under ``mpirun``.

    Returns
    -------
    BoxSet
        Outer approximation on the refined partition after ``steps``
        subdivision steps.  Identical on every MPI rank.

    Notes
    -----
    The partition is refined at each step along ``argmin(dims)`` —
    the dimension with the fewest cells — exactly matching Julia's
    ``subdivide(B)`` default.  No early-exit is performed; all ``steps``
    iterations always execute.

    When running under MPI (``comm.Get_size() > 1``) the implementation
    delegates to :func:`gaio.mpi.distributed_attractor.distributed_relative_attractor`
    which distributes the ``F(B)`` map step across ranks.

    Examples
    --------
    >>> import numpy as np
    >>> from gaio.core.box import Box
    >>> from gaio.core.partition import BoxPartition
    >>> from gaio.core.boxset import BoxSet
    >>> from gaio.maps.base import SampledBoxMap
    >>> domain = Box([0.0, 0.0], [1.0, 1.0])
    >>> f = lambda x: np.array([x[0], x[1] * 0.5])
    >>> pts = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    >>> g = SampledBoxMap(f, domain, pts)
    >>> P = BoxPartition(domain, [1, 1])
    >>> rga = relative_attractor(g, BoxSet.full(P), steps=4)
    >>> rga.partition.dims.tolist()
    [4, 4]
    """
    # ── Resolve communicator ─────────────────────────────────────────────────
    if comm is False:
        _use_mpi = False
        _comm    = None
    else:
        from gaio.mpi.comm import get_comm, is_mpi_active as _is_mpi_active
        _comm = get_comm() if comm is None else comm
        _use_mpi = _comm.Get_size() > 1

    if _use_mpi:
        from gaio.mpi.distributed_attractor import distributed_relative_attractor
        return distributed_relative_attractor(F, B0, steps=steps, comm=_comm)

    # Serial path (unchanged)
    B = B0
    for _ in range(steps):
        dim = int(np.argmin(B.partition.dims))
        B = B.subdivide(dim)
        B = B & F(B)
    return B
