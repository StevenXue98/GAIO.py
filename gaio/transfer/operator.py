"""
gaio/transfer/operator.py
=========================
TransferOperator — sparse-matrix discretisation of the Perron-Frobenius
(transfer) operator.

Correspondence with GAIO.jl
----------------------------
``TransferOperator`` ↔ ``TransferOperator{B,T,S,M}`` in
``src/transfer_operator.jl``.

Matrix convention (matches Julia)
----------------------------------
``mat[i, j]`` = transition weight **from** domain cell ``j`` **to**
codomain cell ``i``.  Columns are normalised to sum to 1 (or 0 if no
test point leaves the domain).

    domain  (columns, source)  →
codomain    . . . . .
  (rows,    . . mat .
  target)   . . . . .
        ↓

Construction
------------
For each cell ``j`` in ``domain``:

1. Generate ``M`` test points by rescaling ``F._unit_points`` into cell ``j``.
2. Apply ``F.map`` to each test point.
3. Record how many hits land in each codomain cell ``i``.
4. Normalise column ``j`` by the total number of hits (column-stochastic).

Phase 4 MPI
-----------
When running under ``mpirun`` (or when an explicit *comm* is passed),
:func:`_build_transitions` distributes ``domain._keys`` across ranks via
Morton-order decomposition (see :mod:`gaio.mpi.decompose`).  Each rank
processes its local shard independently and contributes its COO entries to
a global ``Allgatherv`` (see :mod:`gaio.mpi.gather`).  Every rank then
assembles the same full sparse matrix, so all downstream spectral
operations (``eigs``, ``svds``) work unchanged on every rank.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from gaio.core.box import F64, I64
from gaio.core.boxset import BoxSet
from gaio.core.boxmeasure import BoxMeasure
from gaio.maps.base import SampledBoxMap


def _build_transitions(
    F: SampledBoxMap,
    domain: BoxSet,
    codomain: BoxSet,
    comm=None,
    partition_weights=None,
) -> tuple[NDArray, NDArray, NDArray, dict]:
    """
    Compute raw transition counts for building the transfer matrix.

    Returns COO arrays ``(row_indices, col_indices, values)`` where each
    entry records one test-point hit.  Repeated ``(i, j)`` pairs are
    summed by :func:`scipy.sparse.coo_matrix`.

    Parameters
    ----------
    F : SampledBoxMap (or AcceleratedBoxMap)
        The box map.  If ``F`` is an :class:`~gaio.cuda.AcceleratedBoxMap`
        with a GPU or CPU backend, Stage 2 (map application) runs on that
        backend — exactly as in :meth:`~gaio.maps.base.SampledBoxMap.map_boxes`.
    domain : BoxSet
        Source cells (columns of the matrix).
    codomain : BoxSet
        Target cells (rows of the matrix).
    comm : mpi4py communicator, _SerialComm, or None
        MPI communicator to use.  ``None`` (default) auto-detects via
        :func:`gaio.mpi.comm.get_comm`; pass ``False`` to force serial mode.

    Returns
    -------
    rows, cols, vals : ndarray
        COO triplets (un-normalised; each val = 1.0).  In MPI mode these
        are the **global** triplets gathered from all ranks (identical on
        every rank).

    Algorithm
    ---------
    Stage 1 — generate all (local_K × M) test points (vectorised NumPy).
    Stage 2 — call ``F._apply_map(test_pts)`` — GPU kernel if available.
    Stage 3 — vectorised key lookup + COO assembly.
    Stage 4 — MPI ``Allgatherv`` to collect global COO (serial: no-op).

    MPI domain decomposition
    ------------------------
    ``domain._keys`` are Morton-sorted and split across ranks before
    Stage 1.  Each rank processes its contiguous spatial shard; the column
    indices in the COO output use **global** domain indices (position in
    the full ``domain._keys`` array, not the local shard).  After
    ``Allgatherv``, every rank holds the complete transition data and can
    assemble the same global sparse matrix independently.
    """
    # ── Resolve communicator ─────────────────────────────────────────────────
    if comm is False:
        # Caller explicitly requested serial mode
        from gaio.mpi.comm import _SerialComm
        comm = _SerialComm()
    elif comm is None:
        from gaio.mpi.comm import get_comm
        comm = get_comm()

    mpi_size = comm.Get_size()
    mpi_rank = comm.Get_rank()

    # ── Global domain keys — needed for correct COO column indices ───────────
    all_domain_keys = domain._keys          # shape (K,), int64, sorted

    K     = len(all_domain_keys)
    M     = F.n_test_points
    ndim  = F.ndim

    if K == 0:
        empty_stats = {
            "n_ranks":           mpi_size,
            "per_rank_nnz":      np.zeros(mpi_size, dtype=np.int64),
            "total_nnz_raw":     0,
            "partition_weights": np.empty(0, dtype=np.float32),
        }
        return np.empty(0, I64), np.empty(0, I64), np.empty(0, F64), empty_stats

    unit_pts   = F._unit_points             # (M, ndim)
    cell_r_dom = domain.partition.cell_radius  # (ndim,)

    # ── MPI domain decomposition ─────────────────────────────────────────────
    # Morton-sort all domain keys for spatial locality, then take local shard.
    # Phase 5: when partition_weights is provided (and length matches K), use a
    # weighted prefix-sum split so each rank gets ≈ equal total hit density.
    # Phase 4 fallback (partition_weights=None): identical uniform K/P split.
    if mpi_size > 1:
        from gaio.mpi.decompose import morton_sort_keys, local_keys
        morton_keys = morton_sort_keys(all_domain_keys, domain.partition)
        if partition_weights is not None and len(partition_weights) == K:
            from gaio.mpi.load_balance import weighted_local_keys
            local_domain_keys = weighted_local_keys(
                morton_keys, partition_weights, mpi_rank, mpi_size
            )
        else:
            local_domain_keys = local_keys(morton_keys, mpi_rank, mpi_size)
        # Global column index for each local key: position in all_domain_keys
        local_col_offsets = np.searchsorted(all_domain_keys, local_domain_keys).astype(I64)
    else:
        local_domain_keys = all_domain_keys
        local_col_offsets = np.arange(K, dtype=I64)

    local_K = len(local_domain_keys)
    hits_per_cell = np.zeros(local_K, dtype=np.int32)   # Phase 5: per-cell hit count
    if local_K == 0:
        local_rows = np.empty(0, I64)
        local_cols = np.empty(0, I64)
        local_vals = np.empty(0, F64)
    else:
        # ── Stage 1: generate (local_K × M) test points ──────────────────────
        # Build a temporary BoxSet view with only the local keys so we can
        # call partition.centers() efficiently via the boxset helper.
        from gaio.core.boxset import BoxSet as _BoxSet
        local_bs  = _BoxSet(domain.partition, local_domain_keys)
        centers   = local_bs.centers()          # (local_K, ndim)

        test_pts = (
            centers[:, np.newaxis, :]
            + unit_pts[np.newaxis, :, :] * cell_r_dom[np.newaxis, np.newaxis, :]
        ).reshape(local_K * M, ndim)             # (local_K*M, ndim) C-contiguous

        # ── Stage 2: apply map — GPU kernel if F is AcceleratedBoxMap ────────
        mapped = F._apply_map(test_pts)          # (local_K*M, ndim)

        # ── Stage 3: vectorised key lookup + local COO assembly ──────────────
        hit_keys = codomain.partition.point_to_key_batch(mapped)   # (local_K*M,)
        cod_keys = codomain._keys                # sorted int64, shape (m,)
        m_cod    = len(cod_keys)

        valid = hit_keys >= 0
        if not valid.any():
            local_rows = np.empty(0, I64)
            local_cols = np.empty(0, I64)
            local_vals = np.empty(0, F64)
        else:
            hit_valid = hit_keys[valid]
            flat_idx  = np.where(valid)[0]
            # Local source-cell index within THIS rank's shard
            local_src = (flat_idx // M).astype(I64)
            # Map local index → global COO column
            col_valid = local_col_offsets[local_src]

            row_cand   = np.searchsorted(cod_keys, hit_valid)
            if m_cod == 0:
                matched = np.zeros(len(row_cand), dtype=bool)
            else:
                rc_clipped = np.minimum(row_cand, m_cod - 1)
                matched    = (row_cand < m_cod) & (cod_keys[rc_clipped] == hit_valid)

            local_rows = row_cand[matched].astype(I64)
            local_cols = col_valid[matched]
            local_vals = np.ones(len(local_rows), dtype=F64)

            # Phase 5: accumulate per-cell hit counts for next-frame weights.
            # local_src[matched] maps each successful hit → local shard index.
            # np.add.at is unbuffered: handles repeated indices correctly.
            if matched.any():
                np.add.at(hits_per_cell, local_src[matched], 1)

    # ── Stage 4: Allgatherv — collect global COO from all ranks ─────────────
    # allgather_coo returns a 4-tuple: (rows, cols, vals, per_rank_counts).
    # per_rank_counts[r] = number of COO entries contributed by rank r.
    # In serial mode (size==1) the fast-path returns inputs unchanged.
    from gaio.mpi.gather import allgather_coo
    rows, cols, vals, per_rank_counts = allgather_coo(
        comm, local_rows, local_cols, local_vals
    )

    # Phase 5: gather per-cell hit counts into a global weight array.
    from gaio.mpi.load_balance import compute_partition_weights
    new_partition_weights = compute_partition_weights(hits_per_cell, comm)

    mpi_stats = {
        "n_ranks":           mpi_size,
        "per_rank_nnz":      per_rank_counts,           # shape (n_ranks,)
        "total_nnz_raw":     int(per_rank_counts.sum()), # before matrix dedup
        "partition_weights": new_partition_weights,      # float32, shape (K,)
    }
    return rows, cols, vals, mpi_stats


class TransferOperator:
    """
    Sparse discretisation of the Perron-Frobenius operator.

    Parameters
    ----------
    F : SampledBoxMap
        The box map used to compute transitions.
    domain : BoxSet
        Source cells.  Must be on the same partition as *codomain*.
    codomain : BoxSet, optional
        Target cells.  If omitted, the codomain is set to the forward
        image of *domain* under *F* (same partition).
    comm : mpi4py communicator, _SerialComm, or None
        MPI communicator for distributed construction.  ``None`` (default)
        auto-detects via :func:`gaio.mpi.comm.get_comm`; pass ``False`` to
        force serial mode even when running under ``mpirun``.
    partition_weights : ndarray, shape (K,), float32 or None
        Phase 5 adaptive load balancing.  Pass ``T_prev.partition_weights``
        from a previous frame to replace the uniform Morton split with a
        weighted split that assigns keys proportionally to per-cell hit
        density.  ``None`` (default) is identical to Phase 4.  If length
        does not match ``len(domain)``, falls back silently to Phase 4.

    Attributes
    ----------
    domain : BoxSet
    codomain : BoxSet
    mat : scipy.sparse.csc_matrix
        Column-stochastic sparse matrix of shape ``(|codomain|, |domain|)``.
    mpi_stats : dict
        Per-rank construction statistics.  Keys:

        ``n_ranks``            — number of MPI ranks used (1 in serial mode).
        ``per_rank_nnz``       — ndarray, shape (n_ranks,); COO entries from each rank
                                 before sparse deduplication.
        ``total_nnz_raw``      — sum of per_rank_nnz (≥ mat.nnz due to repeated (i,j)).
        ``partition_weights``  — float32 ndarray, shape (K,); per-cell hit counts in
                                 Morton order.  Pass as *partition_weights* to the
                                 next frame's ``TransferOperator`` to activate Phase 5
                                 weighted decomposition.

    Examples
    --------
    >>> import numpy as np
    >>> from gaio.core.box import Box
    >>> from gaio.core.partition import BoxPartition
    >>> from gaio.core.boxset import BoxSet
    >>> from gaio.maps.base import SampledBoxMap
    >>> from gaio.transfer.operator import TransferOperator
    >>> domain = Box([0.0, 0.0], [1.0, 1.0])
    >>> f = lambda x: x * 0.5
    >>> pts = np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
    >>> F = SampledBoxMap(f, domain, pts)
    >>> P = BoxPartition(domain, [4, 4])
    >>> B = BoxSet.full(P)
    >>> T = TransferOperator(F, B, B)
    >>> T.mat.shape
    (16, 16)
    """

    __slots__ = ("boxmap", "domain", "codomain", "mat", "mpi_stats", "_comm")

    def __init__(
        self,
        F: SampledBoxMap,
        domain: BoxSet,
        codomain: BoxSet | None = None,
        comm=None,
        partition_weights=None,
    ) -> None:
        self.boxmap = F
        self.domain = domain

        # Resolve and store communicator for use in eigs() / svds()
        if comm is False:
            from gaio.mpi.comm import _SerialComm
            self._comm = _SerialComm()
        elif comm is None:
            from gaio.mpi.comm import get_comm
            self._comm = get_comm()
        else:
            self._comm = comm

        if codomain is None:
            codomain = F(domain)
        self.codomain = codomain

        rows, cols, vals, self.mpi_stats = _build_transitions(
            F, domain, codomain, comm=self._comm,
            partition_weights=partition_weights,
        )
        m = len(codomain)
        n = len(domain)

        # Sum repeated (i,j) entries, then normalise columns
        if len(rows) > 0:
            raw = sp.coo_matrix((vals, (rows, cols)), shape=(m, n)).tocsc()
        else:
            raw = sp.csc_matrix((m, n), dtype=F64)

        # Column-stochastic normalisation: divide each column by its sum
        col_sums = np.asarray(raw.sum(axis=0)).ravel()  # (n,)
        scale = np.zeros_like(col_sums)
        nonzero = col_sums != 0.0
        scale[nonzero] = 1.0 / col_sums[nonzero]
        self.mat: sp.csc_matrix = raw @ sp.diags(scale)

    # ------------------------------------------------------------------
    # Shape / repr
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return self.mat.shape

    def __repr__(self) -> str:
        m, n = self.shape
        return (
            f"TransferOperator({m}x{n}, "
            f"nnz={self.mat.nnz}, "
            f"domain={len(self.domain)} cells, "
            f"codomain={len(self.codomain)} cells)"
        )

    @property
    def partition_weights(self):
        """
        Per-cell hit-density weights for Phase 5 adaptive partitioning.

        A float32 array of shape ``(K,)`` (K = ``len(domain)``) in Morton
        order.  Each element is the number of test points from that cell
        that landed in the codomain during this construction pass.

        Pass this to the next frame's ``TransferOperator`` (or
        ``relative_attractor``) via the *partition_weights* argument to
        activate Phase 5 weighted decomposition.  Use
        :func:`gaio.mpi.load_balance.should_rebalance` to check whether
        the imbalance is large enough to make rebalancing worthwhile.

        Returns ``None`` when ``domain`` is empty (K=0).

        See Also
        --------
        gaio.mpi.load_balance.compute_imbalance
        gaio.mpi.load_balance.should_rebalance
        """
        weights = self.mpi_stats.get("partition_weights")
        if weights is None or len(weights) == 0:
            return None
        return weights

    # ------------------------------------------------------------------
    # Measure push/pull
    # ------------------------------------------------------------------

    def _domain_vec(self, measure: BoxMeasure) -> NDArray[F64]:
        """Build a dense vector indexed by domain keys from *measure*."""
        x = np.zeros(len(self.domain), dtype=F64)
        idxs = np.searchsorted(self.domain._keys, measure._keys)
        valid = (
            (idxs < len(self.domain._keys))
            & (self.domain._keys[idxs] == measure._keys)
        )
        x[idxs[valid]] = measure._weights[valid]
        return x

    def _codomain_vec(self, measure: BoxMeasure) -> NDArray[F64]:
        """Build a dense vector indexed by codomain keys from *measure*."""
        x = np.zeros(len(self.codomain), dtype=F64)
        idxs = np.searchsorted(self.codomain._keys, measure._keys)
        valid = (
            (idxs < len(self.codomain._keys))
            & (self.codomain._keys[idxs] == measure._keys)
        )
        x[idxs[valid]] = measure._weights[valid]
        return x

    def push_forward(self, measure: BoxMeasure) -> BoxMeasure:
        """
        Push-forward (T * μ): apply the transfer operator to *measure*.

        Each unit of mass at domain cell j is distributed to codomain cells
        according to column j of ``mat``.

        Matches Julia ``T * μ``.
        """
        x = self._domain_vec(measure)
        y = np.asarray(self.mat @ x).ravel()
        nonzero = y != 0.0
        return BoxMeasure(
            self.codomain.partition,
            self.codomain._keys[nonzero],
            y[nonzero],
        )

    def pull_back(self, measure: BoxMeasure) -> BoxMeasure:
        """
        Pull-back (T' * μ): apply the adjoint (Koopman) operator.

        Matches Julia ``T' * μ`` or ``T' * μ``.
        """
        x = self._codomain_vec(measure)
        y = np.asarray(self.mat.T @ x).ravel()
        nonzero = y != 0.0
        return BoxMeasure(
            self.domain.partition,
            self.domain._keys[nonzero],
            y[nonzero],
        )

    def __matmul__(self, measure: BoxMeasure) -> BoxMeasure:
        """T @ μ  ≡  push_forward(μ)."""
        return self.push_forward(measure)

    # ------------------------------------------------------------------
    # Spectral methods
    # ------------------------------------------------------------------

    def eigs(
        self,
        k: int = 3,
        which: str = "LM",
        v0: NDArray[F64] | None = None,
        **kwargs,
    ) -> tuple[NDArray, list[BoxMeasure]]:
        """
        Compute *k* eigenvalues and corresponding eigenmeasures.

        Uses SLEPc (distributed Krylov–Schur) when running under MPI with
        slepc4py installed; falls back to scipy ARPACK otherwise.

        Parameters
        ----------
        k : int
            Number of eigenvalues / eigenvectors.
        which : str
            Which eigenvalues to find (``'LM'``, ``'SM'``, ``'LR'``, ``'SR'``).
        v0 : ndarray, optional
            Starting vector for ARPACK (scipy fallback only; SLEPc uses its
            own initial vector strategy).

        Returns
        -------
        eigenvalues : ndarray, shape (k,)
        eigenmeasures : list of BoxMeasure, length k

        Notes
        -----
        Matches Julia's ``Arpack.eigs(gstar, ...)``.
        In MPI mode (comm.Get_size() > 1), uses
        :func:`gaio.mpi.distributed_eigs.distributed_eigs`.
        """
        from gaio.mpi.distributed_eigs import distributed_eigs
        if v0 is not None:
            kwargs.setdefault("v0", v0)
        return distributed_eigs(
            self.mat, k,
            self.domain._keys, self.domain.partition,
            comm=self._comm, which=which,
            **kwargs,
        )

    def svds(
        self, k: int = 3, **kwargs
    ) -> tuple[list[BoxMeasure], NDArray[F64], list[BoxMeasure]]:
        """
        Compute *k* singular values and vectors.

        Uses SLEPc Lanczos SVD when running under MPI with slepc4py;
        falls back to scipy ARPACK otherwise.

        Returns
        -------
        U     : list of BoxMeasure  (left singular vectors, codomain-indexed)
        sigma : ndarray, shape (k,)
        V     : list of BoxMeasure  (right singular vectors, domain-indexed)
        """
        from gaio.mpi.distributed_eigs import distributed_svds
        return distributed_svds(
            self.mat, k,
            self.domain._keys,   self.domain.partition,
            self.codomain._keys, self.codomain.partition,
            comm=self._comm,
            **kwargs,
        )
