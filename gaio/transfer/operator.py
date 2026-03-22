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

Phase 3 MPI target
------------------
The inner loop over ``domain._keys`` in :func:`_build_transitions` is
perfectly parallel — each source cell is independent.  For Phase 4,
distribute ``domain._keys`` across MPI ranks, build local COO entries,
then ``Allreduce`` and assemble the global sparse matrix.
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
) -> tuple[NDArray, NDArray, NDArray]:
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

    Returns
    -------
    rows, cols, vals : ndarray
        COO triplets (un-normalised; each val = 1.0).

    Algorithm
    ---------
    Mirrors the three-stage pipeline of :meth:`SampledBoxMap.map_boxes` but
    tracks the source cell index for each test point so that COO column
    indices can be reconstructed after the bulk map call:

    Stage 1 — generate ALL (K×M) test points at once (vectorised NumPy).
    Stage 2 — call ``F._apply_map(test_pts)`` — GPU if available.
    Stage 3 — vectorised key lookup + argsort-based COO assembly.

    Notes
    -----
    Phase 4 MPI: split ``domain._keys`` across ranks (Stage 1), each rank
    calls ``F._apply_map`` on its slice (Stage 2), then ``Allgatherv``
    the COO triplets before assembling the global matrix.
    """
    K   = len(domain._keys)
    M   = F.n_test_points
    ndim = F.ndim

    if K == 0:
        return np.empty(0, I64), np.empty(0, I64), np.empty(0, F64)

    unit_pts   = F._unit_points                      # (M, n)
    cell_r_dom = domain.partition.cell_radius        # (n,)

    # ── Stage 1: generate all (K×M, ndim) test points at once ───────────────
    # col_of[k*M + m] = k  (source cell index for COO column)
    centers  = domain.centers()                      # (K, ndim)
    test_pts = (
        centers[:, np.newaxis, :]
        + unit_pts[np.newaxis, :, :] * cell_r_dom[np.newaxis, np.newaxis, :]
    ).reshape(K * M, ndim)                           # (K*M, ndim) C-contiguous

    # ── Stage 2: apply map — GPU kernel if F is AcceleratedBoxMap ───────────
    mapped = F._apply_map(test_pts)                  # (K*M, ndim)

    # ── Stage 3: vectorised key lookup + COO assembly ────────────────────────
    hit_keys   = codomain.partition.point_to_key_batch(mapped)   # (K*M,)
    cod_keys   = codomain._keys                      # sorted int64, shape (m,)
    m_cod      = len(cod_keys)

    # Filter out misses (key == -1)
    valid      = hit_keys >= 0
    if not valid.any():
        return np.empty(0, I64), np.empty(0, I64), np.empty(0, F64)

    hit_valid  = hit_keys[valid]                     # (n_hits,)
    # Column index = source cell j = flat_index // M
    flat_idx   = np.where(valid)[0]
    col_valid  = (flat_idx // M).astype(I64)         # (n_hits,)

    # Binary-search hit keys into sorted codomain keys
    row_cand   = np.searchsorted(cod_keys, hit_valid)
    rc_clipped = np.minimum(row_cand, m_cod - 1)
    matched    = (row_cand < m_cod) & (cod_keys[rc_clipped] == hit_valid)

    rows = row_cand[matched].astype(I64)
    cols = col_valid[matched]
    vals = np.ones(len(rows), dtype=F64)
    return rows, cols, vals


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

    Attributes
    ----------
    domain : BoxSet
    codomain : BoxSet
    mat : scipy.sparse.csc_matrix
        Column-stochastic sparse matrix of shape ``(|codomain|, |domain|)``.

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

    __slots__ = ("boxmap", "domain", "codomain", "mat")

    def __init__(
        self,
        F: SampledBoxMap,
        domain: BoxSet,
        codomain: BoxSet | None = None,
    ) -> None:
        self.boxmap = F
        self.domain = domain

        if codomain is None:
            codomain = F(domain)
        self.codomain = codomain

        rows, cols, vals = _build_transitions(F, domain, codomain)
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

        Parameters
        ----------
        k : int
            Number of eigenvalues / eigenvectors.
        which : str
            Which eigenvalues to find (``'LM'``, ``'SM'``, etc.).
        v0 : ndarray, optional
            Starting vector for the iterative solver.

        Returns
        -------
        eigenvalues : ndarray, shape (k,)
        eigenmeasures : list of BoxMeasure, length k

        Notes
        -----
        Matches Julia's ``Arpack.eigs(gstar, ...)``.
        """
        n = len(self.domain)
        if v0 is None:
            v0 = np.ones(n, dtype=F64)
        vals, vecs = spla.eigs(self.mat, k=k, which=which, v0=v0, **kwargs)
        measures = [
            BoxMeasure(
                self.domain.partition,
                self.domain._keys.copy(),
                vecs[:, i].real.astype(F64),
            )
            for i in range(k)
        ]
        return vals, measures

    def svds(
        self, k: int = 3, **kwargs
    ) -> tuple[list[BoxMeasure], NDArray[F64], list[BoxMeasure]]:
        """
        Compute *k* singular values and vectors.

        Returns
        -------
        U : list of BoxMeasure  (left singular vectors, indexed by codomain)
        sigma : ndarray, shape (k,)
        V : list of BoxMeasure  (right singular vectors, indexed by domain)
        """
        u, s, vt = spla.svds(self.mat, k=k, **kwargs)
        U = [
            BoxMeasure(
                self.codomain.partition,
                self.codomain._keys.copy(),
                u[:, i].astype(F64),
            )
            for i in range(k)
        ]
        V = [
            BoxMeasure(
                self.domain.partition,
                self.domain._keys.copy(),
                vt[i, :].astype(F64),
            )
            for i in range(k)
        ]
        return U, s, V
