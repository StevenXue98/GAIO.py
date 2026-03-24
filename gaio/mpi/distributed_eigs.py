"""
gaio/mpi/distributed_eigs.py
==============================
Distributed eigensolve and SVD using SLEPc/PETSc, with an automatic
fallback to scipy.sparse.linalg when SLEPc is not installed.

Why SLEPc/PETSc?
-----------------
For K > 500K cells the Perron-Frobenius matrix is a (~500K × 500K) sparse
matrix.  ARPACK (scipy) solves this on a single core with O(K × k) memory
for the Krylov basis.  At K=500K, k=6, that is 3 GB just for the basis —
often more than a single GPU's VRAM.

SLEPc distributes both the matrix (row-partitioned across ranks) and the
Krylov subspace vectors.  Each rank stores only a 1/P slice of the matrix
rows and the same 1/P slice of every basis vector.  Memory scales as 1/P.

SLEPc also implements Krylov–Schur (a refined Arnoldi variant) which is
more numerically stable than vanilla Arnoldi for non-Hermitian problems,
important because our transfer operator is column-stochastic but not
symmetric.

Initialization
--------------
petsc4py / slepc4py must be initialized before mpi4py creates its first
communicator in some configurations (when PETSc is built without
``--with-mpi-dir``).  We call ``petsc4py.init([])`` lazily on first use,
which is safe when mpi4py has already been imported (the two share the same
underlying MPI_COMM_WORLD).

Matrix construction
--------------------
We receive the assembled scipy CSC matrix (already on every rank after the
Allgatherv step) and convert it to a distributed PETSc MPIAIJ matrix.
Rank *r* inserts only the rows it owns; PETSc's assembly handles the rest.

Row ownership follows the same ``array_split`` semantics used by
:func:`gaio.mpi.decompose.local_keys`:

    base, extra = divmod(global_m, size)
    row_start[r] = r * base + min(r, extra)
    row_end[r]   = row_start[r] + base + (1 if r < extra else 0)

Public API
----------
    slepc_available()                     — bool: SLEPc import check
    distributed_eigs(mat, k, comm, ...)   — k eigenvalues + measures
    distributed_svds(mat, k, comm, ...)   — k singular triplets
"""
from __future__ import annotations

from typing import Optional
import numpy as np
from numpy.typing import NDArray

import scipy.sparse as sp

from gaio.core.box import F64, I64
from gaio.core.boxset import BoxSet
from gaio.core.boxmeasure import BoxMeasure


# ---------------------------------------------------------------------------
# Availability probe
# ---------------------------------------------------------------------------

def slepc_available() -> bool:
    """Return True iff both petsc4py and slepc4py can be imported."""
    try:
        import petsc4py   # noqa: F401
        import slepc4py   # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Row-range helper (mirrors decompose.local_keys semantics)
# ---------------------------------------------------------------------------

def _row_range(global_m: int, rank: int, size: int) -> tuple[int, int]:
    """Return (row_start, row_end) for *rank* in a global matrix of *global_m* rows."""
    base, extra = divmod(global_m, size)
    start = rank * base + min(rank, extra)
    end   = start + base + (1 if rank < extra else 0)
    return int(start), int(end)


# ---------------------------------------------------------------------------
# PETSc Mat builder
# ---------------------------------------------------------------------------

def _build_petsc_mat(
    mat_scipy: sp.csc_matrix,
    comm,
) -> "petsc4py.PETSc.Mat":
    """
    Convert a scipy CSC matrix into a distributed PETSc MPIAIJ matrix.

    Each rank inserts only the rows it owns.  PETSc assembly via
    ``assemblyBegin/End`` handles the off-process communication needed to
    finalize the distributed structure.

    Parameters
    ----------
    mat_scipy : scipy.sparse.csc_matrix (or any format; converted internally)
    comm      : mpi4py communicator (must wrap MPI_COMM_WORLD)

    Returns
    -------
    petsc4py.PETSc.Mat — assembled, read-only after this call
    """
    from petsc4py import PETSc

    # PETSc works best with CSR; convert once
    mat_csr = mat_scipy.tocsr().astype(np.float64)
    global_m, global_n = mat_csr.shape

    r     = comm.Get_rank()
    size  = comm.Get_size()
    r_lo, r_hi = _row_range(global_m, r, size)
    local_m = r_hi - r_lo

    # ── Create parallel AIJ matrix ────────────────────────────────────────────
    A = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    A.setSizes(((local_m, global_m), (PETSc.DECIDE, global_n)))
    A.setType(PETSc.Mat.Type.MPIAIJ)
    A.setFromOptions()
    A.setUp()

    # ── Insert local rows ─────────────────────────────────────────────────────
    local_slice = mat_csr[r_lo:r_hi, :]          # (local_m, global_n) CSR
    coo         = local_slice.tocoo()             # (row, col, val) — local rows only
    global_rows = (coo.row + r_lo).astype(np.int32)
    global_cols = coo.col.astype(np.int32)
    values      = coo.data.astype(np.float64)

    if len(values) > 0:
        A.setValues(global_rows, global_cols, values,
                    addv=PETSc.InsertMode.INSERT_VALUES)

    A.assemblyBegin(PETSc.Mat.AssemblyType.FINAL_ASSEMBLY)
    A.assemblyEnd(PETSc.Mat.AssemblyType.FINAL_ASSEMBLY)
    return A


def _build_petsc_mat_from_coo(
    local_rows: NDArray[I64],
    local_cols: NDArray[I64],
    local_vals: NDArray[F64],
    global_m: int,
    global_n: int,
    comm,
) -> "petsc4py.PETSc.Mat":
    """
    Build a distributed, column-normalised PETSc MPIAIJ matrix from local COO.

    Each rank supplies only the edges it computed (its Morton-order shard of
    domain columns).  PETSc's ``assemblyBegin/End`` handles off-process row
    insertions — entries whose row is owned by another rank are buffered and
    communicated automatically.  Column normalisation (column-stochastic) is
    applied in-place after assembly using ``multTranspose`` + ``diagonalScale``.

    This avoids the ``Allgatherv`` / global scipy CSC round-trip required by
    :func:`_build_petsc_mat`: each rank contributes O(nnz/P) entries rather
    than O(nnz), and PETSc never receives the full matrix.

    Parameters
    ----------
    local_rows, local_cols : ndarray, int64
        Global row and column indices of COO entries computed by this rank.
        Intra-rank duplicate ``(i, j)`` pairs (from multiple test points
        landing in the same target cell) are summed via ``ADD_VALUES``.
        No cross-rank ``(i, j)`` duplicates exist — each column ``j`` belongs
        to exactly one rank via Morton decomposition.
    local_vals : ndarray, float64
        Raw hit counts (all 1.0); normalisation is applied post-assembly.
    global_m, global_n : int
        Full matrix dimensions (``len(codomain)``, ``len(domain)``).
    comm : mpi4py communicator

    Returns
    -------
    petsc4py.PETSc.Mat — assembled and column-normalised
    """
    from petsc4py import PETSc

    r     = comm.Get_rank()
    size  = comm.Get_size()
    r_lo, r_hi = _row_range(global_m, r, size)
    local_m = r_hi - r_lo

    A = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    A.setSizes(((local_m, global_m), (PETSc.DECIDE, global_n)))
    A.setType(PETSc.Mat.Type.MPIAIJ)
    A.setFromOptions()
    A.setUp()

    # ADD_VALUES: intra-rank duplicate (i,j) pairs are summed correctly.
    # Off-process rows (owned by other ranks) are buffered and communicated
    # by assemblyBegin/End — no explicit Allgatherv needed.
    if len(local_vals) > 0:
        A.setValues(
            local_rows.astype(np.int32),
            local_cols.astype(np.int32),
            local_vals.astype(np.float64),
            addv=PETSc.InsertMode.ADD_VALUES,
        )

    A.assemblyBegin(PETSc.Mat.AssemblyType.FINAL_ASSEMBLY)
    A.assemblyEnd(PETSc.Mat.AssemblyType.FINAL_ASSEMBLY)

    # ── Column-stochastic normalisation ───────────────────────────────────────
    # col_sums[j] = sum_i A[i,j].  Divide each column j by col_sums[j],
    # leaving zero columns unchanged (matches TransferOperator scipy path).
    ones     = A.createVecLeft()    # Vec of length global_m, all 1s
    ones.set(1.0)
    col_sums = A.createVecRight()   # Vec of length global_n — filled by multTranspose
    A.multTranspose(ones, col_sums)

    arr = col_sums.getArray().copy()
    nz  = arr != 0.0
    arr[nz] = 1.0 / arr[nz]
    col_sums.setArray(arr)
    col_sums.assemble()

    A.diagonalScale(l=None, r=col_sums)   # A[:,j] *= 1/col_sums[j]

    ones.destroy()
    col_sums.destroy()
    return A


# ---------------------------------------------------------------------------
# SLEPc eigensolve
# ---------------------------------------------------------------------------

def _slepc_eigs(
    mat_scipy: sp.csc_matrix,
    k: int,
    which: str,
    domain_keys: NDArray[I64],
    domain_partition,
    comm,
    local_coo=None,
) -> tuple[NDArray, list[BoxMeasure]]:
    """
    Compute *k* eigenvalues / eigenvectors of *mat_scipy* via SLEPc EPS.

    Parameters
    ----------
    mat_scipy        : scipy sparse, shape (n, n) — column-stochastic transfer matrix
    k                : number of eigenvalues
    which            : 'LM' (largest magnitude), 'SM', etc.  Mapped to SLEPc.
    domain_keys      : sorted int64 keys (for wrapping eigenvectors as BoxMeasure)
    domain_partition : BoxPartition
    comm             : mpi4py communicator
    local_coo        : tuple (local_rows, local_cols, local_vals) or None
        When provided, PETSc is built directly from per-rank local COO edges,
        skipping the Allgatherv / global scipy intermediate.  When None,
        falls back to :func:`_build_petsc_mat` (scipy → PETSc conversion).

    Returns
    -------
    eigenvalues : ndarray, shape (nconv,)
    eigenmeasures : list of BoxMeasure, length nconv
    """
    from petsc4py import PETSc
    from slepc4py import SLEPc

    if local_coo is not None:
        local_rows, local_cols, local_vals = local_coo
        A = _build_petsc_mat_from_coo(
            local_rows, local_cols, local_vals,
            mat_scipy.shape[0], mat_scipy.shape[1], comm,
        )
    else:
        A = _build_petsc_mat(mat_scipy, comm)

    eps = SLEPc.EPS()
    eps.create(comm=PETSc.COMM_WORLD)
    eps.setOperators(A)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)   # non-Hermitian
    eps.setDimensions(nev=k)

    # Map scipy which-string to SLEPc enum
    _which_map = {
        "LM": SLEPc.EPS.Which.LARGEST_MAGNITUDE,
        "SM": SLEPc.EPS.Which.SMALLEST_MAGNITUDE,
        "LR": SLEPc.EPS.Which.LARGEST_REAL,
        "SR": SLEPc.EPS.Which.SMALLEST_REAL,
    }
    eps.setWhichEigenpairs(_which_map.get(which, SLEPc.EPS.Which.LARGEST_MAGNITUDE))
    eps.setTolerances(tol=1e-10, max_it=500)
    eps.setFromOptions()
    eps.solve()

    nconv = eps.getConverged()
    n     = mat_scipy.shape[0]

    eigenvalues   = []
    eigenmeasures = []
    vr = A.createVecRight()
    vi = A.createVecRight()

    r    = comm.Get_rank()
    r_lo, r_hi = _row_range(n, r, comm.Get_size())

    for i in range(min(nconv, k)):
        lam = eps.getEigenpair(i, vr, vi)
        eigenvalues.append(complex(lam))

        # Gather full eigenvector to all ranks via PETSc scatter
        vr_seq  = _petsc_scatter_to_all(vr, n)
        weights = np.array(vr_seq.getArray(), dtype=F64)
        eigenmeasures.append(
            BoxMeasure(domain_partition, domain_keys.copy(), weights)
        )

    vr.destroy(); vi.destroy()
    eps.destroy(); A.destroy()

    return np.array(eigenvalues), eigenmeasures


def _slepc_svds(
    mat_scipy: sp.csc_matrix,
    k: int,
    domain_keys: NDArray[I64],
    domain_partition,
    codomain_keys: NDArray[I64],
    codomain_partition,
    comm,
    local_coo=None,
) -> tuple[list[BoxMeasure], NDArray[F64], list[BoxMeasure]]:
    """
    Compute *k* singular triplets of *mat_scipy* via SLEPc SVD.

    Parameters
    ----------
    local_coo : tuple (local_rows, local_cols, local_vals) or None
        When provided, PETSc is built directly from per-rank local COO edges.
        See :func:`_slepc_eigs` for full description.

    Returns
    -------
    U : list of BoxMeasure   (left singular vectors,  indexed by codomain)
    sigma : ndarray, shape (nconv,)
    V : list of BoxMeasure   (right singular vectors, indexed by domain)
    """
    from petsc4py import PETSc
    from slepc4py import SLEPc

    if local_coo is not None:
        local_rows, local_cols, local_vals = local_coo
        A = _build_petsc_mat_from_coo(
            local_rows, local_cols, local_vals,
            mat_scipy.shape[0], mat_scipy.shape[1], comm,
        )
    else:
        A = _build_petsc_mat(mat_scipy, comm)

    svd = SLEPc.SVD()
    svd.create(comm=PETSc.COMM_WORLD)
    svd.setOperator(A)
    svd.setType(SLEPc.SVD.Type.LANCZOS)
    svd.setDimensions(nsv=k)
    svd.setTolerances(tol=1e-10, max_it=500)
    svd.setFromOptions()
    svd.solve()

    nconv = svd.getConverged()
    m, n  = mat_scipy.shape

    U_list   = []
    sigma    = []
    V_list   = []

    u_vec = A.createVecLeft()
    v_vec = A.createVecRight()

    for i in range(min(nconv, k)):
        sig = svd.getSingularTriplet(i, u_vec, v_vec)
        sigma.append(float(sig))

        u_seq  = _petsc_scatter_to_all(u_vec, m)
        v_seq  = _petsc_scatter_to_all(v_vec, n)

        U_list.append(
            BoxMeasure(codomain_partition, codomain_keys.copy(),
                       np.array(u_seq.getArray(), dtype=F64))
        )
        V_list.append(
            BoxMeasure(domain_partition, domain_keys.copy(),
                       np.array(v_seq.getArray(), dtype=F64))
        )

    u_vec.destroy(); v_vec.destroy()
    svd.destroy(); A.destroy()

    return U_list, np.array(sigma, dtype=F64), V_list


# ---------------------------------------------------------------------------
# PETSc scatter helper — replicate a distributed Vec on every rank
# ---------------------------------------------------------------------------

def _petsc_scatter_to_all(vec_dist, global_n: int):
    """
    Scatter a distributed PETSc Vec into a sequential Vec replicated on
    every MPI rank.

    Parameters
    ----------
    vec_dist : PETSc.Vec   — distributed (parallel) input vector
    global_n : int         — global length

    Returns
    -------
    PETSc.Vec — sequential Vec, same data on all ranks.
    Note: caller is responsible for calling .destroy() on the result.
    """
    from petsc4py import PETSc

    vec_seq = PETSc.Vec().createSeq(global_n, comm=PETSc.COMM_SELF)
    scatter, _ = PETSc.Scatter.toAll(vec_dist)
    scatter.begin(vec_dist, vec_seq, addv=PETSc.InsertMode.INSERT_VALUES,
                  mode=PETSc.ScatterMode.FORWARD)
    scatter.end(vec_dist, vec_seq, addv=PETSc.InsertMode.INSERT_VALUES,
                mode=PETSc.ScatterMode.FORWARD)
    scatter.destroy()
    return vec_seq


# ---------------------------------------------------------------------------
# Public wrappers with scipy fallback
# ---------------------------------------------------------------------------

def distributed_eigs(
    mat: sp.csc_matrix,
    k: int,
    domain_keys: NDArray[I64],
    domain_partition,
    comm=None,
    which: str = "LM",
    local_coo=None,
    **scipy_kwargs,
) -> tuple[NDArray, list[BoxMeasure]]:
    """
    Compute *k* eigenvalues using SLEPc (MPI) or scipy ARPACK (serial).

    Tries SLEPc first when ``comm`` is active (size > 1) and slepc4py is
    installed; falls back to ``scipy.sparse.linalg.eigs`` otherwise.

    Parameters
    ----------
    mat              : scipy.sparse.csc_matrix — column-stochastic transfer mat
    k                : number of eigenvalues
    domain_keys      : sorted int64 keys (for BoxMeasure construction)
    domain_partition : BoxPartition
    comm             : mpi4py communicator or None
    which            : eigenvalue selection ('LM', 'SM', 'LR', 'SR')
    local_coo        : tuple (local_rows, local_cols, local_vals) or None
        When provided and SLEPc is used, PETSc is built directly from per-rank
        local COO edges via :func:`_build_petsc_mat_from_coo`, bypassing the
        global scipy matrix intermediate.  Ignored when falling back to scipy.
    **scipy_kwargs   : forwarded to scipy.sparse.linalg.eigs in fallback mode

    Returns
    -------
    eigenvalues   : ndarray, shape (k,)
    eigenmeasures : list of BoxMeasure, length k
    """
    if comm is None:
        from gaio.mpi.comm import get_comm
        comm = get_comm()

    use_slepc = (comm.Get_size() > 1) and slepc_available()

    if use_slepc:
        try:
            return _slepc_eigs(mat, k, which, domain_keys, domain_partition, comm,
                               local_coo=local_coo)
        except Exception as exc:
            import warnings
            warnings.warn(
                f"SLEPc eigensolve failed ({exc}); falling back to scipy ARPACK.",
                RuntimeWarning, stacklevel=2,
            )

    # scipy ARPACK fallback
    import scipy.sparse.linalg as spla
    n  = mat.shape[0]
    v0 = scipy_kwargs.pop("v0", np.ones(n, dtype=F64))
    vals, vecs = spla.eigs(mat, k=k, which=which, v0=v0, **scipy_kwargs)
    measures = [
        BoxMeasure(domain_partition, domain_keys.copy(),
                   vecs[:, i].real.astype(F64))
        for i in range(k)
    ]
    return vals, measures


def distributed_svds(
    mat: sp.csc_matrix,
    k: int,
    domain_keys: NDArray[I64],
    domain_partition,
    codomain_keys: NDArray[I64],
    codomain_partition,
    comm=None,
    local_coo=None,
    **scipy_kwargs,
) -> tuple[list[BoxMeasure], NDArray[F64], list[BoxMeasure]]:
    """
    Compute *k* singular triplets using SLEPc (MPI) or scipy ARPACK (serial).

    Parameters
    ----------
    mat                : scipy.sparse.csc_matrix
    k                  : number of singular triplets
    domain_keys        : int64 keys for right singular vectors
    domain_partition   : BoxPartition
    codomain_keys      : int64 keys for left singular vectors
    codomain_partition : BoxPartition
    comm               : mpi4py communicator or None
    local_coo          : tuple (local_rows, local_cols, local_vals) or None
        When provided and SLEPc is used, PETSc is built directly from per-rank
        local COO edges.  See :func:`distributed_eigs` for full description.

    Returns
    -------
    U     : list of BoxMeasure  (left singular vectors)
    sigma : ndarray, shape (k,)
    V     : list of BoxMeasure  (right singular vectors)
    """
    if comm is None:
        from gaio.mpi.comm import get_comm
        comm = get_comm()

    use_slepc = (comm.Get_size() > 1) and slepc_available()

    if use_slepc:
        try:
            return _slepc_svds(mat, k, domain_keys, domain_partition,
                               codomain_keys, codomain_partition, comm,
                               local_coo=local_coo)
        except Exception as exc:
            import warnings
            warnings.warn(
                f"SLEPc SVD failed ({exc}); falling back to scipy ARPACK.",
                RuntimeWarning, stacklevel=2,
            )

    # scipy ARPACK fallback
    import scipy.sparse.linalg as spla
    u, s, vt = spla.svds(mat, k=k, **scipy_kwargs)
    U = [
        BoxMeasure(codomain_partition, codomain_keys.copy(), u[:, i].astype(F64))
        for i in range(k)
    ]
    V = [
        BoxMeasure(domain_partition, domain_keys.copy(), vt[i, :].astype(F64))
        for i in range(k)
    ]
    return U, s, V
