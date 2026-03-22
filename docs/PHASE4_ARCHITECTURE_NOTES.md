# Phase 4 Architecture Notes: Multi-GPU Scaling via MPI

## Overview

Phase 4 delivers true distributed computation across the full GAIO pipeline.
Every rank performs only a fraction of the work; all ranks arrive at the
same result.  Three components are distributed:

| Component | Module | Bottleneck addressed |
|-----------|--------|----------------------|
| Attractor / manifold subdivision | `gaio/mpi/distributed_attractor.py` | Per-rank GPU VRAM (test-point array) |
| Transfer operator construction | `gaio/transfer/operator.py` + `gaio/mpi/gather.py` | Per-rank GPU VRAM (same bottleneck) |
| Eigensolve / SVD | `gaio/mpi/distributed_eigs.py` | Per-rank CPU/RAM for Krylov basis |

A fourth utility — GPUDirect RDMA (`gaio/mpi/rdma.py`) — eliminates the
device-to-host staging copy from the Allgatherv step when CUDA-aware MPI
is available.

Phase 5 adds dynamic load balancing via weighted prefix-sum decomposition.
Everything in Phase 4 uses static Morton decomposition; Phase 5 adds the
adaptive layer on top — see [PHASE5_ARCHITECTURE_NOTES.md](PHASE5_ARCHITECTURE_NOTES.md).

---

## 1. Static Morton Z-Order Decomposition

### Why Morton order?

Flat int64 keys in `BoxPartition` are assigned in C row-major order.  A
naive slice gives each rank a thin slab — spatially non-compact and
poorly aligned with the fractal attractor geometry.

Morton (Z-order) codes interleave the bits of each dimension's cell index,
mapping the N-dimensional grid onto a 1-D curve that visits nearby cells
in nearby positions.  Splitting by contiguous Morton ranges gives each rank
a spatially compact subdomain, reducing boundary-cell communication from
O(K) to O(K^{(d-1)/d}).

**Bit-interleaving (2-D example)**

```
Bit positions:   ... 5  4  3  2  1  0
x-index bits:    ... x2    x1    x0
y-index bits:    ...    y2    y1    y0
Morton code:     ... y2 x2 y1 x1 y0 x0
```

For d=3 with 21 bits per dimension: 63-bit Morton codes, safe for grids up
to 2,097,151 cells per axis.

### Row-range ownership

The same `array_split` semantics are used throughout Phase 4 for any 1-D
array of size K across P ranks:

```python
base, extra = divmod(K, P)
start[r] = r * base + min(r, extra)
end[r]   = start[r] + base + (1 if r < extra else 0)
```

This ensures `sum(end[r] - start[r]) == K` exactly, with the first
`K % P` ranks getting one extra element.

---

## 2. Distributed Relative Attractor

### Memory model

The full BoxSet `S` (key array only) is replicated on every rank.
At K=1M cells the key array is only 8 MB — negligible.  The memory
saving comes from the **test-point array**, which each rank generates only
for its local shard:

| Mode | Test-point array size (1M cells, M=27, d=3) |
|------|---------------------------------------------|
| Serial / single GPU | 648 MB |
| MPI, P=4 ranks | 162 MB per rank |
| MPI, P=8 ranks | 81 MB per rank |

### Algorithm (one subdivision step)

```
All ranks:    S = S.subdivide(dim)          ← deterministic, O(K), no comm

Rank r:       local_keys = morton_slice(S._keys, r, P)
              local_S    = BoxSet(partition, local_keys)
              local_img  = F(local_S)        ← GPU kernel, K/P × M test pts

All ranks:    Allgatherv( local_img._keys )  ← O(K_image × 8 B) traffic
              global_img = np.unique(all_keys)
              S          = S & global_img    ← sorted intersection, O(K)
```

### Convergence guarantee

Because test points are disjoint between ranks (each source cell is owned
by exactly one rank), the union of all per-rank image sets equals the image
of the full S under F.  No cell is incorrectly included or excluded —
the outer-approximation guarantee is fully preserved.

### Communication volume per step

| Quantity | Volume |
|----------|--------|
| Allgatherv payload | `P × K_image × 8` bytes (K_image ≤ K) |
| np.unique merge | O(P × K_image) |
| Intersection S & img | O(K log K) |

For K=100K cells, P=4, K_image≈80K: `4 × 80K × 8 = 2.6 MB` per step —
orders of magnitude less than the test-point computation.

---

## 3. Distributed Transfer Operator Construction

The `TransferOperator` construction follows the same pattern as the
attractor, extended to also track source-cell column indices for the COO
matrix.

### Pipeline (per rank)

```
Stage 1: Generate local_K × M test points   (NumPy broadcast)
Stage 2: F._apply_map(test_pts)              (GPU kernel, local_K × M pts)
Stage 3: Key lookup + local COO assembly     (NumPy / CUDA lookup)
Stage 4: Allgatherv rows, cols, vals         (GPUDirect RDMA if available)
All ranks: assemble coo_matrix → csc_matrix → column-normalise
```

Column indices in the local COO always reference the **global** domain
index (position in the full `domain._keys` array), so the assembled matrix
is identical on every rank.

### Dynamic buffer allocation

```
┌───────────────────────────────────────────────────────────────┐
│  gather_sizes: one Allgather of int64 scalar per rank         │
│  → counts[r] = local COO entries for rank r                   │
│  → total = sum(counts)                                         │
├───────────────────────────────────────────────────────────────┤
│  np.empty(total, int64/float64) — exact-sized receive buffers │
│  3 × Allgatherv (rows, cols, vals)                             │
└───────────────────────────────────────────────────────────────┘
```

After construction, `T.mpi_stats` exposes `per_rank_nnz` and
`total_nnz_raw` for load-imbalance diagnostics.

---

## 4. Distributed Eigensolve / SVD via SLEPc/PETSc

### Why SLEPc?

At K > 500K cells the transfer matrix is ~500K × 500K sparse.
ARPACK (scipy, single-process) needs O(K × k) RAM for the Krylov basis —
at K=500K, k=6, that is ~2.4 GB on a single core.

SLEPc distributes both the matrix (each rank owns a row range) and the
Krylov basis vectors (same row range), scaling memory as 1/P.

| Algorithm | SLEPc type | Problem |
|-----------|-----------|---------|
| Eigenvalues | `EPS.KRYLOVSCHUR` | Non-Hermitian (NHEP) |
| Singular values | `SVD.LANCZOS` | Rectangular |

Krylov–Schur is numerically more stable than vanilla Arnoldi for
non-Hermitian problems — important because our transfer operator is
column-stochastic but not symmetric.

### Matrix construction

We receive the assembled scipy CSC matrix (already on every rank after
the Allgatherv step) and convert it to a distributed PETSc MPIAIJ matrix.
Rank r inserts only its local row range; PETSc assembly handles the
off-process communication needed to finalize the distributed structure.

```python
# Each rank inserts rows [row_start, row_end)
local_slice = mat_csr[row_start:row_end, :]   # (local_m, global_n) CSR
A.setValues(global_rows, global_cols, values,
            addv=PETSc.InsertMode.INSERT_VALUES)
A.assemblyBegin(...)
A.assemblyEnd(...)
```

### Eigenvector distribution

SLEPc returns distributed Vecs (each rank holds a row range).
`_petsc_scatter_to_all` uses `PETSc.Scatter.toAll` to replicate the
full eigenvector on every rank, so `BoxMeasure` construction works
identically to the serial path.

### Graceful fallback

When `slepc4py` is not installed (e.g., WSL without SLEPc), or if the
SLEPc solve fails, `distributed_eigs` / `distributed_svds` automatically
fall back to `scipy.sparse.linalg.eigs` / `svds` on each rank
independently.  A `RuntimeWarning` is issued but computation continues.

---

## 5. GPUDirect RDMA

### What it eliminates

Without RDMA, the Allgatherv in Stage 4 of the transfer operator pipeline
requires:

```
GPU VRAM → host RAM  (device-to-host copy, PCIe 4.0 ≈ 32 GB/s)
host RAM → NIC       (DMA to network interface)
```

With CUDA-aware MPI (Open MPI ≥ 4.0 + UCX + GPUDirect RDMA driver):

```
GPU VRAM → NIC       (single DMA, no CPU staging)
```

For 100K cells, M=27, d=3: COO values = `100K × 27 × 8 = 21 MB`.
At PCIe 4.0 x16 (32 GB/s), the eliminated copy costs ~0.7 ms per rank per
frame — ~17 ms total for a 24-frame animation, ~42 ms for 60-frame.

### Detection

Three-level probe in `gaio/mpi/rdma.py`:

1. `OMPI_MCA_opal_cuda_support=1` env var (Open MPI)
2. `MV2_USE_CUDA=1` env var (MVAPICH2)
3. Runtime: Allgather a 4-byte CUDA device array; verify result

The probe is cached per communicator after the first call.

### Current activation status

The RDMA path in `rdma_allgatherv` is **architecturally complete**: it
detects device arrays and tries the device-direct Allgatherv.  Full
activation requires Stages 2–3 of `_build_transitions` to produce device
arrays (keeping the mapped points and COO in GPU VRAM) rather than copying
them to host first.  This pipeline change is a candidate for a future phase.

---

## 6. Phase 5: Adaptive Load Balancing

Phase 4 uses a static Morton decomposition — the key-to-rank mapping does
not change during computation.  For highly non-uniform attractors (long
filaments, Cantor-set dust), one rank may receive 5–10× more COO work than
another.  Fast ranks sit idle at the Allgatherv barrier.

Phase 5 is **implemented** and documented separately:

→ **[PHASE5_ARCHITECTURE_NOTES.md](PHASE5_ARCHITECTURE_NOTES.md)**

Quick reference:

```python
# Nonautonomous usage — Phase 5 kicks in from frame 1 onwards
weights = None
for t in range(n_frames):
    A = relative_attractor(F_t, S, steps=steps, comm=comm)
    T = TransferOperator(F_t, A, A, comm=comm, partition_weights=weights)
    weights = T.partition_weights   # reuse next frame
```

---

## 7. How to Test

### WSL / single machine (CPU, no GPU)

```bash
# Install mpi4py
conda install -c conda-forge mpi4py openmpi

# Serial path (exercises _SerialComm stub, auto-detected)
python tests/test_mpi_scaling.py

# 2-rank MPI
mpiexec -n 2 python tests/test_mpi_scaling.py

# 4-rank MPI (CPU)
mpiexec -n 4 python tests/test_mpi_scaling.py --steps 6 --grid-res 2
```

Expected output:
```
  Per-rank COO contribution (Morton decomposition)
    Rank    COO entries    % of total
  ----------------------------------------
       0          5,234         24.8%
       1          5,089         24.1%
       2          5,411         25.7%
       3          5,361         25.4%
  ----------------------------------------
   Total         21,095        100.0%

  Load imbalance (max/min): 1.063  (good)
  [PASS] All ranks produced identical TransferOperator.
```

### SLEPc availability check

```bash
python -c "from gaio.mpi import slepc_available; print('SLEPc:', slepc_available())"
```

Install SLEPc (optional, needed for K > 500K):
```bash
conda install -c conda-forge slepc4py petsc4py
```

### Multi-GPU cloud instance (e.g., AWS p3.8xlarge = 4× V100)

```bash
conda env create -f environment.yml && conda activate gaio

# Verify CUDA-aware MPI
python -c "from gaio.mpi import check_cuda_aware_mpi; check_cuda_aware_mpi()"

# Verify RDMA capability
python -c "from gaio.mpi import is_rdma_capable, get_comm; print('RDMA:', is_rdma_capable(get_comm()))"

# Bind one rank per GPU; full pipeline
mpiexec -n 4 --bind-to socket python tests/test_mpi_scaling.py --gpu --steps 12

# Full benchmark
mpiexec -n 4 python benchmarks/benchmark_backends.py --mpi-gpu --steps 14
```

### Verify distributed attractor equals serial

```python
# Quick parity check (WSL, no GPU needed)
mpiexec -n 4 python - <<'EOF'
import numpy as np
from gaio import Box, BoxPartition, BoxSet, SampledBoxMap, rk4_flow_map
from gaio import relative_attractor
from gaio.mpi import get_comm, rank

def henon(x): return np.array([1 - 1.4*x[0]**2 + x[1], 0.3*x[0]])
domain = Box([0.,0.], [1.5, 1.5])
P = BoxPartition(domain, [2,2])
unit_pts = np.array([[-0.5,-0.5],[-0.5,0.5],[0.5,-0.5],[0.5,0.5]])
F = SampledBoxMap(henon, domain, unit_pts)

comm = get_comm()
A_dist   = relative_attractor(F, BoxSet.full(P), steps=10, comm=comm)
A_serial = relative_attractor(F, BoxSet.full(P), steps=10, comm=False)

if rank() == 0:
    assert np.array_equal(A_dist._keys, A_serial._keys), \
        f"Mismatch: {len(A_dist)} vs {len(A_serial)} cells"
    print(f"[PASS] Distributed attractor matches serial: {len(A_dist)} cells")
EOF
```

### Common failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: mpi4py` | Not installed | `pip install mpi4py` |
| `ModuleNotFoundError: petsc4py` | SLEPc path used without install | Install via `conda install slepc4py petsc4py`, or let fallback handle it |
| Hang at Allgatherv | One rank crashed earlier | Check stderr for exception |
| `Inconsistent TransferOperator` | Attractor not synced before T_op build | Compute attractor before `TransferOperator(comm=comm)` |
| RDMA returns wrong data | MPI lacks CUDA support | Probe fails gracefully, CPU staging used |
| SLEPc `EPSSolve` diverged | Tight tolerance + non-normal matrix | Relax `tol`, increase `max_it` via `setTolerances` |
