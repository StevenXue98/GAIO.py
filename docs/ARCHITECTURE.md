# GAIO.py — Architecture & Developer Guide

A self-contained reference explaining *why* each design decision was made, *what* mathematical object it corresponds to in GAIO.jl, and *how* the Python implementation encodes it.

---

## Table of Contents

1. [Mathematical Background](#1-mathematical-background)
2. [Directory Structure](#2-directory-structure)
3. [Phase Roadmap](#3-phase-roadmap)
4. [Core Data Structures (Phase 1)](#4-core-data-structures-phase-1)
   - 4.1 [Box](#41-box)
   - 4.2 [BoxPartition](#42-boxpartition)
   - 4.3 [BoxSet and BoxMeasure](#43-boxset-and-boxmeasure)
5. [BoxMap Types (Phase 2)](#5-boxmap-types-phase-2)
6. [Algorithms (Phase 2)](#6-algorithms-phase-2)
   - 6.1 [Relative Attractor](#61-relative-attractor)
   - 6.2 [Unstable and Stable Manifolds](#62-unstable-and-stable-manifolds)
   - 6.3 [Transfer Operator and Spectral Analysis](#63-transfer-operator-and-spectral-analysis)
   - 6.4 [Almost-Invariant Sets and Morse Decomposition](#64-almost-invariant-sets-and-morse-decomposition)
7. [GPU Acceleration (Phase 3)](#7-gpu-acceleration-phase-3)
   - 7.1 [Backends](#71-backends)
   - 7.2 [AcceleratedBoxMap](#72-acceleratedboxmap)
   - 7.3 [Writing a CUDA Flow Map](#73-writing-a-cuda-flow-map)
   - 7.4 [Full GPU Pipeline](#74-full-gpu-pipeline)
   - 7.5 [GPU Systems Engineering](#75-gpu-systems-engineering)
8. [Distributed Computing (Phase 4)](#8-distributed-computing-phase-4)
   - 8.1 [MPI Domain Decomposition](#81-mpi-domain-decomposition)
   - 8.2 [MPI Systems Engineering](#82-mpi-systems-engineering)
9. [Load Balancing (Phase 5)](#9-load-balancing-phase-5)
10. [Key Design Decisions](#10-key-design-decisions)
11. [Correspondence with GAIO.jl](#11-correspondence-with-gaiojl)
12. [API Reference](#12-api-reference)

---

## 1. Mathematical Background

### 1.1 Boxes

A **box** in ℝⁿ is:

```
B(c, r) = { x ∈ ℝⁿ : cᵢ - rᵢ ≤ xᵢ < cᵢ + rᵢ  for all i }
```

Defined by its **centre** `c ∈ ℝⁿ` and **radius** `r ∈ ℝⁿ`. The half-open convention ensures boxes tile without overlap: exactly one cell claims any interior point.

### 1.2 Box Partition

A **box partition** of `B(c₀, r₀)` into a `d₁ × … × dₙ` grid produces `N = ∏dₖ` cells. Cell `(i₁, …, iₙ)` has:

```
cell_radius[k] = r₀[k] / dₖ
center[k]      = lo[k] + 2 * cell_radius[k] * (iₖ + 0.5)
```

where `lo = c₀ - r₀`.

### 1.3 Transfer Operator (Ulam's Method)

For a map `f : ℝⁿ → ℝⁿ`, the **transfer matrix** `T` has entry:

```
Tᵢⱼ ≈  |{ test points in Bⱼ that map into Bᵢ }|
        ─────────────────────────────────────────
                 total test points in Bⱼ
```

`T` is column-stochastic (each column sums to 1). Its leading left-eigenvector (eigenvalue = 1) approximates the **natural invariant measure** of `f`. The second eigenvector identifies **almost-invariant sets** — lobes where trajectories are trapped.

### 1.4 Relative Attractor

The **relative attractor** `A(f, Q)` is the largest set `A ⊆ Q` with `f(A) ⊆ A`. Computed by the **subdivision algorithm**:

```
Q₀ = Q
Qₖ₊₁ = { B ∈ refine(Qₖ) : F(B) ∩ Qₖ ≠ ∅ }
A = lim_{k→∞} Qₖ
```

Each step doubles the grid resolution in every dimension. After `steps=16`, the cell width is `initial_width / 2^16`.

---

## 2. Directory Structure

```
GAIO.py/
│
├── pyproject.toml              ← pip install -e .
├── environment.yml             ← conda env (name: gaio)
├── README.md                   ← benchmarks + setup only
│
├── gaio/                       ← installable Python package
│   ├── __init__.py             ← re-exports all public symbols
│   │
│   ├── core/                   ← Phase 1: data structures
│   │   ├── box.py              ← Box: typed hyperrectangle
│   │   ├── partition.py        ← BoxPartition: uniform grid, flat int64 key scheme
│   │   ├── boxset.py           ← BoxSet: sparse sorted set of active cells
│   │   └── boxmeasure.py       ← BoxMeasure: weighted BoxSet (discrete measure)
│   │
│   ├── maps/                   ← Phase 2: map discretisation
│   │   ├── base.py             ← abstract BoxMap protocol
│   │   ├── grid_map.py         ← GridMap: deterministic test-point grid per cell
│   │   ├── montecarlo_map.py   ← MonteCarloMap: random test points
│   │   ├── rk4.py              ← rk4_flow_map: RK4 integrator closure from a vector field
│   │   └── nonautonomous.py    ← non-autonomous / time-dependent maps
│   │
│   ├── algorithms/             ← Phase 2: set-oriented algorithms
│   │   ├── attractor.py        ← relative_attractor, subdivision loop
│   │   ├── manifolds.py        ← unstable_set, alpha_limit_set, preimage
│   │   ├── invariant_sets.py   ← maximal_invariant_set, recurrent_set
│   │   ├── morse.py            ← morse_sets, morse_tiles
│   │   └── ftle.py             ← finite-time Lyapunov exponent fields
│   │
│   ├── transfer/               ← Phase 2: Perron-Frobenius operator
│   │   └── operator.py         ← TransferOperator: sparse matrix, eigs, svds
│   │
│   ├── graph/                  ← Phase 2: graph structure of the transfer operator
│   │   └── boxgraph.py         ← BoxGraph, SCC decomposition
│   │
│   ├── cuda/                   ← Phase 3: GPU acceleration
│   │   ├── accelerated_map.py  ← AcceleratedBoxMap: drop-in GPU/CPU-parallel BoxMap
│   │   ├── gpu_backend.py      ← @cuda.jit kernels: batched test-point evaluation
│   │   ├── cpu_backend.py      ← @njit(parallel=True) CPU backend
│   │   ├── backends.py         ← backend selection and device detection
│   │   └── rk4_cuda.py         ← make_cuda_rk4_flow_map: CUDA RK4 factory
│   │
│   ├── mpi/                    ← Phase 4+5: distributed multi-GPU
│   │   ├── decompose.py        ← Morton-order domain decomposition
│   │   ├── distributed_attractor.py  ← MPI relative_attractor
│   │   ├── distributed_eigs.py ← SLEPc/PETSc distributed eigensolver
│   │   ├── gather.py           ← Allgatherv helpers
│   │   ├── load_balance.py     ← Phase 5: weighted Morton repartitioning
│   │   ├── rdma.py             ← GPUDirect RDMA detection and staging
│   │   └── comm.py             ← _SerialComm stub (zero-overhead single-process)
│   │
│   ├── viz/                    ← visualisation helpers
│   │   ├── plot2d.py           ← matplotlib 2D BoxSet plots
│   │   └── plot3d.py           ← PyVista 3D BoxSet plots
│   │
│   └── examples/               ← runnable examples (python -m gaio.examples.*)
│       ├── attractor.py        ← Hénon attractor
│       ├── four_wing.py        ← 3D Four-Wing (GPU showcase)
│       ├── invariant_measure_1d.py
│       ├── invariant_measure_2d.py
│       ├── almost_invariant_sets.py
│       ├── unstable_manifold.py
│       ├── transfer_operator.py
│       └── double_gyre_*.py    ← double-gyre coherent sets and FTLE
│
├── benchmarks/
│   ├── benchmark_phase3.py     ← single-GPU vs Python/Numba
│   ├── benchmark_vs_julia.py   ← GAIO.py vs GAIO.jl head-to-head
│   ├── benchmark_phase4.py     ← multi-GPU MPI scaling
│   ├── benchmark_phase5.py     ← load-balancing (Lozi attractor)
│   ├── benchmark_imbalance.py  ← COO imbalance diagnostics
│   └── gaio_julia_benchmark.jl ← Julia-side benchmark runner
│
└── docs/
    ├── ARCHITECTURE.md         ← this file
    ├── VAST_AI_SETUP.md        ← vast.ai cloud deployment
    ├── LAMBDA_CLOUD_SETUP.md   ← Lambda Cloud deployment
    └── PHASE*_ARCHITECTURE_NOTES.md  ← per-phase design notes
```

---

## 3. Phase Roadmap

| Phase | Status | Deliverable |
|-------|--------|-------------|
| **1** | ✅ | `Box`, `BoxPartition`, `BoxSet`, `BoxMeasure` — typed NumPy foundation |
| **2** | ✅ | `BoxMap` variants, all set-oriented algorithms, `TransferOperator` |
| **3** | ✅ | Numba CUDA kernels; `AcceleratedBoxMap`; single-GPU speedup |
| **4** | ✅ | MPI domain decomposition; multi-GPU `TransferOperator`; SLEPc eigensolver |
| **5** | ✅ | Weighted Morton repartitioning; load-balanced T_op on non-uniform attractors |

Each phase is a strict extension of the previous: phases 3–5 are drop-in accelerators, and the CPU algorithms from phase 2 are preserved as fallbacks.

---

## 4. Core Data Structures (Phase 1)

### 4.1 Box

**File:** `gaio/core/box.py`

A single axis-aligned hyperrectangle `B(c, r) = [c-r, c+r)`.

```python
class Box:
    center : np.ndarray  # shape (n,), dtype float64, C-contiguous
    radius : np.ndarray  # shape (n,), dtype float64, C-contiguous
```

`__slots__` prevents `__dict__` allocation — important when algorithms generate millions of temporary `Box` objects during subdivision.

`contains_point` tests `lo ≤ p < hi` (half-open). `rescale(u)` maps `[-1,1]ⁿ → box`, used by `BoxMap` test-point generators.

| Method | Returns | Notes |
|--------|---------|-------|
| `contains_point(p)` | `bool` | half-open `[lo, hi)` |
| `intersects(other)` | `bool` | open-interior check |
| `intersection(other)` | `Box` | raises if empty |
| `rescale(u)` | `ndarray` | maps `[-1,1]ⁿ → box` |
| `subdivide(dim)` | `(Box, Box)` | bisect along one axis |
| `subdivide_all()` | `list[Box]` | `2ⁿ` children |

### 4.2 BoxPartition

**File:** `gaio/core/partition.py`

Uniform Cartesian grid dividing a `Box` domain into `prod(dims)` equal cells. Equivalent to `BoxGrid{N,T,I}` in GAIO.jl.

#### Flat key scheme

Every cell is identified by a single `int64` from flattening its multi-index in C (row-major) order:

```python
flat_key = np.ravel_multi_index((i₀, i₁, …, iₙ₋₁), dims)
```

**Why flat keys instead of tuple keys?** Numba `@cuda.jit` kernels and MPI send/recv buffers require contiguous C-typed arrays. An `int64` array passes to both without boxing or serialisation.

| Method | Returns | Notes |
|--------|---------|-------|
| `key_to_box(k)` | `Box` | geometry of cell `k` |
| `point_to_key(p)` | `int \| None` | `None` if outside domain |
| `point_to_key_batch(pts)` | `ndarray (N,)` | `-1` for out-of-domain |
| `keys_in_box(query)` | `ndarray` | range query, O(hits) not O(size) |
| `subdivide_all()` | `BoxPartition` | 2× finer in every dimension |

#### Cell geometry

```
lo             = domain.center - domain.radius
cell_width[k]  = 2 * domain.radius[k] / dims[k]
center of cell m = lo + cell_width * (m + 0.5)
```

### 4.3 BoxSet and BoxMeasure

**File:** `gaio/core/boxset.py`, `gaio/core/boxmeasure.py`

```python
class BoxSet:
    partition : BoxPartition
    _keys     : np.ndarray   # shape (N,), dtype int64, sorted, unique, C-contiguous
```

`_keys` is **always sorted and unique** after construction. This enables:
- **O(log N) membership** via `np.searchsorted`
- **O(N log N) set algebra** via `np.union1d`, `np.intersect1d`, etc.
- Direct CUDA kernel and MPI buffer passing without conversion

All set operations return new `BoxSet` objects; `_keys` is never mutated. `centers()` returns an `(N, n)` float64 array of all cell centres in one NumPy call.

`BoxMeasure` pairs a `BoxSet` with a float64 weight vector, representing a discrete measure (invariant density, eigenmeasure).

Set algebra operators: `|` (union), `&` (intersection), `-` (difference), `^` (symmetric difference), `<=` / `>=` (subset/superset).

Convenience constructors: `BoxSet.full(p)`, `BoxSet.empty(p)`, `BoxSet.cover(p, points)`, `BoxSet.from_box(p, query)`.

---

## 5. BoxMap Types (Phase 2)

**Files:** `gaio/maps/`

A `BoxMap` discretises a continuous map `f: ℝⁿ → ℝⁿ` onto a partition. Given a source `BoxSet`, it returns the **outer-approximation image**: every cell that *could* be hit by any point in any source cell.

| Class | Test-point strategy | When to use |
|-------|---------------------|-------------|
| `SampledBoxMap` | User-supplied `(M, n)` coordinate array | Full control over placement |
| `GridMap` | Uniform Cartesian grid, `n_points` per dimension | Default for deterministic maps |
| `MonteCarloMap` | Random uniform samples | Stochastic maps or high dimensions |

`rk4_flow_map(vfield, step_size, steps)` returns a closure implementing fixed-step RK4 from a vector field — the standard way to build a box map from an ODE.

```python
from gaio import GridMap, rk4_flow_map
import numpy as np

def four_wing_v(x):
    return np.array([0.2*x[0] + x[1]*x[2],
                     -0.4*x[1] - 0.01*x[0] - x[2]*x[1],
                     -x[2] - x[0]*x[1]])

f = rk4_flow_map(four_wing_v, step_size=0.01, steps=20)
F = GridMap(f, domain, n_points=4)   # 4³ = 64 test points per cell in 3D
```

---

## 6. Algorithms (Phase 2)

### 6.1 Relative Attractor

**File:** `gaio/algorithms/attractor.py`

```python
from gaio import relative_attractor, BoxSet

S = BoxSet.full(P)
A = relative_attractor(F, S, steps=16)
```

Each step: subdivide all active cells → keep only those whose image intersects the current set. Resolution doubles each step; after `steps=N` the cell width is `initial_width / 2^N`.

### 6.2 Unstable and Stable Manifolds

**File:** `gaio/algorithms/manifolds.py`

```python
from gaio import unstable_set, alpha_limit_set, preimage

W_u = unstable_set(F, seed)          # grows forward from seed
W_s = alpha_limit_set(F, seed)       # grows backward from seed
pre = preimage(F, B, C)              # pre-image of C restricted to B
```

### 6.3 Transfer Operator and Spectral Analysis

**File:** `gaio/transfer/operator.py`

The `TransferOperator` builds a column-stochastic sparse matrix `T` where `T[i,j]` = fraction of test points from cell `j` landing in cell `i`.

```python
from gaio import TransferOperator

T = TransferOperator(F, domain, codomain)

# Spectral analysis
eigenvalues, eigenmeasures = T.eigs(k=3)    # leading k eigenpairs (ARPACK)
U, s, Vt = T.svds(k=3)                     # leading k singular triplets

# Push-forward and pull-back
mu_new  = T.push_forward(mu)   # T · μ
mu_pull = T.pull_back(mu)      # Tᵀ · μ
mu_new  = T @ mu               # same as push_forward
```

The underlying matrix `T.mat` is a `scipy.sparse.csc_matrix`. Sparse eigensolvers (ARPACK) run on CPU — a 10,000–100,000 cell matrix is solved in seconds; GPU sparse eigensolvers offer no practical speedup at this scale.

### 6.4 Almost-Invariant Sets and Morse Decomposition

**File:** `gaio/algorithms/morse.py`

**Almost-invariant sets** are identified by the second eigenvector of the transfer operator (eigenvalue λ₂ < 1 but close to 1): its sign partitions the attractor into two lobes.

```python
eigenvalues, eigenmeasures = T.eigs(k=2)
mu2 = eigenmeasures[1]

positive_cells = BoxSet(A.partition, mu2._keys[mu2._weights > 0])
negative_cells = BoxSet(A.partition, mu2._keys[mu2._weights < 0])
```

A full **Morse decomposition** — coarse-to-fine invariant components connected by gradient-like flow — is computed from the transfer operator graph:

```python
from gaio import morse_sets, morse_tiles, recurrent_set

R      = recurrent_set(T)
M_sets = morse_sets(T)
M_tiles = morse_tiles(T)
```

---

## 7. GPU Acceleration (Phase 3)

### 7.1 Backends

**Files:** `gaio/cuda/backends.py`, `gaio/cuda/cpu_backend.py`, `gaio/cuda/gpu_backend.py`

| Backend | Hardware | Library | Notes |
|---------|----------|---------|-------|
| `"python"` | CPU (single-core) | NumPy | Baseline |
| `"cpu"` | CPU (multi-core) | Numba `@njit(parallel=True)` | Parallelises via `prange`, releases GIL |
| `"gpu"` | NVIDIA GPU | Numba `@cuda.jit` | Register-file RK4, float64 end-to-end |

The backend is selected at construction time and transparent to all downstream algorithms. `cuda_available()` / `numba_available()` return runtime device availability.

### 7.2 AcceleratedBoxMap

**File:** `gaio/cuda/accelerated_map.py`

Drop-in replacement for `SampledBoxMap`. Takes an additional `f_device` argument (a `@cuda.jit(device=True)` function) and a `backend` string.

```python
from numba import cuda
from gaio.cuda import AcceleratedBoxMap

@cuda.jit(device=True)
def four_wing_device(x, out):
    out[0] = 0.2*x[0] + x[1]*x[2]
    out[1] = -0.4*x[1] - 0.01*x[0] - x[2]*x[1]
    out[2] = -x[2] - x[0]*x[1]

F = AcceleratedBoxMap(f_cpu, domain, unit_pts,
                      f_device=four_wing_device, backend="gpu")

# Usage identical to SampledBoxMap
A = relative_attractor(F, BoxSet.full(P), steps=16)
T = TransferOperator(F, A, A)
```

**GPU dedup path**: `map_boxes` on GPU returns a Numba device array directly (no `copy_to_host`). The caller wraps it with `cp.asarray()` for zero-copy CuPy access, then runs `cp.unique()` entirely on-device. The only PCIe transfer is the final compact unique-key array (~1–5% of the raw hit buffer).

### 7.3 Writing a CUDA Flow Map

**File:** `gaio/cuda/rk4_cuda.py`

`make_cuda_rk4_flow_map` wraps a user-defined vector field device function in a fixed-step RK4 integrator, matching the CPU `rk4_flow_map` pattern.

```python
from numba import cuda
from gaio.cuda import make_cuda_rk4_flow_map

@cuda.jit(device=True)
def four_wing_v_device(x, out):
    out[0] = 0.2*x[0] + x[1]*x[2]
    out[1] = -0.4*x[1] - 0.01*x[0] - x[2]*x[1]
    out[2] = -x[2] - x[0]*x[1]

f_gpu = make_cuda_rk4_flow_map(four_wing_v_device, ndim=3,
                                step_size=0.01, steps=20)
```

The `f_device` signature is `f_device(x, out) -> None`: reads from `x`, writes result in-place to `out`. No heap allocation is allowed in CUDA device functions.

Both `rk4_flow_map` and `make_cuda_rk4_flow_map` implement the same Butcher tableau with identical step sizes — CPU and GPU results are numerically identical.

### 7.4 Full GPU Pipeline

```
BoxSet.full(P)
    │
    ▼
relative_attractor(F, S, steps=N)          ← GPU: CUDA kernel over all K×M test points
    │  returns BoxSet (attractor A)
    ▼
TransferOperator(F, A, A)                  ← GPU: same CUDA kernel + on-device CuPy dedup
    │  returns sparse matrix T (scipy.sparse on CPU)
    ▼
T.eigs(k) / T.svds(k)                     ← CPU: ARPACK/BLAS (not the bottleneck)
    │  returns eigenvalues, eigenmeasures
    ▼
visualise / analyse
```

### 7.5 GPU Systems Engineering

**Register-file RK4.** The kernel generated by `make_cuda_rk4_flow_map` uses `@cuda.jit` device functions operating entirely on thread-local arrays. The RK4 state vector and `k1–k4` slopes live in GPU registers, never touching shared or global memory until the final position is written. For a 3D system: 3 (state) + 12 (k1–k4) = 15 floats × 4 bytes = 60 bytes per thread — within the 256-byte register budget of compute capability 7.0+. This yields near-peak arithmetic throughput with no memory-bandwidth bottleneck.

**float64 throughout.** `float32` accumulates significant rounding error in iterated box containment tests at high spatial resolution. Phase 3 kernels operate in float64 throughout. (GAIO.jl's GPU path propagates float32 end-to-end through Julia's type system — this accounts for the ~0.16 s gap between GAIO.py GPU and GAIO.jl GPU in benchmarks at small scales.)

**On-device dedup.** The raw GPU hit buffer is K×M int64 entries (~36 MB at 562k cells). Before the Phase 3 GPU optimisations, this was copied to CPU via PCIe for `np.unique`. After: the caller uses `cp.asarray()` (zero-copy CuPy wrap), runs `cp.unique()` on-device, and only the compact unique array crosses PCIe. Reduces PCIe traffic ~64× per step.

---

## 8. Distributed Computing (Phase 4)

### 8.1 MPI Domain Decomposition

**Files:** `gaio/mpi/`

Each MPI rank owns a disjoint slice of the domain's `_keys` array, computes its local test-point batch independently, and contributes COO entries to the global transfer matrix. To run:

```bash
mpiexec -n 8 python my_gaio_script.py
```

For the eigensolver path (`T.eigs`, `T.svds`), GAIO.py uses **SLEPc/PETSc** via petsc4py/slepc4py. Each rank inserts only its local COO entries into the PETSc `MPIAIJ` matrix using `ADD_VALUES`. PETSc handles off-process row communication during `assemblyBegin/End`. This eliminates the Allgatherv-to-every-rank bottleneck: instead of each rank building the full 3.3M-entry global matrix and discarding 7/8 of it, each rank only ever holds its 1/8 slice.

Column normalisation is done in PETSc: `A.multTranspose(ones, col_sums)` computes column sums; `A.diagonalScale(r=col_sums_inv)` normalises in-place — no scipy matrix needed in the SLEPc path.

### 8.2 MPI Systems Engineering

**Morton-order spatial decomposition.** Domain cells are sorted by Morton (Z-order) code, which interleaves the bits of cell multi-indices. This groups spatially adjacent cells on the same rank, minimising the fraction of test-point outputs that cross MPI rank boundaries from O(K) to O(K^{(d-1)/d}). For a 3D attractor with 100K cells and 4 ranks, this reduces cross-rank traffic by ~15×.

**Zero-copy serial fallback.** When running single-process (no `mpiexec`), the MPI code path is exercised through a `_SerialComm` stub (`gaio/mpi/comm.py`) that returns input arrays unchanged with no copies and no mpi4py import. The entire MPI machinery adds zero overhead to single-GPU usage.

**GPUDirect RDMA.** On clusters with CUDA-aware MPI (OpenMPI ≥ 4.0 + UCX + GPUDirect RDMA), `Allgatherv` can transfer directly from GPU VRAM to the network interface, bypassing the device-to-host copy. Check availability:

```python
from gaio.mpi import check_cuda_aware_mpi; check_cuda_aware_mpi()
```

---

## 9. Load Balancing (Phase 5)

**File:** `gaio/mpi/load_balance.py`

Phase 4 uses a uniform Morton split: each rank gets K/P consecutive cells in Morton order. For non-uniform attractors (e.g., the Lozi map's thin angular filament), some ranks receive mostly "cold" fringe cells with few hits and very little COO work, while other ranks hold the dense core. The resulting imbalance in T_op wall time can reach 30× in extreme cases.

**Weighted Morton repartitioning** counts the hit density per cell after the attractor computation and sorts cells so each rank receives equal total COO work, not equal cell count. This reduces imbalance from ~30× to ~1.0× on the extreme Lozi scenario.

Run `mpiexec -n N python benchmarks/benchmark_phase5.py` to compare static vs weighted decomposition on two Lozi scenarios (moderate ~4.5× and extreme ~30× imbalance).

---

## 10. Key Design Decisions

**D1 — Flat `int64` keys.** Julia uses `CartesianIndex` tuples. Python tuple objects cannot enter Numba `nopython=True` functions or MPI send buffers without boxing. Every cell is a plain `int64`; multi-index ↔ flat conversion uses `np.ravel_multi_index` / `np.unravel_index` (O(1), no allocation).

**D2 — Half-open intervals `[lo, hi)`.** Makes `point_to_key` injective: no point falls in two cells. Matches standard rasterisation conventions. Exception: `contains_box` uses closed containment for subset tests.

**D3 — Sorted unique key arrays for `BoxSet`.** Alternatives (Python `set`, hash maps) require boxing for Numba/MPI. The O(log N) membership cost of `searchsorted` is acceptable because the dominant cost in every algorithm is dynamics evaluation, not set lookups.

**D4 — `float64` only.** `float32` accumulates rounding error in iterated box containment tests at fine grid resolutions. Phase 3 kernels use float64 throughout.

**D5 — `__slots__` on `Box` and `BoxPartition`.** Prevents `__dict__` allocation per object, reducing GC pressure during subdivision where many temporary `Box` objects are created and discarded.

---

## 11. Correspondence with GAIO.jl

| GAIO.jl | GAIO.py |
|---------|---------|
| `Box{N,T}` | `gaio.core.Box` |
| `BoxGrid{N,T,I}` | `gaio.core.BoxPartition` |
| `BoxSet{P}` | `gaio.core.BoxSet` |
| `SampledBoxMap` | `gaio.maps.GridMap`, `gaio.maps.MonteCarloMap` |
| `GridBoxMap` | `gaio.cuda.AcceleratedBoxMap(backend="cpu"/"gpu")` |
| `construct_transfers` | `gaio.transfer.operator._build_transitions` |
| `relative_attractor` | `gaio.algorithms.attractor.relative_attractor` |
| `unstable_manifold` | `gaio.algorithms.manifolds.unstable_set` |
| `TransferOperator` | `gaio.transfer.operator.TransferOperator` |
| CUDA.jl / Metal.jl dispatch | Numba `@cuda.jit` with explicit kernel launches |
| SIMDExt `CPUSampledBoxMap` | `AcceleratedBoxMap(backend="cpu")` via `@njit(parallel=True)` |

**Notable differences from GAIO.jl:**

1. **Key type**: Julia uses `CartesianIndex`; Python uses flat `int64` (see D1).
2. **GPU**: Julia's `CUDAExt.jl` GPU path has two known bugs (see `docs/VAST_AI_SETUP.md §8`); GAIO.py's GPU path is fully correct.
3. **Multi-GPU**: GAIO.jl has no MPI/multi-GPU support; GAIO.py phases 4–5 add this.
4. **Immutability**: Julia's `subdivide!` mutates; Python's `subdivide` always returns a new object.
5. **Typing**: Julia's parametric types (`Box{N,T}`) are compile-time; Python enforces at runtime via `__init__` assertions and explicit `dtype` coercion.

---

## 12. API Reference

### Core Types

| Symbol | Description |
|--------|-------------|
| `Box(center, radius)` | Axis-aligned hyperrectangle |
| `BoxPartition(domain, n_boxes)` | Uniform Cartesian grid over `domain` |
| `BoxSet(partition, keys)` | Sparse sorted set of active cells |
| `BoxMeasure(partition, keys, weights)` | Discrete measure on a set of cells |

### Box Maps

| Symbol | Description |
|--------|-------------|
| `SampledBoxMap(f, domain, unit_points)` | BoxMap from explicit test-point array |
| `GridMap(f, domain, n_points)` | BoxMap with uniform grid test points |
| `MonteCarloMap(f, domain, n_points)` | BoxMap with random test points |
| `rk4_flow_map(vfield, step_size, steps)` | RK4 flow map closure from a vector field |

### GPU Acceleration

| Symbol | Description |
|--------|-------------|
| `AcceleratedBoxMap(f, domain, unit_pts, f_device, backend)` | Drop-in GPU/CPU-parallel BoxMap |
| `make_cuda_rk4_flow_map(vfield_device, ndim, step_size, steps)` | CUDA RK4 flow map factory |
| `cuda_available()` | Runtime CUDA device check |
| `numba_available()` | Runtime Numba availability check |

### Algorithms

| Symbol | Description |
|--------|-------------|
| `relative_attractor(F, B, steps)` | Subdivision-based attractor |
| `unstable_set(F, B)` | Unstable manifold from seed set |
| `alpha_limit_set(F, B)` | Alpha-limit / stable manifold |
| `preimage(F, B, C)` | Pre-image of `C` restricted to `B` |
| `maximal_invariant_set(F, B)` | Maximal invariant set in `B` |

### Transfer Operator

| Symbol | Description |
|--------|-------------|
| `TransferOperator(F, domain, codomain)` | Sparse Perron-Frobenius matrix |
| `T.mat` | Underlying `scipy.sparse.csc_matrix` |
| `T.eigs(k, which, v0)` | `k` eigenpairs (ARPACK; SLEPc in MPI mode) |
| `T.svds(k)` | `k` singular triplets |
| `T.push_forward(μ)` | Push-forward: `T·μ` |
| `T.pull_back(μ)` | Pull-back (Koopman): `Tᵀ·μ` |
| `T @ μ` | Shorthand for `push_forward(μ)` |

### Graph and Morse Theory

| Symbol | Description |
|--------|-------------|
| `BoxGraph(T)` | Directed graph of the transfer operator |
| `recurrent_set(T)` | Strongly connected recurrent cells |
| `morse_sets(T)` | List of Morse sets (invariant components) |
| `morse_tiles(T)` | Tiles connecting Morse sets |

### Examples

| Command | Description |
|---------|-------------|
| `python -m gaio.examples.four_wing` | 3D Four-Wing attractor (GPU showcase) |
| `python -m gaio.examples.attractor` | Hénon attractor |
| `python -m gaio.examples.invariant_measure_1d` | Logistic map invariant density |
| `python -m gaio.examples.invariant_measure_2d` | 2D invariant measure heatmap |
| `python -m gaio.examples.almost_invariant_sets` | Chua's circuit double-scroll lobes |
| `python -m gaio.examples.unstable_manifold` | Lorenz system unstable manifold |
| `python -m gaio.examples.double_gyre_ftle` | Double-gyre FTLE field |
| `python -m gaio.examples.double_gyre_coherent` | Double-gyre coherent sets |
