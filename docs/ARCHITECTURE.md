# GAIO.py — Architecture & Developer Guide

> **What this document is.**  A self-contained reference that explains *why*
> each design decision was made, *what* mathematical object it corresponds to
> in GAIO.jl, and *how* the Python implementation encodes it.  It is meant to
> be read top-to-bottom on first contact and used as a lookup reference later.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Mathematical Background](#2-mathematical-background)
3. [Directory Structure](#3-directory-structure)
4. [Phase Roadmap](#4-phase-roadmap)
5. [Phase 1 — Core Data Structures](#5-phase-1--core-data-structures)
   - 5.1 [Box](#51-box--coreboxyp)
   - 5.2 [BoxPartition](#52-boxpartition--corepartitionpy)
   - 5.3 [BoxSet](#53-boxset--coreboxsetpy)
6. [Key Design Decisions](#6-key-design-decisions)
7. [Environment Setup](#7-environment-setup)
8. [Correspondence with GAIO.jl](#8-correspondence-with-gaiojl)

---

## 1. Project Overview

**GAIO** (Global Analysis of Invariant Objects) is a *set-oriented numerical
framework* for studying dynamical systems.  Instead of tracking individual
trajectories, it discretises the state space into a finite collection of boxes
(hyperrectangles) and asks questions like:

- Which boxes map back into themselves? → **invariant set / attractor**
- Which boxes are visited arbitrarily often? → **chain-recurrent set**
- How do unstable/stable manifolds propagate? → **manifold computation**

The original library is [GAIO.jl](https://github.com/gaioguys/GAIO.jl) (Julia).
This repository is a full Python port targeting three additional requirements
not present in the Julia version:

| Requirement | Technology |
|---|---|
| Strict OOP + typed arrays | NumPy `float64` / `int64` throughout |
| GPU acceleration | Numba `@cuda.jit` kernels (Phase 3) |
| Distributed memory | `mpi4py` domain decomposition (Phase 4) |

---

## 2. Mathematical Background

### 2.1 Boxes

A **box** (or *hyperrectangle*, or *k-cell*) in ℝⁿ is:

```
B(c, r) = { x ∈ ℝⁿ : cᵢ - rᵢ ≤ xᵢ < cᵢ + rᵢ  for all i }
```

It is defined by its **centre** `c ∈ ℝⁿ` and **radius** `r ∈ ℝⁿ` (component-wise
half-widths).  The interval is **half-open**: it includes the lower boundary but
excludes the upper boundary.  This convention is necessary so that boxes tile
the space without overlap.

### 2.2 Box Partition

A **box partition** of a domain `B(c₀, r₀)` into a `d₁ × d₂ × … × dₙ` grid
produces `N = d₁ · d₂ · … · dₙ` cells.  Cell `(i₁, …, iₙ)` has:

```
cell_radius[k]  = r₀[k] / dₖ
cell_width[k]   = 2 · r₀[k] / dₖ
center[k]       = lo[k] + cell_width[k] · (iₖ + 0.5)
```

where `lo = c₀ - r₀` is the lower corner of the domain and `iₖ ∈ [0, dₖ)`.

### 2.3 Transfer Operator (Ulam's Method)

For a map `f : ℝⁿ → ℝⁿ`, the **transfer matrix** `P` has entry:

```
Pᵢⱼ ≈  |{ test points in Bᵢ  that map into Bⱼ }|
        ────────────────────────────────────────────
              total test points in Bᵢ
```

The matrix `P` is a row-stochastic approximation of the **Perron–Frobenius
operator** of `f`.  Its left-leading eigenvector approximates the **natural
invariant measure** of `f`.

### 2.4 Relative Attractor

Given a compact set `Q` and a map `f`, the **relative attractor** `A(f, Q)` is
the largest set `A ⊆ Q` such that `f(A) ⊆ A`.  It is computed by the
**subdivision algorithm**:

```
Q₀ = Q
Qₖ₊₁ = { B ∈ Qₖ : f(B) ∩ Qₖ ≠ ∅ }    (backward-image intersection)
A = lim_{k→∞} Qₖ
```

---

## 3. Directory Structure

```
GAIO.py/                        ← git repo root (not a Python package)
│
├── pyproject.toml              ← pip install -e .  makes `gaio` importable everywhere
├── environment.yml             ← conda environment definition (env name: gaio)
├── ARCHITECTURE.md             ← this file
│
├── gaio/                       ← the installable Python package
│   ├── __init__.py             ← re-exports Box, BoxPartition, BoxSet
│   │
│   ├── core/                   ← Phase 1 ✅
│   │   ├── __init__.py
│   │   ├── box.py              ← Box: typed hyperrectangle
│   │   ├── partition.py        ← BoxPartition: uniform grid, flat int64 key scheme
│   │   └── boxset.py           ← BoxSet: sparse sorted set of active cells
│   │
│   ├── maps/                   ← Phase 2
│   │   ├── __init__.py
│   │   ├── base.py             ← abstract BoxMap protocol
│   │   ├── grid_map.py         ← deterministic test-point grid per cell
│   │   └── montecarlo_map.py   ← Monte Carlo test-point sampling
│   │
│   ├── algorithms/             ← Phase 2
│   │   ├── __init__.py
│   │   ├── attractor.py        ← relative_attractor, subdivision loop
│   │   ├── manifolds.py        ← stable / unstable manifold computation
│   │   └── transfer.py         ← transfer_matrix (Ulam's method), Perron-Frobenius
│   │
│   ├── cuda/                   ← Phase 3
│   │   ├── __init__.py
│   │   └── kernels.py          ← @cuda.jit kernels: batch containment, COO accumulation
│   │
│   ├── mpi/                    ← Phase 4
│   │   ├── __init__.py
│   │   ├── decomp.py           ← distribute flat keys across MPI ranks
│   │   └── comm.py             ← halo exchange, Allreduce patterns
│   │
│   └── utils/
│       ├── __init__.py
│       └── profiling.py        ← cProfile / line_profiler helpers
│
└── old/                        ← original prototype files (reference only)
    ├── box.py
    ├── boxgrid.py
    └── structures.ipynb
```

---

## 4. Phase Roadmap

| Phase | Deliverable | Key dependencies |
|---|---|---|
| **1** ✅ | `Box`, `BoxPartition`, `BoxSet` — typed NumPy foundation | `numpy` |
| **2** | `BoxMap` variants + `relative_attractor`, `transfer_matrix` | `numpy`, `scipy.sparse` |
| **3** | Numba CUDA kernels replacing Phase 2 inner loops | `numba`, `cudatoolkit` |
| **4** | MPI domain decomposition + distributed manifold loop + profiling | `mpi4py`, `line_profiler` |

Each phase builds strictly on the previous.  Phases 3 and 4 are **drop-in
accelerators**: the CPU algorithms from Phase 2 are preserved as fallbacks.

---

## 5. Phase 1 — Core Data Structures

### 5.1 `Box` — `gaio/core/box.py`

#### What it models

A single axis-aligned hyperrectangle `B(c, r) = [c-r, c+r)`.

#### Memory layout

```python
class Box:
    center : np.ndarray  # shape (n,), dtype float64, C-contiguous
    radius : np.ndarray  # shape (n,), dtype float64, C-contiguous
```

`__slots__` is used so that no `__dict__` is allocated — important when
millions of Box objects would otherwise be created during subdivision.

In practice, algorithms never store individual Box objects at scale.
`Box` is used for:
- Describing the **domain** passed to `BoxPartition`
- The return value of `BoxPartition.key_to_box(k)` (geometry lookup)
- One-off spatial queries (`contains_point`, `intersects`, `intersection`)

#### Half-open convention

`contains_point` tests `lo ≤ p < hi`.  This matches GAIO.jl's `∈` semantics
and ensures that `point_to_key` is injective: exactly one cell claims any
interior point.

#### `subdivide(dim)` and `subdivide_all()`

```
B(c, r)  ──subdivide(0)──►  B(c - r[0]/2 · e₀,  r with r[0]/2)
                             B(c + r[0]/2 · e₀,  r with r[0]/2)
```

`subdivide_all()` applies bisection along every dimension in sequence, producing
`2ⁿ` children whose union is the original box.

#### Key methods at a glance

| Method | Returns | Notes |
|---|---|---|
| `contains_point(p)` | `bool` | half-open `[lo, hi)` |
| `intersects(other)` | `bool` | open-interior check |
| `intersection(other)` | `Box` | raises if empty |
| `bounding_box(other)` | `Box` | smallest enclosing box |
| `rescale(u)` | `ndarray` | maps `[-1,1]ⁿ → box`; used by BoxMap test-point generators |
| `normalize(p)` | `ndarray` | inverse of `rescale` |
| `subdivide(dim)` | `(Box, Box)` | bisect along one axis |
| `subdivide_all()` | `list[Box]` | `2ⁿ` children |

---

### 5.2 `BoxPartition` — `gaio/core/partition.py`

#### What it models

A uniform Cartesian grid that divides a `Box` domain into `prod(dims)` equal
cells.  This is the Python equivalent of `BoxGrid{N,T,I}` in GAIO.jl's
`partition_regular.jl`.

#### The flat key scheme

Every cell is identified by a **single `int64`** obtained by flattening its
multi-dimensional grid index in **C (row-major) order**:

```python
flat_key = np.ravel_multi_index((i₀, i₁, …, iₙ₋₁), dims)
# inverse:
multi_index = np.unravel_index(flat_key, dims)
```

**Why flat keys instead of tuple keys?**

Tuple keys require Python object boxing.  Numba `@cuda.jit` kernels and MPI
`send`/`recv` buffers can only hold contiguous C-typed arrays.  An array of
`int64` flat keys (`np.ndarray[int64]`) passes directly to both without any
conversion or heap allocation.

#### Cell geometry formula

```
lo             = domain.center - domain.radius
cell_width[k]  = 2 · domain.radius[k] / dims[k]
cell_radius[k] = domain.radius[k] / dims[k]

center of cell m = lo + cell_width · (m + 0.5)
                 = lo + 2 · cell_radius · (m + 0.5)
```

#### `point_to_key(p)` — spatial lookup

```python
multi = floor((p - lo) / cell_width)     # which cell in each dimension?
multi = clip(multi, 0, dims - 1)         # guard floating-point edge cases
key   = ravel_multi_index(multi, dims)
```

`point_to_key_batch(points)` processes an `(N, n)` array in one vectorised
call.  Out-of-domain points receive key `-1`.  This is the function that Phase 3
replaces with a CUDA kernel.

#### `keys_in_box(query)` — range query

Instead of scanning all `prod(dims)` cells, the method computes the tight
multi-index range that could intersect the query box, then uses
`np.meshgrid` + `np.ravel_multi_index` to enumerate only the candidate keys.
This is `O(hits)` instead of `O(size)`.

#### Key methods at a glance

| Method | Returns | Notes |
|---|---|---|
| `key_to_box(k)` | `Box` | geometry of cell `k` |
| `key_to_multi(k)` | `ndarray (n,)` | flat → multi-index |
| `multi_to_key(m)` | `int` | multi-index → flat |
| `point_to_key(p)` | `int \| None` | `None` if outside domain |
| `point_to_key_batch(pts)` | `ndarray (N,)` | `-1` for out-of-domain |
| `keys_in_box(query)` | `ndarray` | range query, `O(hits)` |
| `all_keys()` | `ndarray` | `[0, 1, …, size-1]` |
| `subdivide(dim)` | `BoxPartition` | 2× finer along `dim` |
| `subdivide_all()` | `BoxPartition` | 2× finer in every dimension |

---

### 5.3 `BoxSet` — `gaio/core/boxset.py`

#### What it models

A finite subset of cells drawn from one `BoxPartition`.  Algorithms operate
almost exclusively on `BoxSet` objects.  This is the Python equivalent of
`BoxSet` in GAIO.jl's `boxset.jl`.

#### Internal representation

```python
class BoxSet:
    partition : BoxPartition
    _keys     : np.ndarray   # shape (N,), dtype int64, sorted, unique, C-contiguous
```

The `_keys` array is **always sorted and unique** after construction (`np.unique`
is called in `__init__`).  This invariant enables:

- **O(log N) membership** via `np.searchsorted`
- **O(N log N) set algebra** via `np.union1d / intersect1d / setdiff1d / setxor1d`
- **Direct kernel passing**: the array is a valid CUDA device buffer in Phase 3
- **Direct MPI buffer**: the array is a valid `MPI.INT64_T` send buffer in Phase 4

#### Immutability

All set operations return **new** `BoxSet` objects; `_keys` is never mutated
after construction.  This makes it safe to share references across threads, GPU
copies, and MPI ranks.

#### `centers()` — vectorised geometry access

```python
def centers(self) -> np.ndarray:           # shape (N, n), float64, C-contiguous
    multi = np.stack(np.unravel_index(self._keys, self.partition.dims), axis=1)
    return lo + 2 * cell_radius * (multi + 0.5)
```

This is the primary geometry accessor used by all Phase 2 algorithms.  It
avoids constructing `N` individual `Box` objects and runs entirely in NumPy.

#### `subdivide(dim)` — refinement

Each active cell maps to exactly 2 children in the `2×`-finer partition along
`dim`.  The spatial coverage is identical; only the resolution changes.

```
cell m  →  child 2m      (lower half)
        →  child 2m + 1  (upper half)
```

#### Convenience constructors

| Constructor | Description |
|---|---|
| `BoxSet.full(p)` | all `prod(dims)` cells |
| `BoxSet.empty(p)` | zero cells |
| `BoxSet.cover(p, points)` | smallest set covering a point cloud |
| `BoxSet.from_box(p, query)` | all cells intersecting a Box |

#### Set algebra operators

| Python operator | Set operation |
|---|---|
| `A \| B` | union `A ∪ B` |
| `A & B` | intersection `A ∩ B` |
| `A - B` | difference `A \ B` |
| `A ^ B` | symmetric difference `A △ B` |
| `A <= B` | subset `A ⊆ B` |
| `A >= B` | superset `A ⊇ B` |

---

## 6. Key Design Decisions

### D1 — Flat `int64` keys everywhere

Julia uses `CartesianIndex` tuples as cell identifiers.  Python tuple
objects cannot enter Numba JIT functions (`nopython=True`) without boxing,
and cannot be sent over MPI without serialisation.

**Decision:** Every cell is a plain `int64`.  Multi-index ↔ flat conversion
uses `np.ravel_multi_index` / `np.unravel_index` (O(1), no allocation).
All key arrays are `dtype=int64, order='C'`.

### D2 — Half-open intervals `[lo, hi)`

Julia uses closed intervals `[lo, hi]`.  For a partition to tile without
overlap, adjacent cells must agree on which one owns the shared boundary.
The half-open convention — include `lo`, exclude `hi` — makes `point_to_key`
injective (no point falls in two cells) and matches standard rasterisation
conventions.

**Exception:** `contains_box` uses closed containment to check whether one
box is *fully inside* another (for subset tests in algorithms).

### D3 — Sorted unique key arrays for `BoxSet`

Alternatives considered:
- **Python `set`**: O(1) membership, but unhashable by NumPy, requires boxing
  for Numba/MPI.
- **Hash array (`np.ndarray` + hash map)**: fast lookup, but set ops require
  auxiliary data structures.
- **Sorted array**: O(log N) membership via `searchsorted`, O(N log N) set
  ops via `union1d` etc., directly usable as CUDA/MPI buffers.

**Decision:** Sorted unique `int64` array.  The O(log N) vs O(1) membership
penalty is acceptable because the dominant cost in every algorithm is the
dynamics evaluation (`f(x)`), not the set lookups.

### D4 — `float64` only, no `float32`

CUDA `float32` is faster on consumer GPUs, but `float32` accumulates
significant rounding error in iterated box containment tests at high spatial
resolution (fine grids).  For scientific correctness `float64` is mandatory.
Phase 3 kernels operate in `float64` throughout.

### D5 — `__slots__` on `Box` and `BoxPartition`

`__slots__` prevents Python from allocating a `__dict__` per object.  In
Phase 2, the algorithms generate and discard many temporary `Box` objects
(one per cell per subdivision step).  Without `__slots__` this causes
significant GC pressure.

---

## 7. Environment Setup

```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate gaio

# Install the package in editable mode (run once; works from any directory after)
cd /path/to/GAIO.py
pip install -e .

# Verify — works from any working directory
python -c "from gaio import Box, BoxPartition, BoxSet; print('OK')"
```

### Package roles

| Package | Phase | Purpose |
|---|---|---|
| `numpy` | 1+ | typed arrays, `ravel_multi_index`, set ops |
| `scipy` | 2 | sparse CSR transfer matrix, eigenvectors |
| `numba` | 3 | `@cuda.jit` GPU kernels, `@njit` CPU JIT |
| `cudatoolkit 11.8` | 3 | CUDA runtime matched to numba 0.59 |
| `mpi4py` | 4 | distributed send/recv, `Allreduce`, `Scatter` |
| `cupy` | 3 (opt.) | CuPy arrays for host↔device transfers |
| `line_profiler` | 4 | per-line timing of manifold hot paths |
| `matplotlib` | — | visualisation of box sets and manifolds |

---

## 8. Correspondence with GAIO.jl

| GAIO.jl type / file | GAIO.py equivalent |
|---|---|
| `Box{N,T}` in `box.jl` | `gaio.core.Box` |
| `BoxGrid{N,T,I}` in `partition_regular.jl` | `gaio.core.BoxPartition` |
| `BoxTree{N,T,I}` in `partition_tree.jl` | *(Phase 2 stretch goal)* |
| `BoxSet{P}` in `boxset.jl` | `gaio.core.BoxSet` |
| `SampledBoxMap` in `boxmap_sampled.jl` | `gaio.maps.GridMap`, `gaio.maps.MonteCarloMap` (Phase 2) |
| `IntervalBoxMap` in `boxmap_intervals.jl` | *(not planned — requires IntervalArithmetic.jl port)* |
| `construct_transfers` | `gaio.algorithms.transfer.transfer_matrix` (Phase 2) |
| `relative_attractor` | `gaio.algorithms.attractor.relative_attractor` (Phase 2) |
| `unstable_manifold` | `gaio.algorithms.manifolds.unstable_manifold` (Phase 2) |

### Notable differences from GAIO.jl

1. **Key type**: Julia uses `CartesianIndex`; Python uses flat `int64` (see D1).
2. **Partition type**: Julia has both `BoxGrid` (regular) and `BoxTree`
   (adaptive binary tree); Phase 1 implements only the regular grid.  The tree
   partition is a Phase 2 stretch goal.
3. **In-place mutation**: Julia's `subdivide!` mutates the partition; Python's
   `subdivide` always returns a new object (immutability, see D3/D5).
4. **GPU backend**: Julia uses CUDA.jl / Metal.jl dispatch; Python uses
   Numba `@cuda.jit` with explicit kernel launches.
5. **Typing**: Julia's parametric types (`Box{N,T}`) express dimension and
   precision at compile time; Python enforces them at runtime via `__init__`
   assertions and explicit `dtype` coercion.
