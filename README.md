# GAIO.py

**Global Analysis of Invariant Objects** — a Python library for set-oriented numerical analysis of dynamical systems.

GAIO.py is a Python port of [GAIO.jl](https://github.com/gaioguys/GAIO.jl), providing the same subdivision-based algorithms for computing attractors, invariant manifolds, invariant measures, and almost-invariant sets. It runs on any hardware — from a laptop to a multi-GPU HPC cluster — through a unified API that automatically dispatches to the fastest available backend (pure Python, CPU-parallel via Numba, or NVIDIA CUDA).

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Core Concepts](#core-concepts)
  - [Box and BoxPartition](#box-and-boxpartition)
  - [BoxSet](#boxset)
  - [BoxMap](#boxmap)
  - [TransferOperator](#transferoperator)
- [Algorithms](#algorithms)
  - [Relative Attractor](#relative-attractor)
  - [Unstable Manifold](#unstable-manifold)
  - [Invariant Measure](#invariant-measure)
  - [Almost-Invariant Sets and Morse Decomposition](#almost-invariant-sets-and-morse-decomposition)
- [GPU Acceleration](#gpu-acceleration)
  - [Backends](#backends)
  - [AcceleratedBoxMap](#acceleratedboxmap)
  - [Writing a CUDA Flow Map](#writing-a-cuda-flow-map)
  - [Full GPU Pipeline](#full-gpu-pipeline)
- [Examples](#examples)
- [Distributed Computing (MPI)](#distributed-computing-mpi)
- [API Reference](#api-reference)

---

## Overview

Many problems in dynamical systems — finding attractors, computing invariant densities, identifying coherent structures — can be reduced to questions about where sets go under iteration of a map. GAIO.py answers these questions by covering the phase space with a grid of boxes and tracking which boxes map to which other boxes.

This **set-oriented** approach, developed by [Michael Dellnitz and collaborators](https://gaioguys.github.io/GAIO.jl/dev/), is:

- **Rigorous**: the outer-approximation guarantee means no true invariant structure is missed.
- **Scalable**: the subdivision algorithm refines only where the dynamics are active, so memory grows with the complexity of the invariant set, not the phase space volume.
- **Composable**: attractors, invariant measures, Morse decompositions, and almost-invariant sets are all computed from the same `TransferOperator` object.

---

## Installation

### Minimal (CPU only)

```bash
pip install -e .
```

Requires Python ≥ 3.11, NumPy ≥ 1.26, SciPy ≥ 1.12.

### With GPU support

```bash
pip install -e ".[gpu]"
```

Requires an NVIDIA GPU and a matching CUDA toolkit. The library detects CUDA at runtime and falls back to CPU automatically if no GPU is found.

### Full environment (recommended)

Using conda/mamba for a fully pinned environment including CUDA 11.8, MPI, and development tools:

```bash
conda env create -f environment.yml
conda activate gaio
pip install -e .
```

### Optional extras

| Extra | Command | Adds |
|-------|---------|------|
| GPU | `pip install -e ".[gpu]"` | Numba CUDA, CuPy |
| MPI | `pip install -e ".[mpi]"` | mpi4py |
| Dev | `pip install -e ".[dev]"` | pytest, matplotlib, Jupyter, profilers |

---

## Core Concepts

### Box and BoxPartition

A `Box` is an axis-aligned hyperrectangle defined by its center and radius:

```
Box([c₁, c₂, ...], [r₁, r₂, ...])  ↔  [c₁-r₁, c₁+r₁) × [c₂-r₂, c₂+r₂) × ...
```

A `BoxPartition` divides a `Box` into a uniform Cartesian grid. Each cell in the grid is uniquely identified by a flat `int64` key computed from the cell's multi-index:

```python
import numpy as np
from gaio import Box, BoxPartition

domain = Box([0.0, 0.0], [1.0, 1.0])   # unit square centered at origin, shifted
P = BoxPartition(domain, [8, 8])         # 8×8 grid → 64 cells

# Look up which cell contains a point
key = P.point_to_key(np.array([0.3, 0.7]))

# Recover the Box for a given key
cell = P.key_to_box(key)
print(cell.center, cell.radius)
```

### BoxSet

A `BoxSet` is a sparse, sorted collection of active cells within a partition. It supports set algebra and is the fundamental data type passed to and returned by all algorithms.

```python
from gaio import BoxSet

# Full grid (all 64 cells active)
B = BoxSet.full(P)

# Empty grid
E = BoxSet.empty(P)

# From an explicit list of keys
S = BoxSet(P, np.array([0, 1, 5, 10], dtype=np.int64))

# Set operations
union        = B | S
intersection = B & S
difference   = B - S
symmetric    = B ^ S

print(len(B))          # 64
print(B.contains(key)) # True/False
```

A `BoxMeasure` pairs a `BoxSet` with a weight vector — it represents a discrete measure supported on a set of cells, used to store invariant densities and eigenmeasures.

### BoxMap

A `BoxMap` discretises a continuous map `f: ℝⁿ → ℝⁿ` onto a partition. Given a source `BoxSet`, it returns the **outer-approximation image**: every cell that *could* be hit by any point in any source cell.

The library provides three map types, all with the same interface:

| Class | Test-point strategy | When to use |
|-------|---------------------|-------------|
| `SampledBoxMap` | User-supplied `(M, n)` coordinate array | Full control over placement |
| `GridMap` | Uniform Cartesian grid, `n_points` per dimension | Default for deterministic maps |
| `MonteCarloMap` | Random uniform samples | Stochastic maps or high dimensions |

```python
from gaio import SampledBoxMap, GridMap, MonteCarloMap, rk4_flow_map

# Define the map (e.g., Hénon map)
def henon(x):
    return np.array([1 - 1.4 * x[0]**2 + x[1],
                     0.3 * x[0]])

# GridMap with 3×3 = 9 test points per cell
F = GridMap(henon, domain, n_points=3)

# For ODE-based maps: build an RK4 flow map from a vector field
def lorenz_v(x):
    return np.array([10*(x[1]-x[0]),
                     x[0]*(28-x[2])-x[1],
                     x[0]*x[1] - (8/3)*x[2]])

f_lorenz = rk4_flow_map(lorenz_v, step_size=0.05, steps=5)
F_lorenz = SampledBoxMap(f_lorenz, domain, unit_pts)
```

`rk4_flow_map(vfield, step_size, steps)` returns a closure implementing a fixed-step RK4 integrator. This is the standard way to build a box map from an ODE vector field.

### TransferOperator

The `TransferOperator` builds the sparse matrix representation of the Perron-Frobenius operator: a column-stochastic matrix `T` where `T[i, j]` is the fraction of test points from cell `j` that land in cell `i`.

```python
from gaio import TransferOperator

# Build T on the attractor A (square matrix)
T = TransferOperator(F, A, A)
print(T)   # TransferOperator(1024x1024, nnz=4812, domain=1024 cells, codomain=1024 cells)

# The underlying sparse matrix
print(T.mat.shape, T.mat.nnz)

# Push-forward a measure: μ → T·μ
mu_new = T.push_forward(mu)

# Pull-back (Koopman/adjoint): μ → Tᵀ·μ
mu_pull = T.pull_back(mu)

# Operator syntax: T @ μ ≡ push_forward(μ)
mu_new = T @ mu
```

---

## Algorithms

### Relative Attractor

The **subdivision algorithm** for attractors iteratively refines a covering of the phase space. At each step every active cell is subdivided and only the cells that map into themselves (under the set-valued map) are kept. After enough steps, the covering converges to the attractor.

```python
from gaio import relative_attractor

domain = Box([0.0, 0.0], [1.5, 1.5])
P = BoxPartition(domain, [2, 2])       # coarse initial grid
F = GridMap(henon, domain, n_points=4)

S = BoxSet.full(P)
A = relative_attractor(F, S, steps=16)

print(f"Attractor: {len(A)} cells")
```

Each step doubles the resolution in every dimension, so after `steps=16` the partition has `2^16 = 65,536` cells along each axis (in 1D). Resolution grows as `2^steps / initial_partition_size`.

### Unstable Manifold

`unstable_set` grows the unstable manifold from a seed set by repeatedly taking the image and keeping cells not already in the set:

```python
from gaio import unstable_set

# Seed: the fixed point or a small neighbourhood
seed = BoxSet(P, keys_near_fixed_point)
W_u = unstable_set(F, seed)
```

`alpha_limit_set` computes the analogous stable/alpha-limit set (pre-images).

### Invariant Measure

The **invariant measure** (physical measure / SRB measure) is the eigenvector of the Perron-Frobenius operator corresponding to eigenvalue 1. It describes the long-term time-average density of trajectories on the attractor.

```python
from gaio import TransferOperator

T = TransferOperator(F, A, A)

# Compute leading k eigenpairs
eigenvalues, eigenmeasures = T.eigs(k=3)

# Sort by descending |λ|
order = np.argsort(-np.abs(eigenvalues))
mu_inv = eigenmeasures[order[0]]   # eigenmeasure for λ ≈ 1

# Normalise to unit mass
weights = np.abs(mu_inv._weights)
weights /= weights.sum()
```

For the full 1D example (logistic map):

```bash
python -m gaio.examples.invariant_measure_1d
```

### Almost-Invariant Sets and Morse Decomposition

**Almost-invariant sets** are regions of phase space where trajectories spend a long time before leaving — the "lobes" of a chaotic attractor. They are identified by the second eigenvector of the transfer operator (eigenvalue λ₂ close to but less than 1): the sign of the eigenvector partitions the attractor into two almost-invariant pieces.

```python
eigenvalues, eigenmeasures = T.eigs(k=2)
mu2 = eigenmeasures[1]   # second eigenvector

# Partition by sign
positive_keys = mu2._keys[mu2._weights > 0]
negative_keys = mu2._keys[mu2._weights < 0]

set_A = BoxSet(A.partition, positive_keys)
set_B = BoxSet(A.partition, negative_keys)
```

A full **Morse decomposition** — a coarse-to-fine decomposition of the chain-recurrent set into invariant "Morse sets" connected by gradient-like flow — is computed from the transfer operator's graph structure:

```python
from gaio import morse_sets, morse_tiles, recurrent_set

# Recurrent set: cells that are strongly connected in the transfer graph
R = recurrent_set(T)

# Morse decomposition
M_sets, M_tiles = morse_sets(T), morse_tiles(T)
```

---

## GPU Acceleration

### Backends

GAIO.py provides three computation backends for the map application step (evaluating `f` on all test points):

| Backend | Hardware | Library | Typical speedup over Python |
|---------|----------|---------|----------------------------|
| `"python"` | CPU (single-core) | NumPy + Python loop | 1× (baseline) |
| `"cpu"` | CPU (multi-core) | Numba `@njit(parallel=True)` | 4–16× |
| `"gpu"` | NVIDIA GPU | Numba `@cuda.jit` | 100–1000× (large N) |

The backend is selected at construction time and transparent to all downstream algorithms.

### AcceleratedBoxMap

`AcceleratedBoxMap` is a drop-in replacement for `SampledBoxMap` that dispatches map evaluation to a compiled backend. It takes an additional `f_device` argument — a `@cuda.jit(device=True)` function for the GPU path — and a `backend` string.

```python
from numba import cuda
from gaio import AcceleratedBoxMap, cuda_available

@cuda.jit(device=True)
def henon_device(x, out):
    out[0] = 1.0 - 1.4 * x[0] * x[0] + x[1]
    out[1] = 0.3 * x[0]

unit_pts = np.array([[-0.5,-0.5],[-0.5,0.5],[0.5,-0.5],[0.5,0.5]], dtype=np.float64)

if cuda_available():
    F = AcceleratedBoxMap(henon, domain, unit_pts,
                          f_device=henon_device, backend="gpu")
else:
    F = AcceleratedBoxMap(henon, domain, unit_pts, backend="cpu")

# Usage identical to SampledBoxMap — GPU dispatched transparently
A = relative_attractor(F, BoxSet.full(P), steps=16)
T = TransferOperator(F, A, A)   # also GPU-accelerated
```

`cuda_available()` returns `True` if Numba can find a CUDA device at runtime.

### Writing a CUDA Flow Map

For ODE-based maps, GAIO.py provides `make_cuda_rk4_flow_map` — a factory that wraps a user-defined vector field device function in a full fixed-step RK4 integrator, matching the CPU `rk4_flow_map` pattern exactly.

You write only the vector field (3 lines for a 3D system); the factory handles all RK4 bookkeeping:

```python
from numba import cuda
from gaio.cuda import make_cuda_rk4_flow_map, AcceleratedBoxMap
from gaio import rk4_flow_map

# CPU version (for correctness reference and CPU fallback)
def lorenz_v(x):
    return np.array([10*(x[1]-x[0]),
                     x[0]*(28-x[2])-x[1],
                     x[0]*x[1] - (8/3)*x[2]])

f_lorenz_cpu = rk4_flow_map(lorenz_v, step_size=0.05, steps=5)

# GPU version — write only the vector field
@cuda.jit(device=True)
def lorenz_v_device(x, out):
    out[0] = 10.0 * (x[1] - x[0])
    out[1] = x[0] * (28.0 - x[2]) - x[1]
    out[2] = x[0] * x[1] - (8.0/3.0) * x[2]

f_lorenz_gpu = make_cuda_rk4_flow_map(lorenz_v_device, ndim=3,
                                       step_size=0.05, steps=5)

F = AcceleratedBoxMap(f_lorenz_cpu, domain, unit_pts,
                      f_device=f_lorenz_gpu, backend="gpu")
```

Both `rk4_flow_map` and `make_cuda_rk4_flow_map` implement the **same Butcher tableau RK4** with the same step sizes, so CPU and GPU results are numerically identical.

The `f_device` signature is `f_device(x, out) -> None`: it reads from `x` (1D array of length `ndim`) and writes the result in-place to `out`. It cannot return a new array (no heap allocation in CUDA device functions).

### Full GPU Pipeline

Every step of the GAIO workflow dispatches to GPU when `F` is an `AcceleratedBoxMap(backend="gpu")`:

```
BoxSet.full(P)
    │
    ▼
relative_attractor(F, S, steps=N)          ← GPU: CUDA kernel over all K×M test points
    │  returns BoxSet (attractor A)
    ▼
TransferOperator(F, A, A)                  ← GPU: same CUDA kernel, batched over A
    │  returns sparse matrix T (on CPU, scipy.sparse)
    ▼
T.eigs(k=1)                                ← CPU: ARPACK/BLAS (not the bottleneck)
    │  returns eigenvalues, eigenmeasures
    ▼
visualise / analyse
```

The sparse linear algebra (`eigs`, `svds`, `push_forward`, `pull_back`) runs on CPU via scipy. This is correct — a sparse matrix of order 10,000–100,000 is solved in seconds by ARPACK; GPU sparse eigensolvers do not offer practical speedups at this scale.

---

## Examples

All examples run as Python modules and accept command-line arguments:

### Hénon Attractor

```bash
python -m gaio.examples.attractor
python -m gaio.examples.attractor --steps 20 --no-gpu
```

Computes the Hénon attractor using the subdivision algorithm. Uses GPU automatically if available.

### Invariant Measure (1D — Logistic Map)

```bash
python -m gaio.examples.invariant_measure_1d
```

Computes the invariant density of the logistic map `f(x) = 4x(1-x)` on [0,1]. The exact invariant density `ρ(x) = 1/(π√(x(1-x)))` is plotted for comparison.

### Invariant Measure (2D)

```bash
python -m gaio.examples.invariant_measure_2d
```

Computes the invariant measure of a 2D map and visualises it as a heatmap.

### Transfer Operator

```bash
python -m gaio.examples.transfer_operator
```

Demonstrates direct construction and spectral analysis of the transfer operator.

### Unstable Manifold (Lorenz System)

```bash
python -m gaio.examples.unstable_manifold
```

Grows the unstable manifold of the Lorenz system from a seed near an equilibrium using the 3D subdivision algorithm.

### Almost-Invariant Sets (Chua's Circuit)

```bash
python -m gaio.examples.almost_invariant_sets
python -m gaio.examples.almost_invariant_sets --no-show
```

Identifies the two lobes of Chua's circuit double-scroll attractor using the second eigenvector of the transfer operator.

### Four-Wing Attractor (3D, GPU showcase)

```bash
# Default (GPU if available, grid_res=4, steps=14)
python -m gaio.examples.four_wing

# Finer resolution
python -m gaio.examples.four_wing --grid-res 6 --steps 18

# Force CPU
python -m gaio.examples.four_wing --no-gpu

# Headless (no display window, for clusters)
python -m gaio.examples.four_wing --no-show
```

The four-wing attractor is a 3D chaotic system. This example demonstrates the complete GPU pipeline: GPU-accelerated attractor computation, GPU-accelerated transfer operator construction, and 3D/2D visualisation via PyVista and matplotlib. Memory scales with the number of attractor cells: `grid_res=4, steps=14` uses ~13 MB GPU VRAM and is safe on machines with 8+ GB RAM.

---

## Distributed Computing (MPI)

> The MPI backend is under active development.

GAIO.py is structured to support distributed computation across multiple nodes or GPUs via MPI. The natural decomposition is by domain cells: each MPI rank owns a disjoint slice of the `domain._keys` array, computes its local test-point batch independently, and contributes its COO entries to a global `Allgatherv` before assembling the shared sparse matrix.

The `gaio/mpi/` subpackage provides the infrastructure. To run with MPI:

```bash
mpirun -n 4 python my_gaio_script.py
```

For CUDA-aware MPI (GPUDirect RDMA — direct GPU-to-GPU transfers without CPU staging), ensure your MPI build supports it:

```bash
python -c "from gaio.mpi import check_cuda_aware_mpi; check_cuda_aware_mpi()"
```

### MPI Decomposition Strategy

GAIO.py uses **Morton (Z-order) curve sorting** for domain decomposition. Morton codes interleave the bits of cell multi-indices, producing a space-filling curve that preserves spatial locality: cells that are spatially adjacent are also adjacent in Morton order. Decomposing by contiguous Morton ranges gives each rank a spatially compact subdomain, which minimises the number of inter-rank boundary cells and reduces cross-rank communication from O(K) to O(K^{(d-1)/d}).

---

## API Reference

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
| `AcceleratedBoxMap(f, domain, unit_points, f_device, backend)` | Drop-in GPU/CPU-parallel BoxMap |
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

### Transfer Operator and Spectral Analysis

| Symbol | Description |
|--------|-------------|
| `TransferOperator(F, domain, codomain)` | Sparse Perron-Frobenius matrix |
| `T.mat` | Underlying `scipy.sparse.csc_matrix` |
| `T.eigs(k, which, v0)` | `k` eigenpairs (ARPACK) |
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

---

## License

MIT License. See `LICENSE` for details.

GAIO.py is based on [GAIO.jl](https://github.com/gaioguys/GAIO.jl) by Michael Dellnitz, Oliver Junge, and collaborators. The set-oriented methods are described in:

> Dellnitz, M. and Junge, O. (2002). *Set Oriented Numerical Methods for Dynamical Systems.* In Handbook of Dynamical Systems, Vol. 2. Elsevier.
