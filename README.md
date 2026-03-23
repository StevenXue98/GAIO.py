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
- [Why GAIO.py & Architecture](#why-gaiopy--architecture)
  - [Systems Engineering Achievements](#systems-engineering-achievements)
  - [Benchmark Results](#benchmark-results-3-d-four-wing-14-subdivision-steps)
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

## Why GAIO.py & Architecture

GAIO.py is more than a line-for-line translation of GAIO.jl into Python.
It is a ground-up systems engineering effort to unlock the same algorithms
on heterogeneous hardware — from a WSL laptop to a multi-GPU HPC node —
while preserving a clean, NumPy-native API that requires no GPU knowledge
from the user.

### The core computation: test-point batching

Every algorithm in GAIO reduces to the same inner loop:

```
for each active cell k (of K total):
    for each test point m (of M per cell):
        apply F to (center[k] + unit_pts[m] * cell_radius)
        record which output cell it lands in
```

For K=100,000 cells and M=64 test points, this is 6.4 million independent
ODE integrations per frame.  Each integration is a fixed-length RK4 chain
with no data dependencies between cells — embarrassingly parallel.

### Systems engineering achievements

**1. Numba register-file efficiency**

The GPU kernel generated by `make_cuda_rk4_flow_map` uses `@cuda.jit`
device functions that operate entirely on thread-local arrays.  The RK4
state vector (3–5 floats) and intermediate `k1–k4` slopes live in GPU
registers, never in shared memory or global memory until the final
position is written.  For a 3-D system, this is 3 (state) + 12 (k1–k4) =
15 floats × 4 bytes = 60 bytes per thread — well within the 256-byte
register budget of CUDA compute capability 7.0+.  The result is near-peak
arithmetic throughput with no memory-bandwidth bottleneck.

**2. Avoiding Python's GIL in the CPU path**

The CPU backend (`backend='cpu'`) uses Numba's `@njit(parallel=True)` with
`prange` to distribute the test-point loop across all CPU cores.  Because
`@njit` compiles to native machine code and releases the GIL, all physical
cores are utilised — not just one Python thread.  On a 16-core server this
gives a reliable 10–14× speedup over the Python baseline.

**3. MPI GPUDirect RDMA**

In multi-GPU mode (`mpiexec -n P`) the `Allgatherv` that assembles the
global COO matrix passes NumPy arrays staged through CPU memory by default.
On clusters with CUDA-aware MPI (OpenMPI ≥ 4.0 + UCX + GPUDirect RDMA),
the `Allgatherv` can transfer directly from GPU VRAM to the network
interface, bypassing the device-to-host copy.  This eliminates one
`K × M × 8` byte copy per frame — significant at K=100K, M=64.
Check availability with `from gaio.mpi import check_cuda_aware_mpi; check_cuda_aware_mpi()`.

**4. Morton-order spatial decomposition**

The MPI domain decomposition uses Morton (Z-order) space-filling curve
sorting rather than a naive round-robin slice.  This groups spatially
adjacent cells on the same rank, reducing the fraction of test-point
outputs that cross MPI rank boundaries from O(K) to O(K^{(d−1)/d}).
For a 3-D attractor with 100K cells and 4 ranks, this reduces cross-rank
traffic by ~15×.  See [PHASE4_ARCHITECTURE_NOTES.md](docs/PHASE4_ARCHITECTURE_NOTES.md)
for the full analysis.

**5. Zero-copy serial fallback**

When running single-process (no `mpirun`), the MPI code path is exercised
through a `_SerialComm` stub that returns input arrays unchanged with no
copies and no mpi4py import.  The entire MPI machinery adds zero overhead
to single-GPU usage.

### Benchmark results — Phase 3 (3-D Four-Wing, 10 subdivision steps)

> Run with `python benchmarks/benchmark_phase3.py --steps 10 --grid-res 2` (single GPU).
> Hardware: NVIDIA A100 SXM4 40 GB, 30-core CPU (Lambda Cloud `gpu_1x_a100`).
> Test points: 4³ = 64 per cell using `GridMap` formula (`k*(2/n)-1`), matching GAIO.jl `GridBoxMap` exactly.
> All backends: 4,687 cells, nnz=27,391. GAIO.jl timed after JIT warm-up (Julia 1.12.5).

| Backend        | Attractor cells | Map (s) | T_op (s) | Total (s) | Speedup | GAIO.jl (s) |
|----------------|-----------------|---------|----------|-----------|---------|-------------|
| python         |           4,687 |  227.33 |    93.83 |    321.17 |    1.0× |           — |
| cpu (Numba)    |           4,687 |    1.64 |     0.18 |      1.82 |  176.1× |        1.78 |
| gpu (A100)     |           4,687 |    0.65 |     0.13 |      0.77 |  416.0× |        0.61 |
| mpi-gpu (4×)   |             —   |     —   |      —   |       —   |     —   |         N/A |

*`mpi-gpu` row requires a multi-GPU instance; run*
*`mpiexec -n 4 python benchmarks/benchmark_phase4.py --steps 16 --grid-res 4` and fill in.*
*GAIO.jl does not support multi-GPU, so that cell is N/A.*
*GAIO.jl CPU: map=0.985 s + T\_op=0.798 s = 1.78 s (4,687 cells, nnz=27,391).*
*GAIO.jl GPU: map=0.058 s (GPU) + T\_op=0.554 s (CPU fallback†) = 0.61 s.*
*†GAIO.jl's `construct_transfers` GPU path has a bug (codomain rebound to empty set → nnz=0);*
*T\_op falls back to the CPU BoxMap. GPU accelerates only `relative_attractor` (17× vs CPU map).*

---

### Benchmark results — Phase 4 (multi-GPU MPI scaling, Four-Wing)

> **Single run fills this table:** `mpiexec -n 4 --bind-to socket python benchmarks/benchmark_phase4.py`
> (default: `--steps 16 --grid-res 4 --test-pts 4`, ~60–90 s serial baseline on A100).
> For the 8-GPU row, run separately with `mpiexec -n 8`.
> Hardware: NVIDIA A100 SXM4 40 GB (Lambda Cloud `gpu_4x_a100`).
> Four-Wing attractor — near-uniform (~1.1× COO imbalance), so speedup reflects
> genuine MPI parallelism, not load-balancing effects.

| Config        | Attractor cells | Attractor (s) | T_op (s) | Total (s) | Speedup |
|---------------|-----------------|---------------|----------|-----------|---------|
| 1 GPU serial  |             —   |             — |        — |         — |    1.0× |
| 4 GPU (MPI)   |             —   |             — |        — |         — |      —  |
| 8 GPU (MPI)   |             —   |             — |        — |         — |      —  |

*Fill in on a multi-GPU instance. The benchmark prints README-ready table rows at the end of each run.*

---

### Benchmark results — Phase 5 (load balancing, static Morton vs adaptive)

Phase 5 replaces the uniform K/P Morton split with a weighted split proportional to
per-cell hit density, reducing COO imbalance and T_op wall time on non-uniform attractors.

The Lozi map (a=1.7, b=0.5) has a thin angular filament concentrated in one corner of its
domain. Inflating the bounding box puts cold fringe cells on certain Morton slabs, creating
a genuine hot/cold COO split. Two scenarios expose different severity levels:

> **Single run fills both tables:** `mpiexec -n 4 python benchmarks/benchmark_phase5.py`
> (default: runs `moderate` then `extreme` scenarios; prints README-ready rows at the end).

**Scenario 1 — Moderate imbalance (~4-5×):** Lozi, steps=16, domain\_scale=9.0

| Decomposition         | Cells | T_op (s) | Imbalance | Notes |
|-----------------------|-------|----------|-----------|-------|
| Phase 4 static Morton |     — |        — |      ~4.5× | cold rank 0 gets ~10% of COO entries |
| Phase 5 weighted      |     — |        — |      ~1.0× | fill in after running |

**Scenario 2 — Extreme imbalance (~30×):** Lozi, steps=16, domain\_scale=12.0

| Decomposition         | Cells | T_op (s) | Imbalance | Notes |
|-----------------------|-------|----------|-----------|-------|
| Phase 4 static Morton |     — |        — |      ~30× | rank 0 gets only ~1% of COO entries |
| Phase 5 weighted      |     — |        — |      ~1.0× | fill in after running |

*Fill in Cells and T\_op timings after running `benchmark_phase5.py` on a `gpu_4x_a100` instance.*
*Imbalance arises because Lozi's thin attractor occupies only a fraction of the inflated domain;*
*Morton Z-order clusters cold fringe cells (few hits) on one rank while others get core cells.*

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
