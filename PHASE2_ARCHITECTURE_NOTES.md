# PHASE 2 ARCHITECTURE NOTES
## GAIO.py — Algorithms and Operators
### Technical Engineering Design Document & Study Guide

---

## 1. The Mathematical Foundation

### 1.1 BoxMap: Outer-Approximation of a Dynamical System

A **BoxMap** `F: 𝒫 → 𝒫` (where `𝒫` is the set of all BoxSets on a partition) is a combinatorial approximation of a continuous map `f: ℝⁿ → ℝⁿ`. The construction is:

For each source cell `B_j`:
1. Sample M test points `x₁, ..., x_M` from `B_j`
2. Compute images `f(x₁), ..., f(x_M)`
3. Add cell `B_i` to the image if any `f(xₖ) ∈ B_i`

This gives the **outer approximation**:
```
F(S) = ⋃_{j: B_j ∈ S} {B_i : ∃ xₖ ∈ B_j, f(xₖ) ∈ B_i}
```

The key theorem: the true image `f(S)` is always a subset of `F(S)`. No cell containing a true image point is ever missed. Error shrinks as `M → ∞` (more test points) or as the grid is refined (smaller cells).

### 1.2 RK4: Classical 4th-Order Runge-Kutta Integration

For an ODE `ẋ = g(x)`, the classical RK4 step advances the state by time `τ`:

```
k₁ = g(xₙ)
k₂ = g(xₙ + τ/2 · k₁)
k₃ = g(xₙ + τ/2 · k₂)
k₄ = g(xₙ + τ · k₃)

x_{n+1} = xₙ + (τ/6)(k₁ + 2k₂ + 2k₃ + k₄)
```

This uses the **Butcher tableau** with nodes `(0, 1/2, 1/2, 1)` and weights `(1/6, 1/3, 1/3, 1/6)`. The local truncation error is O(τ⁵); the **global error** over a fixed integration time T = N·τ is O(τ⁴). In practice:

- For `τ = 0.01`, `steps = 100` (T = 1.0 s): global error ≈ 1e-7
- For `τ = 0.001`, `steps = 6283` (T ≈ 2π, one SHO period): global error ≈ 5e-4

`rk4_flow_map(g, step_size=τ, steps=N)` produces the **time-T flow map** `Φᵀ: x ↦ x(T)`, which is then wrapped in a `SampledBoxMap` for use in all algorithms. The flow map is the standard way to apply GAIO to ODEs — the ODE is converted to a discrete-time map by integrating forward by one period (or unit time).

### 1.3 The Perron-Frobenius (Transfer) Operator

The **Perron-Frobenius operator** `ℒ: L¹(ℝⁿ) → L¹(ℝⁿ)` governs the evolution of probability densities under `f`:

```
(ℒρ)(y) = ∑_{x: f(x)=y} ρ(x) / |det Jf(x)|
```

Intuitively: mass at `x` flows to `f(x)`. The fixed points of `ℒ` are **invariant measures** — probability densities that do not change under the dynamics.

#### 1.3.1 Discretisation as a Markov Transition Matrix

When restricted to a BoxPartition, `ℒ` becomes a finite-dimensional matrix `T ∈ ℝ^{m×n}` where:

```
T[i, j] = (number of test points in B_j whose image lands in B_i) / M
```

Column `j` of `T` is a probability vector (sums to 1, or 0 if all test points leave the domain) — this is exactly a **column-stochastic matrix**, also known as a **Markov transition matrix**. Specifically:

- **Columns** represent **source** cells (where mass comes from)
- **Rows** represent **target** cells (where mass goes to)
- `T[i,j] > 0` means there is a transition from cell `j` to cell `i`
- Column stochasticity: `∑ᵢ T[i,j] = 1` (all mass is accounted for)

This is the **Monte-Carlo approximation of the Frobenius-Perron operator** with respect to the uniform (Lebesgue) measure on each cell.

#### 1.3.2 Invariant Measures as Fixed Points

An invariant measure `μ` satisfies `T @ μ = μ` — it is an **eigenvector of T with eigenvalue 1**. The **Perron-Frobenius theorem** guarantees that for an irreducible, aperiodic Markov chain, this eigenvector is unique (up to scaling) and has non-negative entries. The GAIO `TransferOperator.eigs(k=1)` call finds this leading eigenvector.

For general (non-ergodic) systems with multiple invariant components, `eigs(k)` returns `k` eigenvectors, each corresponding to an invariant measure supported on a different ergodic component. This is a quantitative spectral approach to finding **multiple invariant measures simultaneously**.

#### 1.3.3 The Koopman Operator (Pull-Back)

The adjoint of the Perron-Frobenius operator is the **Koopman operator** `𝒦`:

```
(𝒦φ)(x) = φ(f(x))   — observable φ evolved forward
```

In matrix form, `𝒦 ≈ Tᵀ` (the transpose of the transfer matrix). `TransferOperator.pull_back(μ)` computes `Tᵀ @ μ`, which corresponds to computing the expectation of an observable `φ` at the *next* time step.

### 1.4 BoxGraph and Strongly-Connected Components

The **transfer graph** of `T` is the directed graph `G = (V, E)` where:

- `V` = cells in the domain
- `(i → j) ∈ E` iff `T[j, i] > 0` (cell `i` sends mass to cell `j`)

The adjacency matrix is `A = Tᵀ` (transposed, since `T` is column = source, row = target).

A **strongly-connected component (SCC)** of `G` is a maximal set of nodes `C ⊆ V` such that every pair `(u, v) ∈ C × C` is mutually reachable: there exists a directed path from `u` to `v` and from `v` to `u`. In dynamical terms, an SCC represents a set of cells from which the dynamics can cyclically revisit every cell in the component.

**Tarjan's algorithm** (or Kosaraju's, as implemented in `scipy.sparse.csgraph.connected_components`) finds all SCCs in O(|V| + |E|) time.

A **non-trivial SCC** is one that is:
- Size > 1 (two or more cells in a cycle), OR
- Size 1 with a self-loop (the cell maps back to itself)

Non-trivial SCCs are the candidates for long-term recurrent behavior.

### 1.5 Morse Decomposition

A **Morse decomposition** partitions the chain-recurrent set of a dynamical system into **Morse sets** — isolated, recurrent pieces that are ordered by a gradient-like structure (no trajectories from a "lower" Morse set reach a "higher" one).

In the GAIO discretisation:
- **Morse sets** = non-trivial SCCs of the transfer graph on the current partition
- **Chain-recurrent set** = union of all Morse sets

The function `morse_sets(F, B)` computes this union. The function `morse_tiles(F, B)` assigns each cell a **1-indexed label** identifying which Morse component it belongs to — this labeling is the discrete analogue of the Morse decomposition and can be used directly for visualization.

The iterative algorithm `recurrent_set(F, B₀, steps=12)` refines this by alternating subdivision and Morse-set extraction:

```
for _ in range(steps):
    B = subdivide(B, argmin(dims))   # finer resolution
    B = morse_sets(F, B)             # extract non-trivial SCCs
```

Each subdivision step doubles the grid resolution along one axis, progressively eliminating cells that are not recurrent. The result converges to an outer approximation of the chain-recurrent set.

### 1.6 Invariant Set Algorithms

**Relative attractor** (`relative_attractor = ω`): outer approximation of the ω-limit set (forward attractor):
```
for _ in range(steps):
    B = subdivide(B, argmin(dims))
    B = B ∩ F(B)                      # keep only cells with image in B
```

**Alpha limit set** (`alpha_limit_set = α = maximal_forward_invariant_set`): outer approximation of the α-limit set (backward attractor):
```
for _ in range(steps):
    B = subdivide(B, argmin(dims))
    B = B ∩ F⁻¹(B)                    # keep only cells with preimage in B
```

**Maximal invariant set**: cells that are both forward- and backward-invariant:
```
for _ in range(steps):
    B = subdivide(B, argmin(dims))
    B = B ∩ F(B) ∩ F⁻¹(B)
```

**Unstable set** (BFS flood-fill from a seed):
```
W₀ = seed;  W₁ = seed
while W₁ ≠ ∅:
    W₁ = F(W₁) \ W₀        # new cells reached by the forward image
    W₀ = W₀ ∪ W₁           # accumulate
```
At termination: `F(W₀) ⊆ W₀` (W₀ is forward-invariant within the domain).

**Preimage** `F⁻¹(B) ∩ Q`: cells in Q whose forward image intersects B. Implemented by building `TransferOperator(F, Q, B)` and finding all source columns with at least one nonzero entry.

---

## 2. The Paradigm Shift: Julia to Python

### 2.1 Julia's BoxMap Type Hierarchy

In GAIO.jl, all BoxMap variants share a common abstract type:

```julia
abstract type BoxMap{N,T,F} end

struct SampledBoxMap{N,T,F,D,I} <: BoxMap{N,T,F}
    domain_points::D    # closure: cell → test points
    image_points::I     # closure: image points → lookup points
    map::F
end

struct GridBoxMap{N,T,F} <: BoxMap{N,T,F}
const MonteCarloBoxMap = SampledBoxMap  # aliased, different constructor
```

Julia's multiple dispatch allows `map_boxes(F::GridBoxMap, ...)` and `map_boxes(F::SampledBoxMap, ...)` to be distinct methods with zero runtime cost — the method is selected at compile time based on the concrete type of `F`.

### 2.2 Python's Factory Function Pattern

Python has no zero-cost runtime dispatch on concrete types. The translation strategy:

**One concrete class, multiple constructors:**

```python
class SampledBoxMap:           # the only BoxMap implementation
    unit_points: ndarray       # fixed test points in [-1,1]^n

def GridMap(f, domain, n_points) -> SampledBoxMap:      # factory function
def MonteCarloMap(f, domain, n_points) -> SampledBoxMap: # factory function
def rk4_flow_map(g, step_size, steps) -> callable:       # returns a plain function
```

`GridMap` and `MonteCarloMap` are **not subclasses** of `SampledBoxMap` — they are pure functions that construct a `SampledBoxMap` with a particular `unit_points` array. This means:

- The `map_boxes` hot loop exists in exactly **one place** (no method dispatch overhead)
- All BoxMap variants share identical memory layout (same `__slots__`)
- The CUDA kernel in Phase 3 needs to be written only once

The trade-off: less extensibility (can't subclass to specialize `map_boxes`), but Phase 3 requires that `map_boxes` compiles to a single CUDA kernel — a subclass hierarchy would require dynamic dispatch inside the kernel, which CUDA does not support.

### 2.3 Julia's PointDiscretizedBoxMap

Julia's `PointDiscretizedBoxMap` uses two closures: `domain_points` (test points rescaled into each cell) and `image_points` (how to look up the hit cell from the image). In Python's `SampledBoxMap`, these closures are inlined:

```python
# Julia: domain_points(box) = rescale(unit_pts, box)  →  image_points = center
# Python (inlined in map_boxes):
test_pts = centers[:, np.newaxis, :] + unit_pts[np.newaxis, :, :] * cell_r
```

The `image_points = center` semantics (Julia: the hit cell is determined by the center of the image box, not by the raw image point) is implemented identically in Python via `point_to_key_batch` — passing the raw image point and looking up which cell it belongs to is exactly equivalent.

### 2.4 Julia's Algorithms Module

In GAIO.jl, invariant-set algorithms are written as generic functions dispatching on `BoxMap` and `BoxSet`. The Julia code:

```julia
function ω(F, B::BoxSet; steps=12)
    iter = (S -> S ∩ F(S)) ∘ subdivide
    iterate_until_equal(iter, B; max_iterations=steps)
end
```

The `iterate_until_equal` helper stops early if the state doesn't change. But since `subdivide` changes the partition every step, the state *never* equals the old state — all `steps` iterations always run. Python translates this directly as a `for _ in range(steps)` loop with no early-exit logic, which is both simpler and correct.

### 2.5 TransferOperator: OrderedDict → Sparse COO

Julia's `TransferOperator` stores the matrix in a custom sparse format backed by `OrderedDict`. Python uses `scipy.sparse.csc_matrix` in **Compressed Sparse Column (CSC)** format, constructed via **COO (Coordinate) format**:

1. Build `(row_idx, col_idx, value)` triplets (COO) — each triplet = one test-point hit
2. `scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(m, n))` sums duplicate `(i,j)` entries automatically
3. Convert to CSC via `.tocsc()` for efficient column operations (column-stochastic normalisation, matrix-vector products)

**Why CSC over CSR?** The TransferOperator is fundamentally column-indexed (domain = columns, codomain = rows). Column slicing in CSC is O(nnz_per_column); in CSR it would be O(nnz_per_row) for column access. The dominant operations — column normalisation and `mat @ vector` (push-forward) — are both O(nnz) in CSC.

---

## 3. Specific Architectural Choices

### 3.1 Separation of Map Application and Key Lookup

In `SampledBoxMap.map_boxes` ([gaio/maps/base.py:122](gaio/maps/base.py#L122)), the computation is split into three explicit stages:

```python
# Stage 1: generate test points — pure numpy broadcasting, no Python loop
test_pts = (centers[:, np.newaxis, :] + unit_pts[np.newaxis, :, :] * cell_r).reshape(K*M, n)

# Stage 2: apply map — Python loop, the CUDA target
for i, p in enumerate(test_pts):
    mapped[i] = f(p)

# Stage 3: key lookup — pure numpy, vectorized
hit_keys = partition.point_to_key_batch(mapped)
```

This separation is deliberate: Stage 2 is the **only non-vectorized, non-CUDA-ready part** of the library. In Phase 3, Stages 1 and 3 remain unchanged; Stage 2 is replaced by a CUDA kernel. The array shapes (`(K*M, n)`) were chosen specifically for this future replacement.

### 3.2 COO Accumulation Pattern for TransferOperator

In `_build_transitions` ([gaio/transfer/operator.py:53](gaio/transfer/operator.py#L53)), transitions are accumulated as append-lists then converted to NumPy arrays:

```python
rows_list = []
cols_list = []
# ... inner loop appends to lists ...
rows = np.array(rows_list, dtype=I64)
cols = np.array(cols_list, dtype=I64)
vals = np.ones(len(rows), dtype=F64)
```

This is the standard Python pattern for COO matrix construction when the final size is unknown. The resulting `(rows, cols, vals)` triplets are **MPI-distributable**: each rank computes a local subset of `(rows, cols, vals)` and the root rank assembles the global matrix via:

```python
# Phase 4 pseudocode:
local_coo = _build_transitions(F, local_domain_slice, codomain)
all_rows = comm.gather(local_rows, root=0)
all_cols = comm.gather(local_cols, root=0)
# ... assemble on root ...
```

The independence of each column `j` in `_build_transitions` (each source cell is processed independently) is the key property that makes MPI distribution trivially correct.

### 3.3 Adjacency Matrix Convention in BoxGraph

The `TransferOperator.mat` has shape `(|codomain|, |domain|)` with column = source, row = target. The **adjacency matrix** of the transfer graph uses the transposed convention: `A[i, j] > 0` means there is an edge from node `i` to node `j` (box `i` maps to box `j`):

```python
adj = self.T.mat.T.tocsr()   # transpose: row = source, col = target
```

This convention matches `scipy.sparse.csgraph.connected_components`, which expects adjacency matrices in `A[source, target]` format. Getting this transposition wrong would cause SCC computation to find components of the **reverse graph** — a subtle correctness issue that would not produce obvious errors in small test cases.

### 3.4 RK4 Step Function Design for Numba Compatibility

The `rk4_step` function ([gaio/maps/rk4.py:47](gaio/maps/rk4.py#L47)) is written using only:
- Element-wise scalar multiplication by `float` (`tau * k1`)
- Element-wise addition of 1D `float64` arrays
- Calls to `f(x)` (the ODE RHS)

No Python control flow, no list comprehensions, no dict lookups. Every operation is an element-wise array op on `float64` vectors. To convert this to Numba:

```python
@numba.njit
def rk4_step_jit(f_jit, x, tau):   # f_jit must also be @njit
    ...                              # body is identical
```

For CUDA, the conversion is:

```python
@cuda.jit
def rk4_batch_kernel(f_device, x_batch, tau, steps, out):
    tid = cuda.grid(1)
    if tid < x_batch.shape[0]:
        state = x_batch[tid]                # one thread per initial condition
        for _ in range(steps):
            state = rk4_step_device(f_device, state, tau)
        out[tid] = state
```

The inner `rk4_step` logic remains unchanged. Phase 3 wraps it in a CUDA kernel rather than rewriting it.

### 3.5 Sparse Matrix Storage for Transfer Operator

The final `TransferOperator.mat` is stored as **CSC** (Compressed Sparse Column):

- `mat.indptr`: shape `(n+1,)` — column pointer array; `indptr[j]:indptr[j+1]` are the row indices of column `j`
- `mat.indices`: shape `(nnz,)` — row indices of nonzero entries
- `mat.data`: shape `(nnz,)` `float64` — values

For a typical 2D grid with `dims = (64, 64)` = 4096 cells and 16 test points per cell, `nnz ≤ 4096 * 16 = 65536` (but significantly less after duplicate removal). The density is `nnz / (n²) ≈ 65536 / 16.7M ≈ 0.4%` — sparse enough that sparse storage is essential.

**For Phase 4 (MPI)**: the global matrix is too large to assemble on one rank for large grids. The `mpi4py` integration will use `PETSc.Mat` (via `petsc4py`) or a custom distributed CSR where each rank owns a contiguous block of rows (codomain cells). The COO construction pattern already produces the data in a form compatible with PETSc's `MatSetValues`.

### 3.6 BoxMeasure as a Dense Vector Interface

The `TransferOperator` stores `mat` as a sparse matrix indexed by **position** (column j = j-th cell of domain, not the cell's flat key). The `BoxMeasure` ↔ dense vector conversion methods `_domain_vec` and `_codomain_vec` use `np.searchsorted` to map between:

- Key-indexed space (measure's `_keys` array, sorted flat keys)
- Position-indexed space (matrix column/row indices, 0..n-1)

This two-space design is necessary because: (1) the BoxSet may not be contiguous in key space (e.g., after intersection, only some cells remain active), and (2) scipy sparse matrices are indexed by position, not by key. The `searchsorted` bridge between the two spaces costs O(N log N) and is called twice per push_forward — acceptable overhead for the matrix-vector product that follows.

---

## 4. Known Bottlenecks

### 4.1 `SampledBoxMap.map_boxes` — Python Loop (PRIMARY CUDA TARGET)

**Location**: [gaio/maps/base.py:166](gaio/maps/base.py#L166)

```python
for i, p in enumerate(test_pts):      # K*M iterations, pure Python
    mapped[i] = np.asarray(self.map(p), dtype=F64)
```

For `K = 4096` cells and `M = 16` test points: `K*M = 65536` Python function calls per `map_boxes` invocation. For `recurrent_set` with `steps=12`, `map_boxes` is called inside `morse_sets` inside the loop — potentially `12 × (4096 + overhead)` calls. This is the single most expensive operation in the entire library.

**Phase 3 replacement**:
```python
# All K*M test points are already in test_pts: shape (K*M, n), C-contiguous float64
# Kernel signature:
@cuda.jit
def map_boxes_kernel(test_pts, mapped_out, ...):
    tid = cuda.grid(1)          # one thread per test point
    if tid < test_pts.shape[0]:
        mapped_out[tid] = f_device(test_pts[tid])
```

The `test_pts` array is already the right shape and dtype. Phase 3 requires: (a) `f` compiled with `@cuda.jit(device=True)`, (b) transfer of `test_pts` to device, (c) kernel launch over `K*M` threads.

### 4.2 `_build_transitions` — Double Python Loop (PRIMARY MPI TARGET)

**Location**: [gaio/transfer/operator.py:99](gaio/transfer/operator.py#L99)

```python
for j, src_key in enumerate(domain._keys):    # N iterations, Python
    ...
    for i, p in enumerate(test_pts):           # M iterations, Python
        mapped[i] = np.asarray(F.map(p), dtype=F64)
    for hk in hit_keys:                        # M iterations, Python
        ...
```

For a `TransferOperator` over N cells with M test points: N×M Python function calls plus N×M list appends. For `N = 4096, M = 16`: ~66000 iterations in the outer structure, each containing Python-level function calls.

**Phase 4 MPI decomposition**:
```python
# Each rank processes a slice of domain._keys
local_keys = np.array_split(domain._keys, comm.size)[comm.rank]
local_rows, local_cols, local_vals = _build_transitions(F, local_domain_slice, codomain)
all_data = comm.gather((local_rows, local_cols, local_vals), root=0)
# Root assembles global matrix
```

This is embarrassingly parallel: each source cell is independent of all others.

### 4.3 `morse_sets` / `recurrent_set` — Sequential Subdivision Loop

**Location**: [gaio/algorithms/morse.py:161](gaio/algorithms/morse.py#L161)

`recurrent_set` calls `morse_sets` once per subdivision step. Each `morse_sets` call:
1. Builds a `TransferOperator` (bottleneck 4.2 above)
2. Runs `connected_components` on the adjacency matrix

For `steps=12` starting from a `[1,1]` partition, the grid grows to `[64, 64]` = 4096 cells by step 12 (doubling one axis per step). The 12th `TransferOperator` call dominates total runtime. For `steps=20`, the final grid reaches `[1024, 1024]` = 1M cells — at which point even the `scipy` SCC computation becomes the bottleneck.

**Phase 3 + 4 combined target**: parallelize the `TransferOperator` build across GPU + MPI for the largest steps, where the grid is coarsest for the initial steps (GPU warm-up is not cost-effective) and finest at the end (where parallelism pays off).

### 4.4 `strongly_connected_components` — `scipy.sparse.csgraph`

**Location**: [gaio/graph/boxgraph.py:98](gaio/graph/boxgraph.py#L98)

`scipy.sparse.csgraph.connected_components` implements a single-threaded O(|V| + |E|) DFS-based SCC algorithm (Tarjan's). For the transfer graph with N cells and up to N×M edges: O(N + N×M) = O(N×M). For N = 65536 cells, M = 16: ~1M edge traversals.

This is currently not parallelizable without switching to a parallel SCC algorithm (e.g., the parallel Tarjan variant using CUDA). For Phase 3, this step may need to remain on the CPU while the `TransferOperator` build moves to the GPU — the SCC step has irregular memory access patterns that are poorly suited to SIMT execution.

### 4.5 `TransferOperator.eigs` — ARPACK on the Full Matrix

**Location**: [gaio/transfer/operator.py:286](gaio/transfer/operator.py#L286)

`scipy.sparse.linalg.eigs` calls ARPACK, which performs an Arnoldi iteration. Each Arnoldi step requires one matrix-vector product `mat @ v`. For a `4096×4096` sparse matrix with ~65K nonzeros, each product is fast. But for large grids (`1024×1024` = 1M cells, `nnz ≈ 16M`), each Arnoldi step involves a 16M-flop sparse matrix-vector multiply.

**Phase 3**: replace with `cuSPARSE` sparse matrix-vector multiply inside a power iteration or Arnoldi iteration implemented in CUDA. The `cuSPARSE` SpMV achieves near-memory-bandwidth performance on GPU.

### 4.6 `BoxMeasure._binary_op` — Python Loop Over Union of Keys

**Location**: [gaio/core/boxmeasure.py:179](gaio/core/boxmeasure.py#L179)

```python
w1 = np.array([self[int(k)] for k in all_keys], dtype=F64)   # Python loop
w2 = np.array([other[int(k)] for k in all_keys], dtype=F64)  # Python loop
```

Each `self[k]` call does a `searchsorted` — O(log N) binary search. Total cost: O(N log N) binary searches executed in a Python loop. For N = 65536: ~65536 × 17 ≈ 1.1M loop iterations.

**Phase 3 fix** (even before CUDA): replace with fully vectorized NumPy:
```python
# Since both _keys arrays are sorted, this can be done with np.searchsorted + masking
# without any Python-level loop — O(N) numpy operations only.
```
This is a pre-Phase-3 optimization that requires no GPU.
