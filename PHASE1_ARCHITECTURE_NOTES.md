# PHASE 1 ARCHITECTURE NOTES
## GAIO.py — Foundational Data Structures
### Technical Engineering Design Document & Study Guide

---

## 1. The Mathematical Foundation

### 1.1 Set-Oriented Numerics

GAIO is built on the theory of **set-oriented numerical methods** (Dellnitz & Junge, 1997). The central idea is to replace point-wise analysis of a dynamical system `f: ℝⁿ → ℝⁿ` with a combinatorial analysis over a finite covering of the phase space by axis-aligned boxes.

A **box** (axis-aligned hyperrectangle) is defined as the half-open interval:

```
B = [c - r, c + r)  ⊂ ℝⁿ
```

where `c` is the center and `r` is the radius vector. The half-open convention is essential: it ensures boxes in a grid partition are **disjoint** and their union is the full domain with no overlaps at shared boundaries.

### 1.2 The BoxPartition as a Cartesian Grid

A **BoxPartition** divides a fixed domain `D ⊂ ℝⁿ` into `∏ dims[i]` equal cells arranged in a uniform Cartesian grid. The cell geometry is:

```
cell_width[i]  = 2 * domain.radius[i] / dims[i]
cell_radius[i] = domain.radius[i] / dims[i]
```

For a cell with multi-index `m = (m₀, m₁, ..., mₙ₋₁)`:

```
center[i] = domain.lo[i] + cell_width[i] * (m[i] + 0.5)
```

This is a uniform mesh — all cells share the same `cell_radius`. This uniformity is a deliberate constraint (not present in GAIO.jl's adaptive variants) that enables the flat-key encoding described below.

### 1.3 The BoxSet as an Outer Approximation

A **BoxSet** is a finite union of grid cells:

```
S = ⋃_{k ∈ K} B_k,    K ⊂ {0, 1, ..., ∏dims - 1}
```

All algorithms in GAIO compute **outer approximations**: the true mathematical set is guaranteed to be a subset of the returned BoxSet. No cells are ever missed — but spurious cells may be included. This is the fundamental accuracy contract of the library. Finer grids (more subdivision steps) shrink the error, producing tighter outer approximations.

### 1.4 Subdivision and Adaptive Refinement

The key convergence mechanism is **subdivision**: replacing each cell in a BoxSet with 2 children along a chosen axis. After `k` subdivisions along axis `d`, the resolution along that axis doubles `k` times. The Julia library's standard subdivision strategy always subdivides along `argmin(dims)` to keep the grid as close to square as possible — a convention inherited exactly in our Python port.

### 1.5 BoxMeasure as a Discrete Probability Measure

A **BoxMeasure** represents a discrete signed measure on the partition:

```
μ = ∑_{k ∈ K} w_k · δ_{B_k}
```

where `w_k ∈ ℝ` is the weight (mass) assigned to cell `B_k`. When used as a probability measure, all weights are non-negative and sum to 1. The **density function** of `μ` with respect to Lebesgue measure is:

```
dμ/dx = w_k / vol(B_k)   for x ∈ B_k
```

This connects to the Perron-Frobenius operator formalism used in Phase 2.

---

## 2. The Paradigm Shift: Julia to Python

### 2.1 Julia's Multiple-Dispatch Architecture

In GAIO.jl, `Box`, `BoxPartition`, `BoxSet`, and `BoxMeasure` are all **parameterized types** in Julia's type system:

```julia
struct Box{N, T<:Number}     # N = spatial dim, T = numeric type
struct BoxPartition{N, T}    # parameterized over Box type
struct BoxSet{B<:AbstractBox, K<:Integer, ...}
struct BoxMeasure{B, K, V, P, D}
```

Julia's multiple dispatch allows functions like `subdivide(B::BoxSet)` and `subdivide(B::BoxSet, dim::Int)` to coexist as distinct methods selected at compile time based on argument types, with **zero runtime overhead**.

### 2.2 Python's OOP Translation

Python has no parameterized types at the value level — type parameters like `{N, T}` are erased at runtime. Our translation:

| Julia mechanism | Python translation |
|---|---|
| `struct Box{N,T}` | `class Box` with `__slots__` and runtime `dtype=float64` enforcement |
| Type parameter `N` | `self.center.shape[0]` (runtime property `.ndim`) |
| Type parameter `T` | Hardcoded `dtype=np.float64` — see §3 |
| Multiple dispatch | Regular methods + factory functions (e.g., `BoxSet.full`, `BoxSet.cover`) |
| Immutable struct | `__slots__` + no public setters; operations return new objects |
| `subdivide(B)` / `subdivide(B, dim)` | `BoxSet.subdivide(dim)` — dim is always explicit in Python |

The key design decision is that **Python's OOP is used structurally, not as a dispatch mechanism**. The `Box`, `BoxPartition`, `BoxSet`, and `BoxMeasure` classes are effectively value-types: immutable after construction (with the exception of `BoxMeasure.__setitem__` for in-place weight updates), and all operations return new objects.

### 2.3 Operator Overloading for Mathematical Readability

To preserve the mathematical readability of Julia code like `B ∩ F(B)`, all set-algebra operations are overloaded:

```python
B & F(B)    # intersection:        BoxSet.__and__
B | F(B)    # union:               BoxSet.__or__
B - F(B)    # difference:          BoxSet.__sub__
```

This allows algorithm code (Phase 2) to read almost identically to the Julia source — a deliberate choice to make correctness audits against the reference implementation tractable.

---

## 3. Specific Architectural Choices

### 3.1 Flat `int64` Keys (The Core Decision)

Every grid cell is identified by a single **flat integer key**:

```python
key = np.ravel_multi_index(multi_index, dims)   # C / row-major order
```

The inverse:
```python
multi_index = np.unravel_index(key, dims)
```

This is the single most important architectural decision in Phase 1. **Why not tuple keys or dict-of-tuples (as Julia uses internally)?**

1. **Numba/CUDA compatibility**: `@cuda.jit` kernels operate on device arrays. CUDA device arrays must be contiguous numeric buffers — no Python objects, no tuples, no dicts. A flat `int64` array passes directly into device memory via `cuda.to_device(boxset._keys)`.

2. **MPI send/recv**: `mpi4py` communicates NumPy arrays directly as byte buffers. A 1D `int64` array is a single `MPI_Send(keys, N, MPI_INT64_T, ...)` call. Tuple keys would require serialization (pickle), destroying performance at scale.

3. **Vectorized set operations**: `np.union1d`, `np.intersect1d`, `np.setdiff1d` on sorted `int64` arrays run in O(n log n) in C — orders of magnitude faster than Python-level set operations on tuples.

4. **Cache locality**: A contiguous sorted `int64` array has excellent spatial locality. The `O(log n)` membership test `np.searchsorted` on a sorted array is cache-friendly in a way that hash-table lookups are not.

### 3.2 `float64` Everywhere

All spatial data (`Box.center`, `Box.radius`, `BoxPartition.cell_radius`, `BoxSet.centers()`, `BoxMeasure._weights`) is stored and computed in **`np.float64`** (IEEE 754 double precision). This is enforced at construction time via `np.ascontiguousarray(..., dtype=np.float64)`.

The reason is Phase 3: CUDA `@cuda.jit` kernels require all array arguments to have known, fixed dtypes. Mixed-precision code introduces both correctness risks (silent promotion/truncation) and performance penalties (dtype negotiation at kernel launch). By enforcing `float64` at the boundary (construction), the entire downstream computation is uniformly typed and CUDA-ready without any casting.

### 3.3 C-Contiguous Memory Layout

All arrays are enforced as **C-contiguous** (row-major, the NumPy default) via `np.ascontiguousarray(...)`. The specific shape conventions are:

| Array | Shape | Meaning |
|---|---|---|
| `Box.center` | `(n,)` | n-dimensional point |
| `Box.radius` | `(n,)` | per-axis half-widths |
| `BoxPartition.dims` | `(n,)` `int64` | grid resolution per axis |
| `BoxSet._keys` | `(N,)` `int64` | sorted flat cell keys |
| `BoxSet.centers()` | `(N, n)` `float64` | row = one cell center |
| `BoxMeasure._weights` | `(N,)` `float64` | parallel to `_keys` |

The `(N, n)` shape for batches of points (N points in n-dimensional space) is CUDA-natural: each CUDA thread handles one row (one point), and coalesced memory access is achieved when threads in a warp access consecutive rows.

### 3.4 Sorted, Unique Key Invariant

`BoxSet._keys` and `BoxMeasure._keys` are always **sorted and unique** — enforced by `np.unique()` on construction. This invariant enables:

- O(log N) membership test: `np.searchsorted(keys, k)`
- O(N log N) set operations that return sorted results (inputs to `np.union1d` etc. are pre-sorted, so the merge step is O(N))
- Deterministic ordering for MPI rank-to-rank array comparisons
- Direct indexing by position in `TransferOperator` column/row extraction (Phase 2)

### 3.5 Immutability and Thread Safety

All `Box`, `BoxPartition`, and `BoxSet` objects are effectively immutable after construction (enforced by `__slots__` and the absence of mutation methods — subdivision always returns a *new* object). This design:

- Makes Phase 4 MPI-safe: multiple ranks can hold read-only references to the same partition object without locking
- Simplifies Phase 3 GPU code: partition metadata (`dims`, `cell_radius`, `domain.lo`) can be passed to CUDA kernels as constant arguments without synchronization

---

## 4. Known Bottlenecks in Phase 1

### 4.1 `BoxSet.repartition` — O(N·M) Naive Scan

[gaio/core/boxset.py:319](gaio/core/boxset.py#L319)

`repartition` transfers a BoxSet to a different-resolution partition by checking, for each new cell, whether its center falls inside any old cell. The current implementation is an explicit Python loop over all `M` new cells, with a vectorized check over all `N` old cells. This is O(N·M) and runs entirely in Python-level NumPy, which is slow for large N or M.

**Phase 3 target**: rewrite as a CUDA kernel that parallelizes over the M new cells.

### 4.2 `BoxPartition.keys_in_box` — Range Computation

[gaio/core/partition.py:263](gaio/core/partition.py#L263)

`keys_in_box` computes the multi-index bounding box of a query, then builds a cartesian product of ranges via `np.meshgrid`. For high-dimensional grids, the meshgrid can produce very large intermediate arrays. This is called inside `BoxSet.from_box` and `BoxSet.cover`.

**Phase 3 note**: for `d > 3`, the meshgrid approach becomes memory-bound. A CUDA kernel that parallelizes over all `prod(dims)` cells and tests containment in parallel avoids constructing the intermediate meshgrid.

### 4.3 `BoxMeasure._binary_op` — Python-Level Key Alignment

[gaio/core/boxmeasure.py:179](gaio/core/boxmeasure.py#L179)

The `_binary_op` method aligns two measures over their union of keys using a Python list comprehension `[self[int(k)] for k in all_keys]`. This is O(N log N) membership tests but executed in a Python loop. For large measures (N > 10⁶), this becomes a bottleneck.

**Phase 4 target**: for distributed measures, each MPI rank holds a contiguous slice of `_keys` and `_weights`. Binary ops reduce to local array arithmetic followed by a single `MPI_Allreduce`.

### 4.4 `BoxMeasure.integrate` — Python Function Call per Cell

[gaio/core/boxmeasure.py:255](gaio/core/boxmeasure.py#L255)

`integrate` calls a Python function `f` once per cell center. For N cells, this is N Python function calls. This is a known Python overhead pattern.

**Phase 3 target**: if `f` is decorated with `@numba.njit`, the loop can be replaced with a Numba-parallelized reduction over the `(N, n)` centers array.
