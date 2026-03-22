# Phase 5 Architecture Notes: Adaptive Load Balancing

## Overview

Phase 4 distributes the GAIO pipeline across MPI ranks via a static Morton
Z-order decomposition: domain keys are sorted by Morton code and split into
equal K/P chunks.  This works well for spatially uniform attractors but
breaks down for fractal structures like Hénon horseshoes or Ikeda spirals,
where the per-cell COO hit rate varies by 10–100× across the domain.  Fast
ranks sit idle at the `Allgatherv` barrier while the overloaded rank finishes.

Phase 5 eliminates this bottleneck with a **weighted prefix-sum split**: each
rank is assigned a contiguous Morton range whose total estimated work is
≈ `total_work / P`.  The estimate comes from the previous TransferOperator
call's per-cell hit counts, making Phase 5 a **predictive, frame-to-frame
load balancer**.

### Key properties

| Property | Value |
|----------|-------|
| Overhead for new `partition_weights` | One `Allgatherv` of K int32 values at T_op construction time |
| Communication cost at reuse | Zero — weights are local arrays on every rank |
| Correctness guarantee | Identical transfer matrix regardless of partition (same COO entries, different ordering) |
| Fallback when K changes | Silent revert to Phase 4 uniform split (`len(weights) != K`) |
| Single-frame use | `partition_weights=None` → identical to Phase 4, zero overhead |

---

## 1. Why TransferOperator, Not the Attractor

Phase 4's `distributed_relative_attractor` is **already balanced** by
construction: it re-Morton-sorts all surviving cells and re-splits K/P at
every subdivision step.  After each intersection step, the surviving cells
are re-distributed uniformly regardless of their spatial density.

The real imbalance appears only in `_build_transitions`:

```
For each cell j in rank r's local shard:
    generate M test points
    apply F → mapped points              ← equal work (K/P × M GPU evals)
    searchsorted into codomain._keys     ← equal work
    record matched (row, col, 1.0) hits  ← VARIABLE: depends on density
```

The GPU compute (`F._apply_map`) is equal per rank.  The COO output size
(`per_rank_nnz`) is not — a cell in the attractor core contributes all M
hits; a cell at the fractal boundary contributes a fraction.  Phase 5
balances the COO output by adjusting which cells each rank owns.

---

## 2. Weighted Prefix-Sum Decomposition

### Algorithm (`gaio/mpi/load_balance.py:weighted_local_keys`)

Given:
- `morton_sorted_keys` — K keys in Morton order (same as Phase 4)
- `weights[i]` — estimated COO contribution of key `morton_sorted_keys[i]`
  (equal to `hits_per_cell[i]` from the previous frame)

Find the contiguous key ranges `[starts[r], ends[r])` such that:

```
sum(weights[starts[r] : ends[r]]) ≈ sum(weights) / P   for all r
```

The algorithm:

```python
cumw       = cumsum(weights.astype(float64))        # prefix sum
boundaries = linspace(0, total, P+1)                # P+1 ideal split points
starts     = searchsorted(cumw, boundaries[:-1], 'left')
ends       = searchsorted(cumw, boundaries[1:],  'left')
ends[-1]   = K                                      # last rank gets tail
return morton_sorted_keys[starts[rank] : ends[rank]]
```

**Why `float64` for the cumsum?**  Summing K float32 values can accumulate
~K × ε₃₂ ≈ K × 1.2e-7 absolute error.  At K=1M cells with mean weight 27,
total ≈ 27M; rounding error ≈ 27M × 1.2e-7 ≈ 3.2 — large enough to
misplace a boundary.  Casting to float64 reduces error to ≈ 27M × 2.2e-16
≈ 6e-9, safely below 1 ULP of any boundary.

**Why contiguous Morton ranges?**  Morton order preserves spatial locality:
nearby cells in physical space have adjacent Morton codes.  A contiguous
range in Morton space is a spatially compact region, maintaining the O(K^{(d-1)/d})
boundary communication property from Phase 4.

### Fallback chain

1. `partition_weights is None` → Phase 4 `local_keys` (uniform K/P split)
2. `len(partition_weights) != K` → same Phase 4 fallback (domain changed)
3. `sum(weights) == 0` → same Phase 4 fallback (all cells have zero hits)

All fallbacks are silent (no warnings) — the caller can detect which path was
taken by comparing `per_rank_nnz` before and after.

---

## 3. Per-Cell Hit Counting

During Stage 3 of `_build_transitions`, each rank computes the number of
test-point hits per source cell (`hits_per_cell`) as a side effect of the
existing COO assembly:

```python
hits_per_cell = np.zeros(local_K, dtype=np.int32)   # one int per local cell

# After the matched mask is computed:
if matched.any():
    np.add.at(hits_per_cell, local_src[matched], 1)
```

`np.add.at` is used (not `+=`) because `local_src[matched]` can contain
repeated indices: a single source cell can map multiple test points to the
codomain, and each such hit must be counted.  `+=` with repeated indices uses
NumPy's buffered increment, which silently under-counts duplicates.

**Memory cost:** one `int32` per local cell = K/P × 4 bytes.  At K=1M cells
with P=4 ranks: 1 MB per rank — negligible.

**int32 vs int64:**  M (test points per cell) is at most ~5³ = 125 for typical
parameters.  An int32 overflows at 2.1B hits, requiring a cell to have ~16M
test points — physically impossible.  int32 halves the `Allgatherv` traffic
for weights vs int64.

---

## 4. Weight Gathering (`compute_partition_weights`)

After Stage 3, each rank has its local `hits_per_cell` array.  These are
gathered into a global array indexed by Morton position:

```
Step 1: gather_sizes(comm, local_K)
        → counts[r] = local_K for rank r
        → total = K (sum of all local_K)

Step 2: Allgatherv(hits_i32, [g_hits, (counts, displs)])
        → g_hits[starts[r]:ends[r]] = hits_per_cell from rank r

Step 3: return g_hits.astype(float32)
```

**Why one extra Allgatherv?**  The three Allgatherv calls in `allgather_coo`
(rows, cols, vals) transfer O(nnz × 20 bytes).  The weight Allgatherv
transfers O(K × 4 bytes) — for K=1M: 4 MB vs potentially hundreds of MB for
the COO.  The weight gather is cheap in absolute terms.

**Weight ordering invariant:**  Each rank's shard is a contiguous slice of
`morton_sorted_keys`.  After `Allgatherv`, position `i` in `g_hits`
corresponds to `morton_sorted_keys[i]`.  On the next call, `weighted_local_keys`
receives the same `morton_sorted_keys` (re-computed by the same `morton_sort_keys`
function on the same `domain._keys`) and the same `weights` — alignment is
guaranteed as long as `domain._keys` does not change between frames.

---

## 5. TransferOperator API Changes

### New constructor parameter

```python
TransferOperator(F, domain, codomain, comm=None, partition_weights=None)
```

`partition_weights` is a float32 array of shape `(K,)` in Morton order.
Passing `None` (default) is identical to Phase 4.

### New `partition_weights` property

```python
T.partition_weights  →  float32 ndarray, shape (K,), or None if K=0
```

Backed by `T.mpi_stats["partition_weights"]`.  Intended to be passed
directly to the next frame's `TransferOperator` call.

### New `mpi_stats` key

`mpi_stats["partition_weights"]` — float32 array, always present after
construction.  The per-rank version is `mpi_stats["per_rank_nnz"]` (Phase 4);
the per-cell version is `partition_weights` (Phase 5).

---

## 6. Nonautonomous Usage Pattern

The intended primary use case is time-varying (nonautonomous) systems where
`TransferOperator` is computed for multiple successive frames.  Frame 0 pays
the one-time cost of building the weight array; all subsequent frames use it:

```python
weights = None                             # frame 0: Phase 4 path
for t in range(n_frames):
    F_t = NonautonomousBoxMap(f, t, ...)   # time-varying map
    A   = relative_attractor(F_t, S, steps=steps, comm=comm)
    T   = TransferOperator(F_t, A, A, comm=comm, partition_weights=weights)
    weights = T.partition_weights          # adaptive from frame 1 onwards

    if should_rebalance(T.mpi_stats["per_rank_nnz"]):
        print(f"frame {t}: imbalance {compute_imbalance(T.mpi_stats['per_rank_nnz']):.2f}×")
```

**When is Phase 5 beneficial?**

Let I = imbalance ratio (Phase 4), P = number of ranks, W = total work,
S = GPU throughput.  Phase 4 wall time is bounded by the slowest rank:
`T₄ = I × W / (P × S)`.  Phase 5 frame 1+ wall time is `T₅ ≈ W / (P × S)`.
Speedup = T₄/T₅ ≈ I.

For nonautonomous systems with N frames:

```
Phase 4 total:  N × I × W / (P × S)
Phase 5 total:  (I + (N-1)) × W / (P × S)   ← frame 0 still pays I
Phase 5 wins:   N > I / (I - 1)              → for I=5: N > 1.25 (always wins for N≥2)
                                              → for I=2: N > 2 (wins from frame 3)
```

For single-frame use (N=1), Phase 5 cannot improve performance (frame 0 =
Phase 4, no second frame to benefit from).  Pass `partition_weights=None` for
single-frame use to avoid the extra Allgatherv.

---

## 7. Imbalance Diagnostics

Two helpers in `gaio.mpi.load_balance`:

```python
from gaio.mpi.load_balance import compute_imbalance, should_rebalance

T = TransferOperator(F, A, A, comm=comm)

# Diagnostic: how imbalanced was this frame?
imb = compute_imbalance(T.mpi_stats["per_rank_nnz"])
print(f"imbalance: {imb:.2f}×")  # 1.0 = perfect, ∞ = idle rank

# Gate: is it worth rebalancing next frame?
if should_rebalance(T.mpi_stats["per_rank_nnz"], threshold=2.0, min_total_nnz=500):
    weights = T.partition_weights   # will be used next frame
else:
    weights = None                  # stay with Phase 4 for near-uniform case
```

`should_rebalance` defaults: trigger if `imbalance > 2.0` and
`total_nnz_raw >= 500`.  The `min_total_nnz` guard prevents rebalancing for
trivially small problems where the extra Allgatherv would dominate.

---

## 8. New File

| File | Description |
|------|-------------|
| `gaio/mpi/load_balance.py` | Core Phase 5 logic: `compute_imbalance`, `should_rebalance`, `weighted_local_keys`, `compute_partition_weights` |

### Modified files

| File | Change |
|------|--------|
| `gaio/transfer/operator.py` | `_build_transitions`: `partition_weights` param, hit counting, weight gather; `TransferOperator.__init__`: `partition_weights` param; `partition_weights` property |
| `gaio/mpi/__init__.py` | Export Phase 5 symbols |
| `gaio/mpi/comm.py` | `get_comm()`: check MPI env-vars before importing mpi4py, preventing segfault on WSL |
| `gaio/__init__.py` | Version → `0.1.0-phase5` |

---

## 9. Benchmark

`benchmarks/benchmark_phase5.py` runs the Phase 4 vs Phase 5 comparison:

```bash
# Hénon horseshoe — extreme imbalance target (expected 5–15× speedup)
mpiexec -n 4 python benchmarks/benchmark_phase5.py --map henon --steps 12

# Ikeda spiral — moderate imbalance (expected 2–5× speedup)
mpiexec -n 4 python benchmarks/benchmark_phase5.py --map ikeda --steps 10

# Four-Wing ODE — near-uniform control (expected ~1.0× speedup, verifies no regression)
mpiexec -n 4 python benchmarks/benchmark_phase5.py --map fourwing --steps 8

# All three maps in sequence
mpiexec -n 4 python benchmarks/benchmark_phase5.py --all-maps --steps 12
```

For Lambda Cloud (large enough to saturate GPUs):
```bash
mpiexec -n 4 --bind-to socket \
    python benchmarks/benchmark_phase5.py \
    --map henon --steps 16 --grid-res 3 --test-pts 4 --gpu
```
