# PHASE 3 ARCHITECTURE NOTES
## GAIO.py — Heterogeneous Acceleration (CPU & GPU)
### Technical Engineering Design Document & Study Guide

---

## 1. Why a Dual-Backend Architecture?

### 1.1 The Problem With Single-Target Optimization

The Phase 2 bottleneck — the Python loop inside `SampledBoxMap.map_boxes` — could be addressed in several ways: vectorizing `f` with NumPy broadcasting, using `multiprocessing`, or replacing it with a single Numba decorator. Each of those approaches solves the problem for exactly one deployment scenario and leaves the others unserved.

GAIO is intended to run across an enormous range of hardware:
- A researcher's laptop with 8 CPU cores and no discrete GPU
- A department workstation with a single NVIDIA RTX GPU
- A national lab HPC node with 4× A100 GPUs and 256 CPU cores

A single-target solution degrades to the Python baseline on any hardware that doesn't match. A dual-backend architecture guarantees that **every machine gets the fastest code it can actually use**, without requiring the user to maintain two separate codebases.

### 1.2 The Specific Backends and Their Roles

| Backend | Hardware target | Mechanism | Speedup (empirical typical) |
|---|---|---|---|
| `python` | Any CPU, fallback | NumPy + Python for-loop | 1× (baseline) |
| `cpu` | Multi-core CPU | Numba `@njit(parallel=True)` + `prange` | 4–16× (scales with core count) |
| `gpu` | NVIDIA GPU | Numba `@cuda.jit` explicit CUDA kernel | 100–1000× for large N |

The GPU backend is not "better" than the CPU backend in all cases. For small problems (N = K×M < ~10,000 test points), the overhead of `cuda.to_device` and `copy_to_host` dominates the actual computation — the CPU parallel backend is faster. The GPU wins at large N (dense grids, high M), which is exactly where GAIO is used scientifically.

### 1.3 How the Backends Coexist Without Code Duplication

The three-stage pipeline in `AcceleratedBoxMap.map_boxes` is identical across all backends:

```
Stage 1 — test-point generation:  vectorised NumPy broadcast  (always)
Stage 2 — map application:        DISPATCHED per backend
Stage 3 — partition key lookup:   vectorised NumPy broadcast  (always)
```

Only Stage 2 differs. This isolation means:
- Stages 1 and 3 are written once, tested once, optimized once
- Adding a new backend (e.g., `OpenCL`, `Metal`, `oneAPI`) only requires implementing a new Stage 2 dispatcher
- The algorithm layer (`relative_attractor`, `recurrent_set`, etc.) is completely unaware of the backend — it calls `F(B)` and receives a `BoxSet`, regardless of how Stage 2 ran

### 1.4 Correspondence With GAIO.jl

In Julia, this dual-backend problem does not exist: Julia's JIT compiles `f` at first call to machine code on whatever hardware the thread pool or CUDA context is targeting. The explicit `f_jit` / `f_device` split in Python is the engineering cost of replicating Julia's "write once, run fast everywhere" behaviour in a language where generic JIT is not the default. The `AcceleratedBoxMap` constructor is the Python analogue of Julia's type specialization: it captures the compiled function at construction time and amortizes the compilation cost across all subsequent `map_boxes` calls.

---

## 2. GPU Kernel Thread Layout

### 2.1 The Work Item

The GPU kernel must apply `f_device` to every row of the `test_pts` array, which has shape `(N, n)` where `N = K × M` (K = active cells, M = test points per cell) and `n` = spatial dimension (typically 2–4).

**Each work item is one row**: one call to `f_device`, reading `test_pts[i]` and writing `mapped[i]`. Each work item is independent — no thread communicates with another.

### 2.2 Why a 1-D Grid

The `test_pts` array is logically 1-D: a list of N independent points. There is no inherent 2-D or 3-D structure that would benefit from a 2-D or 3-D CUDA grid (the spatial dimension `n` is far too small to parallelize — it's 2, 3, or 4). A 1-D grid is the correct abstraction.

```
Grid layout:
┌──────────────────────────────────────────────────────────────┐
│  Block 0     │  Block 1     │  ...  │  Block bpg-1          │
│  [0..255]    │  [256..511]  │  ...  │  [N-(N%256)..bpg*256) │
└──────────────────────────────────────────────────────────────┘
                                                  ↑
                              Tail block: up to 255 threads idle
                              (guarded by `if idx < N`)
```

### 2.3 Thread Index Calculation

```python
idx = cuda.grid(1)
# Equivalent to:
# idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
```

`cuda.grid(1)` is the canonical Numba idiom for 1-D global thread index. It expands to the hardware registers `blockIdx.x`, `blockDim.x`, and `threadIdx.x` — zero overhead at PTX level.

### 2.4 Block Size: 256 Threads

The default `THREADS_PER_BLOCK = 256` satisfies four constraints simultaneously:

1. **Warp alignment**: 256 = 8 × 32 (warp size). All 8 warps in a block execute as full warps — no warp-level lane masking overhead.

2. **Occupancy**: On all NVIDIA architectures from Kepler (compute 3.0) through Hopper (compute 9.0), 256 threads/block allows at least 2 concurrent blocks per SM, achieving ≥ 50% theoretical occupancy. Higher blocks-per-SM increase latency hiding for memory-bound kernels.

3. **Register pressure**: Our kernel body is very lightweight — `idx` computation, array slice, `f_device` call, loop write. Register usage is dominated by `f_device`. With 256 threads, the register file per block is half the SM maximum, leaving headroom for `f_device`'s local variables.

4. **Hardware maximum**: The CUDA spec mandates `blockDim ≤ 1024`. 256 is comfortably below this, with room to increase `threads_per_block` for specific GPU models via the `AcceleratedBoxMap` constructor.

### 2.5 Grid Size Formula

```python
blocks_per_grid = math.ceil(N / threads_per_block)   # ⌈N / 256⌉
```

This is the minimal grid that covers all N work items. The final block may launch up to `threads_per_block - 1` extra threads — all guarded by the `if idx < N` check in the kernel and therefore idle. The idle overhead is at most `(threads_per_block - 1) / N` ≈ 0% for large N, where the GPU is most useful.

For `N = 4,096,000` test points (e.g., a [1024×1024] grid with 4 test points per cell):
```
blocks_per_grid = ⌈4,096,000 / 256⌉ = 16,000 blocks
Total threads   = 16,000 × 256 = 4,096,000 (exact, no tail)
```

### 2.6 Memory Access Coalescing

Threads in a single warp have consecutive `idx` values: `{base, base+1, ..., base+31}`. They access:

```
test_pts[base],   test_pts[base+1],   ...,   test_pts[base+31]
```

In row-major (C-contiguous) storage, these rows are laid out consecutively in memory:

```
Memory address of test_pts[i, j]  =  base_ptr + (i * n + j) * 8
```

For `j = 0` (first element of each row), the stride between consecutive threads is `n * 8` bytes. For `n = 2`: stride = 16 bytes; for `n = 4`: stride = 32 bytes. A warp accessing `n = 2` rows produces a 32 × 16 = 512-byte transaction, which fits within two 256-byte cache line reads — **coalesced access**. For `n = 4`: 32 × 32 = 1024 bytes = four 256-byte cache line reads — still fully coalesced.

The same analysis applies to writes into `mapped[idx]` (the output row view). The GPU achieves near-peak DRAM bandwidth for the memory-bound portion of the kernel.

---

## 3. Memory Transfer Isolation for Phase 4

### 3.1 The Transfer Surface

All host↔device data movement in the GPU backend is confined to exactly **four lines** inside `CUDADispatcher.__call__`:

```python
# gaio/cuda/gpu_backend.py — CUDADispatcher.__call__

d_test_pts = cuda.to_device(pts)           # ← TRANSFER POINT A: H→D
d_mapped   = cuda.device_array((N, n), dtype=F64)  # ← device allocation

self.kernel[bpg, tpb](d_test_pts, d_mapped)        # ← kernel launch

return d_mapped.copy_to_host()            # ← TRANSFER POINT B: D→H
```

These two transfer points are the **only places** in the entire library where data crosses the PCIe bus. Everything above (`map_boxes` test-point generation, partition key lookup) runs on the host; everything below (the kernel) runs on the device.

### 3.2 Why Isolation Enables Phase 4

In Phase 4, each MPI rank owns a contiguous slice of the problem. For the `TransferOperator` build, rank `r` owns `domain._keys[r*chunk : (r+1)*chunk]`. The corresponding test points `test_pts_local` are computed on each rank independently (no communication).

The Phase 4 replacement of Transfer Point A is:

```python
# Phase 4: instead of transferring from host,
# the local slice was already computed on device:
d_test_pts = cuda.device_array_like(local_test_pts)
cuda.to_device(local_test_pts, to=d_test_pts)   # local MPI slice only
```

Or, with GPUDirect RDMA (CUDA-aware MPI):

```python
# Phase 4: GPUDirect — receive data from remote rank directly into VRAM
comm.Recv(d_test_pts, source=sender_rank, tag=0)   # no staging through CPU RAM
```

Transfer Point B is replaced by:

```python
# Phase 4: instead of copying to host, send directly to MPI root
comm.Send(d_mapped, dest=0, tag=1)   # GPUDirect RDMA
```

**The kernel call site is unchanged.** The Phase 4 MPI layer wraps Transfer Points A and B without touching `_map_kernel` or `AcceleratedBoxMap.map_boxes`.

### 3.3 The `test_pts` Array as the Natural MPI Boundary

The `test_pts` array (shape `(K*M, n)`) is the natural boundary between the CPU domain-decomposition logic (which cells does this rank own?) and the GPU computation (apply `f` to those cells' test points). Its properties make it ideal:

- **Flat**: a simple 2-D float64 array. No Python objects, no nested structures.
- **Independently computable per rank**: rank `r` computes its own `test_pts` from its local `centers` slice. No cross-rank communication is needed before Transfer Point A.
- **MPI-sendable**: a C-contiguous float64 buffer is directly transmissible as `MPI_DOUBLE` without serialization.

### 3.4 Morton-Coded Space-Filling Curves (Preview)

In Phase 4, domain decomposition by contiguous ranges of flat keys (e.g., keys `[0..K/2)` to rank 0, keys `[K/2..K)` to rank 1) is the simplest strategy but has poor spatial locality: rank 0 may own cells scattered across the entire domain. Poor locality means cells on rank 0's domain boundary interact frequently with cells on rank 1's domain — every such interaction requires cross-rank communication.

**Morton (Z-order) codes** address this by sorting cells by a space-filling curve that preserves spatial locality: cells that are spatially adjacent are also adjacent in Morton order. The flat key scheme (Section 3.1 of PHASE1_ARCHITECTURE_NOTES.md) is the prerequisite:

```python
# Morton encoding for 2-D: interleave bits of (m0, m1)
def morton_encode_2d(m0: int, m1: int) -> int:
    return _spread_bits(m0) | (_spread_bits(m1) << 1)

def _spread_bits(x: int) -> int:
    # Interleave zeros between bits of x
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF
    x = (x | (x <<  8)) & 0x00FF00FF00FF00FF
    x = (x | (x <<  4)) & 0x0F0F0F0F0F0F0F0F
    x = (x | (x <<  2)) & 0x3333333333333333
    x = (x | (x <<  1)) & 0x5555555555555555
    return x
```

After re-sorting `BoxSet._keys` by Morton code, domain decomposition by contiguous Morton ranges gives each MPI rank a spatially compact subdomain. The number of cells on the boundary between rank domains (which require cross-rank communication in the `TransferOperator` build) drops from O(K) to O(K^{(d-1)/d}) — sublinear in K, which is essential for strong scaling.

The flat `int64` key scheme is already in place. Phase 4 adds:
1. A `morton_sort(boxset: BoxSet) -> BoxSet` function that permutes `_keys` by Morton code
2. MPI scatter/gather based on Morton-sorted key ranges
3. The modified `CUDADispatcher` Transfer Points described in §3.2

---

## 4. Future Use of `cuda.shared.array` (Shared Memory)

### 4.1 The Bandwidth Bottleneck

Every thread in the current kernel independently reads its row of `test_pts` from **global memory** (VRAM). Global memory latency on modern NVIDIA GPUs is ~200–400 cycles. For a kernel with little arithmetic (our `f_device` for a simple ODE is O(n) flops), the kernel is **memory-bandwidth-bound**: most thread time is spent waiting for the load to complete.

`cuda.shared.array` is a software-managed scratchpad inside each SM (Streaming Multiprocessor). It is ~100× faster than global memory (~4-cycle access) and shared by all threads in the same block.

### 4.2 What Belongs in Shared Memory

Not all data benefits from caching in shared memory. The criterion is: data that is **read by multiple threads in the same block** and is **read-only during the kernel**. In our kernel, this is:

| Data | Shape | Reuse pattern | Shared-memory candidate? |
|---|---|---|---|
| `test_pts` | `(N, n)` | Each row read by exactly 1 thread | **No** — no reuse across threads |
| `unit_pts` (if generated in kernel) | `(M, n)` | Every thread in block reads all M rows | **Yes** |
| ODE parameters (e.g., `μ` for van der Pol) | scalar or `(p,)` | Every thread reads the same value | **Yes** |
| `mapped` (output) | `(N, n)` | Each row written by exactly 1 thread | **No** |

### 4.3 The System-Parameters Pattern

For parameterized ODEs — the most common scientific use case — the ODE right-hand side depends on a parameter vector `θ`:

```python
# Example: Lorenz system with parameters [σ, ρ, β]
@cuda.jit(device=True)
def f_device_lorenz(x, out, params):    # params = [σ, ρ, β]
    out[0] = params[0] * (x[1] - x[0])
    out[1] = x[0] * (params[1] - x[2]) - x[1]
    out[2] = x[0] * x[1] - params[2] * x[2]
```

`params` has 3 elements and is read by all 256 threads in every block. Loading it 256 times from global memory (one per thread) burns 256 × 24 bytes = 6 KB of bandwidth per block launch — wasteful.

The shared memory pattern caches `params` once per block:

```python
@cuda.jit
def _map_kernel_with_params(test_pts, mapped, params):
    # Allocate shared memory — size fixed at compile time (e.g., 32 params)
    shared_params = cuda.shared.array(32, dtype=float64)

    # Cooperative load: thread 0 loads all params, rest wait
    if cuda.threadIdx.x < params.shape[0]:
        shared_params[cuda.threadIdx.x] = params[cuda.threadIdx.x]
    cuda.syncthreads()               # barrier: all threads see loaded params

    # Each thread uses shared_params — single L1 hit per thread
    idx = cuda.grid(1)
    if idx < test_pts.shape[0]:
        x   = test_pts[idx]
        out = mapped[idx]
        f_device(x, out, shared_params)    # f_device reads from fast shared mem
```

This reduces the global memory bandwidth for `params` from `256 × |params|` bytes per block to `|params|` bytes per block — a 256× reduction in parameter-related bandwidth.

### 4.4 Why Not Implement This in Phase 3

The shared memory optimization requires:
1. Extending the `f_device` signature to `(x, out, params)` — a breaking change to the user API
2. `cuda.shared.array` size must be a compile-time constant — requires knowing max `|params|` in advance
3. The cooperative load pattern (`cuda.syncthreads()`) must be inserted correctly in the kernel factory

Phase 3 establishes the kernel factory (`make_map_kernel`) and the transfer isolation (`CUDADispatcher`) that Phase 4 will extend with this pattern. The architectural groundwork is already in place: `CUDADispatcher.__call__` currently passes only `(d_test_pts, d_mapped)` to the kernel. Adding `params`:

```python
# Phase 4 kernel launch:
d_params = cuda.to_device(params_array)         # small, rare transfer
self.kernel[bpg, tpb](d_test_pts, d_mapped, d_params)
```

The `make_map_kernel` factory would produce a kernel that copies `d_params` into shared memory before the per-thread computation. No changes to `AcceleratedBoxMap.map_boxes` or the Phase 2 algorithm layer are required.
