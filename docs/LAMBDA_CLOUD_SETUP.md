# Lambda Cloud Deployment Guide: GAIO.py Multi-GPU

Step-by-step instructions for deploying GAIO.py on a fresh Lambda Labs
GPU instance and running the Phase 3/4/5 benchmarks.

**Target instance:**
`gpu_8x_a100` (8× A100 80 GB, NVLink) or `gpu_4x_a10` (4× A10 24 GB).
Smaller instances like `gpu_1x_a100` or `gpu_1x_a10` work for single-GPU
Phase 3 benchmarks only.  All steps assume **Lambda Stack Ubuntu 22.04**
with CUDA 12.x pre-installed by Lambda.

---

## 1. What Lambda Stack 22.04 Provides Out of the Box

Lambda Stack 22.04 instances ship with:

- CUDA toolkit + driver (CUDA 12.x, driver ≥ 525; observed 12.8 on A100 nodes)
- `nvidia-smi` and `nvcc` on PATH
- CUDA-aware OpenMPI at `/usr/mpi/gcc/openmpi-4.1.7rc1/` — **no recompile needed**
- **No conda** — must be installed manually (see §2)

Verify the CUDA version and OpenMPI CUDA support before proceeding:

```bash
nvcc --version
nvidia-smi
ompi_info | grep -i cuda    # should show "MPI extensions: ..., cuda, ..."
```

> **Note on `cudatoolkit` version:** `environment.yml` pins `cudatoolkit=11.8`
> inside the conda env.  This is intentional and correct — CUDA drivers are
> forward-compatible, so a 12.x system driver runs code built against an 11.x
> runtime without changes.  Do not upgrade the pin; numba 0.59 is validated
> against 11.8.

---

## 2. One-Shot Setup Script

Copy, paste, and run the entire block below in a fresh SSH session.
It takes approximately 5–10 minutes on a `gpu_1x_a100` instance
(no OpenMPI compile needed on Lambda Stack 22.04).

```bash
#!/usr/bin/env bash
set -euo pipefail

# ── 0. Variables ─────────────────────────────────────────────────────────────
# Lambda Stack 22.04 ships CUDA-aware OpenMPI here — no custom build required.
SYSTEM_OMPI="/usr/mpi/gcc/openmpi-4.1.7rc1"
GAIO_REPO="$HOME/GAIO.py"
CONDA_ENV="gaio"

# ── 1. Install Miniconda (not pre-installed on Lambda Stack 22.04) ────────────
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# Persist conda initialisation for future SSH sessions
"$HOME/miniconda3/bin/conda" init bash
echo 'source "$HOME/miniconda3/etc/profile.d/conda.sh"' >> ~/.bashrc

# ── 2. Persistent environment variables for system OpenMPI ────────────────────
cat >> ~/.bashrc << EOF

# GAIO.py: Lambda Stack CUDA-aware OpenMPI
export PATH="${SYSTEM_OMPI}/bin:\$PATH"
export LD_LIBRARY_PATH="${SYSTEM_OMPI}/lib:\${LD_LIBRARY_PATH:-}"
export OMPI_MCA_opal_cuda_support=1      # enable GPUDirect at runtime
export UCX_TLS=rc,cuda_copy,cuda_ipc     # UCX transport: IB + CUDA peer
EOF

export PATH="${SYSTEM_OMPI}/bin:$PATH"
export LD_LIBRARY_PATH="${SYSTEM_OMPI}/lib:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_opal_cuda_support=1
export UCX_TLS=rc,cuda_copy,cuda_ipc

# Verify CUDA-aware MPI is active
ompi_info | grep -i "cuda\|MPI extensions"

# ── 3. Conda environment from environment.yml ─────────────────────────────────
# cudatoolkit=11.8 in environment.yml is backward-compatible with CUDA 12.x
# system drivers — do not change the pin.
cd "${GAIO_REPO}"
conda env create -f environment.yml
conda activate "${CONDA_ENV}"

# ── 4. mpi4py — compile against the system CUDA-aware OpenMPI ────────────────
# Must use --no-binary so mpi4py links to the correct libmpi.so, not conda's.
MPICC="${SYSTEM_OMPI}/bin/mpicc" \
    pip install --no-binary mpi4py mpi4py

# ── 5. PETSc / SLEPc — distributed eigensolve (Phase 4, optional) ────────────
# conda-forge builds link against their own PETSc/MPI; they do NOT need to
# link to the system OpenMPI because they only use MPI for CPU-side Krylov
# basis work, not GPU-VRAM transfers.  Install is optional — Phase 4 falls
# back to scipy ARPACK automatically if slepc4py is absent.
conda install -y -c conda-forge petsc4py slepc4py

# ── 6. Install GAIO.py ────────────────────────────────────────────────────────
pip install -e ".[gpu,mpi,hpc,dev]"

# ── 7. One-rank smoke test ────────────────────────────────────────────────────
python -c "
from gaio import Box, BoxPartition, BoxSet, TransferOperator, relative_attractor
from gaio.mpi import check_cuda_aware_mpi
import numpy as np
check_cuda_aware_mpi()
print('GAIO.py import OK  version:', __import__('gaio').__version__)
"

echo ""
echo "=== Setup complete.  Run verification steps below ==="
```

---

## 3. Julia Setup (for `benchmark_vs_julia.py`)

> **Skip this section** if you only need the GAIO.py benchmarks (Phases 3–5).
> Julia is required only for the side-by-side GAIO.py vs GAIO.jl comparison.

### 3a. Install Julia

```bash
# Download Julia 1.10 LTS (adjust version as needed)
JULIA_VERSION=1.10.8
wget -q "https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" \
    -O /tmp/julia.tar.gz
tar -xzf /tmp/julia.tar.gz -C "$HOME"

# Add to PATH permanently
echo "export PATH=\"\$HOME/julia-${JULIA_VERSION}/bin:\$PATH\"" >> ~/.bashrc
export PATH="$HOME/julia-${JULIA_VERSION}/bin:$PATH"

julia --version    # should print: julia version 1.10.x
```

### 3b. Instantiate the GAIO.jl project

The GAIO.jl reference project lives in `references/GAIO.jl-master/`.
`Pkg.instantiate()` downloads and precompiles all its declared dependencies:

```bash
cd ~/GAIO.py
julia --project=references/GAIO.jl-master -e 'using Pkg; Pkg.instantiate()'
# Takes ~2–5 min on first run; subsequent runs are instant.
```

### 3c. Create the benchmark environment (required for SIMD)

> **Why a separate environment?**  `SIMD.jl` and `HostCPUFeatures.jl` are
> declared as extension triggers in `references/GAIO.jl-master/Project.toml`
> (`SIMDExt = ["SIMD", "HostCPUFeatures"]`).  Julia's extension system requires
> triggers to be loaded *after* the parent package finishes initialising.
> Running scripts with `--project=references/GAIO.jl-master` makes GAIO load
> SIMD during its own init, creating a circular dependency that silently
> prevents SIMDExt from loading.
>
> The fix: a **thin wrapper environment** that depends on both GAIO *and* SIMD.
> Julia loads GAIO first (no SIMD yet), then loads SIMD, which triggers
> SIMDExt cleanly.

```bash
# Create a dedicated benchmark environment in ~/gaio_bench_env
mkdir -p ~/gaio_bench_env
cat > ~/gaio_bench_env/Project.toml << 'EOF'
[deps]
GAIO       = "33d280d1-ac47-4b0f-9c2e-fa6a385d0226"
CUDA       = "052768ef-5323-5732-b1bb-66c8b64840ba"
HostCPUFeatures = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
SIMD       = "fdea26ae-647d-5447-a871-4b548cad5224"
EOF

# Install deps into this environment, pointing at our local GAIO.jl copy
julia --project=~/gaio_bench_env -e "
using Pkg
Pkg.develop(path=\"$HOME/GAIO.py/references/GAIO.jl-master\")
Pkg.add([\"SIMD\", \"HostCPUFeatures\", \"CUDA\"])
Pkg.instantiate()
Pkg.precompile()
"
# CUDA.jl: ~5–10 min first time.  SIMD + HostCPUFeatures: ~30 s.
```

Verify SIMD loaded:

```bash
julia --project=~/gaio_bench_env -t auto -e '
using GAIO, SIMD
println("SIMDExt loaded: ", Base.get_extension(GAIO, :SIMDExt) !== nothing)
P = GAIO.BoxGrid(GAIO.Box((0.0,0.0,0.0),(5.0,5.0,5.0)), (2,2,2))
F = GAIO.BoxMap(:simd, identity, P)
println("BoxMap(:simd) OK — type: ", nameof(typeof(F)))
'
# Expected:
#   SIMDExt loaded: true
#   BoxMap(:simd) OK — type: CPUSampledBoxMap
```

### 3d. Run the benchmark using the wrapper environment

Pass `--julia-project` to point at the benchmark environment instead of the
GAIO.jl project directly:

```bash
python benchmarks/benchmark_vs_julia.py \
    --julia-project ~/gaio_bench_env
# julia-simd row should now appear instead of julia-default
```

> **Note:** `--setup-julia` is not needed when using the wrapper environment
> — SIMD.jl and CUDA.jl were already installed in step 3c above.

### Full Julia setup one-liner (with SIMD environment)

```bash
JULIA_VERSION=1.10.8 && \
wget -q "https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" \
    -O /tmp/julia.tar.gz && \
tar -xzf /tmp/julia.tar.gz -C "$HOME" && \
export PATH="$HOME/julia-${JULIA_VERSION}/bin:$PATH" && \
echo "export PATH=\"\$HOME/julia-${JULIA_VERSION}/bin:\$PATH\"" >> ~/.bashrc && \
julia --project="$HOME/GAIO.py/references/GAIO.jl-master" \
    -e 'using Pkg; Pkg.instantiate()' && \
mkdir -p ~/gaio_bench_env && \
printf '[deps]\nGAIO = "33d280d1-ac47-4b0f-9c2e-fa6a385d0226"\nCUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"\nHostCPUFeatures = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"\nSIMD = "fdea26ae-647d-5447-a871-4b548cad5224"\n' > ~/gaio_bench_env/Project.toml && \
julia --project=~/gaio_bench_env -e "
    using Pkg
    Pkg.develop(path=\"\$HOME/GAIO.py/references/GAIO.jl-master\")
    Pkg.add([\"SIMD\", \"HostCPUFeatures\", \"CUDA\"])
    Pkg.instantiate(); Pkg.precompile()
" && \
python "$HOME/GAIO.py/benchmarks/benchmark_vs_julia.py" \
    --julia-project ~/gaio_bench_env
```

---

## 4. Python Verification Checklist

Run these after the setup script finishes, in the activated `gaio` conda env:

### 3a. CUDA-aware MPI probe

```bash
python -c "
from gaio.mpi import check_cuda_aware_mpi
ok = check_cuda_aware_mpi()
assert ok, 'CUDA-aware MPI not detected — check OMPI_MCA_opal_cuda_support=1'
print('PASS: CUDA-aware MPI confirmed')
"
```

### 3b. GPU binding (one rank per GPU)

```bash
# Each rank should see a different GPU ID
mpirun -n 4 --bind-to socket python -c "
from gaio.mpi import rank
from numba import cuda
cuda.select_device(rank())
dev = cuda.get_current_device()
print(f'rank {rank()} → GPU {dev.id}: {dev.name.decode()}')
"
```

Expected output (4× A10):
```
rank 0 → GPU 0: NVIDIA A10
rank 1 → GPU 1: NVIDIA A10
rank 2 → GPU 2: NVIDIA A10
rank 3 → GPU 3: NVIDIA A10
```

### 3c. GPUDirect RDMA probe

```bash
mpirun -n 2 python -c "
from gaio.mpi import is_rdma_capable, get_comm
comm = get_comm()
rdma = is_rdma_capable(comm)
from gaio.mpi import rank
if rank() == 0:
    print('GPUDirect RDMA:', 'YES' if rdma else 'NO (CPU staging fallback)')
"
```

### 3d. Phase 4 distributed attractor parity check

```bash
mpirun -n 4 --bind-to socket python - <<'EOF'
import numpy as np
from gaio import Box, BoxPartition, BoxSet, SampledBoxMap, relative_attractor
from gaio.mpi import get_comm, rank
from numba import cuda
cuda.select_device(rank())

def henon(x): return np.array([1 - 1.4*x[0]**2 + x[1], 0.3*x[0]])
domain = Box(np.array([0.,0.]), np.array([1.5, 0.5]))
P = BoxPartition(domain, [2,2])
pts = np.stack(np.meshgrid(np.linspace(-1,1,3), np.linspace(-1,1,3)), axis=-1).reshape(-1,2)
F = SampledBoxMap(henon, domain, pts)

comm = get_comm()
A_dist   = relative_attractor(F, BoxSet.full(P), steps=10, comm=comm)
A_serial = relative_attractor(F, BoxSet.full(P), steps=10, comm=False)

if rank() == 0:
    assert np.array_equal(A_dist._keys, A_serial._keys), \
        f"Mismatch: dist={len(A_dist)} serial={len(A_serial)}"
    print(f"[PASS] Distributed attractor == serial: {len(A_dist)} cells")
EOF
```

---

## 5. Running the Benchmarks

### Phase 3 — single-GPU acceleration (baseline, no mpirun needed)

```bash
# ~2 min; shows Python → Numba JIT → CUDA speedup across all three backends.
# Verified on A100 SXM4 40 GB: cpu=112×, gpu=174× vs. pure-Python baseline.
python benchmarks/benchmark_phase3.py --steps 10 --grid-res 2
```

### Phase 4 — multi-GPU MPI scaling (1-GPU vs N-GPU)

Requires a multi-GPU instance (`gpu_4x_a10`, `gpu_4x_a100`, etc.).
On a single-GPU instance, run with `-n 1` to confirm correctness only.

```bash
# Bind one rank per GPU; large enough problem for meaningful GPU utilisation
mpirun -n 4 --bind-to socket \
    python benchmarks/benchmark_phase4.py \
    --steps 16 --grid-res 4 --test-pts 5

# Expected: ~3–4× speedup for T_op, ~3.5× for attractor, near-linear scaling
```

### Phase 5 — dynamic load balancing (non-uniform attractor)

```bash
# Hénon horseshoe: extreme COO imbalance, clear Phase 5 benefit
mpirun -n 4 --bind-to socket \
    python benchmarks/benchmark_phase5.py \
    --map henon --steps 12 --n-frames 4

# Ikeda spiral: moderate imbalance
mpirun -n 4 --bind-to socket \
    python benchmarks/benchmark_phase5.py \
    --map ikeda --steps 10 --n-frames 4

# Four-Wing (near-uniform control — Phase 5 overhead should be < 5%)
mpirun -n 4 --bind-to socket \
    python benchmarks/benchmark_phase5.py \
    --map fourwing --steps 8 --n-frames 3
```

---

## 6. Instance Type Selection Guide

| Benchmark goal | Recommended instance | Notes |
|---|---|---|
| Phase 3 single-GPU | `gpu_1x_a10` or `gpu_1x_a100` | Cheapest; Python/Numba/CUDA comparison |
| Phase 4 scaling | `gpu_4x_a10` or `gpu_4x_a100` | Need ≥4 GPUs on same NVLink fabric |
| Phase 5 max imbalance | `gpu_4x_a100` | Larger A100 VRAM fits deeper subdivision |
| Full suite | `gpu_8x_a100` | Reserve for publication-quality numbers |

For a quick Phase 4 + Phase 5 comparison run (~$3–5 total), `gpu_4x_a10`
with `--steps 14 --grid-res 3 --test-pts 4` is a good balance of cost and
statistical significance.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `conda: command not found` | Miniconda not installed or not sourced | Run `source ~/.bashrc` or re-login; if miniconda not installed, run §2 step 1 |
| `PMPI_Init_thread: Unknown error class` | OFI fabric mismatch on Lambda | `export UCX_TLS=tcp,cuda_copy` (disables IB, slower but safe) |
| `mpirun: command not found` | PATH not updated | `source ~/.bashrc` or re-login |
| `ImportError: libmpi.so` | mpi4py linked to wrong libmpi | Reinstall: `MPICC=/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpicc pip install --no-binary mpi4py mpi4py` |
| `NumbaPerformanceWarning: Grid size 2` | Problem too small for GPU | Use `--steps 14 --grid-res 4 --test-pts 5`; warnings already suppressed in benchmarks |
| RDMA probe returns `NO` | Lambda's network stack lacks UCX IB | Normal on A10 nodes; GPU-to-GPU via `cuda_ipc` still available; set `UCX_TLS=cuda_copy,cuda_ipc` |
| `slepc4py` not found | SLEPc optional dependency absent | `conda install -c conda-forge petsc4py slepc4py`; fallback to scipy ARPACK is automatic |
| Rank segfaults at MPI_Init | CUDA context not created before MPI calls | Call `cuda.select_device(rank())` before any GAIO computation |
| `ompi_info` shows no `cuda` in MPI extensions | Wrong mpirun on PATH | Confirm `which mpirun` points to `/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun` |

---

## 8. Known GAIO.jl Bugs Affecting Benchmarks

These bugs were identified while developing `benchmark_vs_julia.py` and the
Julia benchmark harness.  The workarounds are already implemented in
`gaio_julia_benchmark.jl`; this section documents the root cause for
reference if/when the upstream GAIO.jl source is patched.

### Bug 1 — `construct_transfers` GPU path always returns nnz=0

**File:** `GAIO.jl/ext/CUDAExt.jl`, function `construct_transfers`, line 85

**Symptom:** `TransferOperator(F_gpu, A, A).mat` has `nnz == 0` regardless
of the attractor size.  The GPU map (`relative_attractor`) works correctly;
only `construct_transfers` is affected.

**Root cause — variable shadowing:**

```julia
function construct_transfers(
        g::GPUSampledBoxMap, domain::BoxSet{R,Q,S}, codomain::BoxSet{U,H,W}; ...
    ) where {N,T,R<:Box{N,T},Q,S,U,H,W}
    ...
    mat = D()
    codomain = BoxSet(P2, S())   # ← BUG: shadows the function parameter `codomain`
    ...
    for i in 1:nk*np
        ...
        hit in codomain.set || continue   # ← always false: `codomain` is now empty
        mat = mat ⊔ ((hit,key) => 1)
    end
```

The intent was to create a fresh local accumulation set; instead, the local
`codomain` variable **overwrites** the function parameter of the same name.
The membership test `hit in codomain.set` then checks against an **empty**
set, so every hit is discarded and `mat` remains empty.

**Minimal fix (not yet applied upstream):**

Rename the local variable so it does not shadow the parameter:

```julia
_empty_codom = BoxSet(P2, S())   # renamed: no longer shadows the parameter
oob = out_of_bounds(P)
while keys_left > 0
    ...
    for i in 1:nk*np
        ...
        hit in codomain.set || continue  # now checks the real codomain ✓
        mat = mat ⊔ ((hit,key) => 1)
    end
end
```

**Benchmark workaround** (`gaio_julia_benchmark.jl`): when nnz=0 is detected
for a GPU trial, the benchmark falls back to a CPU `BoxMap(:grid, ...)` for
`TransferOperator` timing and records the note
`"GPU map + CPU T_op (GPU construct_transfers nnz=0 bug)"` in the JSON output.
The attractor (map) timing from the GPU path is still valid and reported.

### Bug 2 — SIMD extension silently not loaded when using `--project=GAIO.jl`

**Symptom:** `BoxMap(:simd, ...)` falls back to `:grid` (FLoops default)
without error.  `Base.get_extension(GAIO, :SIMDExt)` returns `nothing`.

**Root cause:** Julia's extension system requires extension triggers (`SIMD`,
`HostCPUFeatures`) to be loaded *after* the parent package finishes
initialising.  Running `julia --project=GAIO.jl-master` makes GAIO load SIMD
during its own `__init__`, creating a circular dependency that silently
prevents `SIMDExt` from activating.

**Fix:** Use a thin wrapper environment that depends on both GAIO *and* SIMD
(see §3b–3c).  Julia loads GAIO first, then loads SIMD, which correctly
triggers `SIMDExt`.  This is the reason `--julia-project ~/gaio_bench_env` is
required rather than pointing directly at the GAIO.jl source directory.
