# vast.ai Deployment Guide: GAIO.py Multi-GPU

Step-by-step instructions for deploying GAIO.py on a **vast.ai** instance
using the **PyTorch template** and running the Phase 3/4/5 benchmarks.

**Tested on:** 8× A100-PCIE-40GB, CUDA 12.8, vast.ai PyTorch template (Ubuntu 22.04).

> **Lambda Cloud users:** see `docs/LAMBDA_CLOUD_SETUP.md` instead.
> The main differences from Lambda are highlighted in §0 below.

---

## 0. Key Differences vs Lambda Cloud

| | Lambda Stack 22.04 | vast.ai PyTorch template |
|---|---|---|
| Conda | Not installed — must install Miniconda | **Pre-installed** at `/opt/miniforge3` |
| CUDA toolkit | Driver only (12.x); conda env provides `cudatoolkit=11.8` | Full toolkit at `/usr/local/cuda` (12.8) |
| OpenMPI | CUDA-aware at `/usr/mpi/gcc/openmpi-4.1.7rc1/` | **Not installed** — install via conda-forge |
| GPU topology | A100 SXM4 NVLink on Lambda HPC nodes | A100 **PCIe** — no NVLink; use `cuda_ipc` |
| InfiniBand | Available on some Lambda nodes | **Not available** — use TCP transport |
| GAIO.jl path | `~/GAIO.py/references/GAIO.jl-master/` | `/workspace/GAIO.jl/` |
| `mpirun` location | `/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun` | `$CONDA_PREFIX/bin/mpirun` (conda-forge) |

---

## 1. What the vast.ai PyTorch Template Provides

- CUDA 12.x driver + full toolkit at `/usr/local/cuda`
- `nvidia-smi`, `nvcc` on PATH
- **Miniforge3 at `/opt/miniforge3`** — conda is available but not initialised
  in the default shell
- No OpenMPI, no Julia
- GAIO.py and GAIO.jl cloned to `/workspace/` (if you selected workspace
  persistence or cloned manually)

Verify before proceeding:

```bash
nvidia-smi                  # confirm GPU count and driver
nvcc --version              # confirm CUDA toolkit version
/opt/miniforge3/bin/conda --version
```

---

## 2. One-Shot Setup Script

Copy, paste, and run the entire block below in a fresh SSH session.
Takes approximately 10–20 minutes on an 8× A100 instance
(most of the time is CUDA.jl precompile in step 7).

```bash
#!/usr/bin/env bash
set -euo pipefail

# ── 0. Variables ──────────────────────────────────────────────────────────────
GAIO_REPO="/workspace/GAIO.py"
GAIO_JL="/workspace/GAIO.jl"
CONDA_ENV="gaio"
CONDA_BASE="/opt/miniforge3"

# ── 1. Initialise conda for this session and future SSH sessions ───────────────
source "${CONDA_BASE}/etc/profile.d/conda.sh"
"${CONDA_BASE}/bin/conda" init bash
echo 'source "/opt/miniforge3/etc/profile.d/conda.sh"' >> ~/.bashrc

# ── 2. Persistent env vars — PCIe A100: no IB, no NVLink ─────────────────────
# UCX_TLS: TCP for inter-rank + CUDA IPC for same-node peer copies.
# Do NOT use "rc" (InfiniBand RDMA) — not available on vast.ai.
cat >> ~/.bashrc << 'EOF'

# GAIO.py: vast.ai MPI transport (PCIe A100 — no InfiniBand)
export OMPI_MCA_opal_cuda_support=1
export UCX_TLS=tcp,cuda_copy,cuda_ipc
EOF

export OMPI_MCA_opal_cuda_support=1
export UCX_TLS=tcp,cuda_copy,cuda_ipc

# ── 3. Create conda env from environment.yml ──────────────────────────────────
# cudatoolkit=11.8 in environment.yml is forward-compatible with CUDA 12.x
# system drivers — do not change the pin.
cd "${GAIO_REPO}"
conda env create -f environment.yml
conda activate "${CONDA_ENV}"

# ── 4. OpenMPI — install CUDA-aware build from conda-forge ────────────────────
# vast.ai has no system OpenMPI; conda-forge's openmpi on Linux x86_64 is built
# with UCX + CUDA support.  Install it into the active env.
conda install -y -c conda-forge openmpi

# Verify CUDA support in the newly installed MPI:
ompi_info | grep -i "cuda\|MPI extensions" || true

# ── 5. mpi4py — compile against the conda-forge OpenMPI ──────────────────────
# --no-binary forces a source build that links to the correct libmpi.so.
MPICC="$(which mpicc)" \
    pip install --no-binary mpi4py mpi4py

# ── 6. Install GAIO.py ────────────────────────────────────────────────────────
pip install -e "${GAIO_REPO}/.[gpu,mpi,dev]"

# ── 7. One-rank smoke test ────────────────────────────────────────────────────
python -c "
from gaio import Box, BoxPartition, BoxSet, TransferOperator, relative_attractor
import numpy as np
print('GAIO.py import OK  version:', __import__('gaio').__version__)
"

echo ""
echo "=== Python setup complete.  Proceed to Julia setup (§3) if needed. ==="
```

---

## 3. Julia Setup (for `benchmark_vs_julia.py`)

> Skip if you only need Phase 4/5 benchmarks.
> Julia is required only for the GAIO.py vs GAIO.jl Phase 3 comparison.

### 3a. Install Julia

```bash
JULIA_VERSION=1.10.8
wget -q "https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" \
    -O /tmp/julia.tar.gz
tar -xzf /tmp/julia.tar.gz -C "$HOME"

echo "export PATH=\"\$HOME/julia-${JULIA_VERSION}/bin:\$PATH\"" >> ~/.bashrc
export PATH="$HOME/julia-${JULIA_VERSION}/bin:$PATH"

julia --version   # should print: julia version 1.10.x
```

### 3b. Instantiate GAIO.jl

On vast.ai, GAIO.jl lives at `/workspace/GAIO.jl` (not inside GAIO.py/references/).

```bash
julia --project=/workspace/GAIO.jl -e 'using Pkg; Pkg.instantiate()'
# Takes ~2–5 min on first run.
```

### 3c. Create the benchmark wrapper environment (required for SIMD)

> **Why a separate environment?**  Pointing Julia directly at `/workspace/GAIO.jl`
> with `--project` causes GAIO to attempt loading SIMD during its own `__init__`,
> creating a circular dependency that silently prevents `SIMDExt` from loading
> (see `docs/LAMBDA_CLOUD_SETUP.md §8 Bug 2` for full diagnosis).
> The thin wrapper environment below loads GAIO first, then SIMD, which
> triggers `SIMDExt` correctly.

```bash
mkdir -p ~/gaio_bench_env
cat > ~/gaio_bench_env/Project.toml << 'EOF'
[deps]
GAIO            = "33d280d1-ac47-4b0f-9c2e-fa6a385d0226"
CUDA            = "052768ef-5323-5732-b1bb-66c8b64840ba"
HostCPUFeatures = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
SIMD            = "fdea26ae-647d-5447-a871-4b548cad5224"
EOF

julia --project=~/gaio_bench_env -e "
using Pkg
Pkg.develop(path=\"/workspace/GAIO.jl\")
Pkg.add([\"SIMD\", \"HostCPUFeatures\", \"CUDA\"])
Pkg.instantiate()
Pkg.precompile()
"
# CUDA.jl: ~5–15 min first time.
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
```

### 3d. Patch two GAIO.jl bugs before benchmarking

Two bugs in the upstream GAIO.jl source affect benchmark correctness.
Apply both patches now — they take under a minute.

#### Bug 1 — `CUDAExt.jl`: GPU `TransferOperator` always returns nnz=0

**File:** `/workspace/GAIO.jl/ext/CUDAExt.jl`

Line 85 reassigns the `codomain` function parameter to an empty `BoxSet`,
so the hit-detection test `hit in codomain.set` is always false and the
resulting matrix is all-zeros.

```bash
# Confirm the bad line is present (prints the line if unfixed)
grep -n "codomain = BoxSet(P2" /workspace/GAIO.jl/ext/CUDAExt.jl

# Delete it
sed -i '/codomain = BoxSet(P2, S())/d' /workspace/GAIO.jl/ext/CUDAExt.jl

# Verify (should print nothing)
grep -n "codomain = BoxSet(P2" /workspace/GAIO.jl/ext/CUDAExt.jl
echo "CUDAExt patch applied"
```

#### Bug 2 — `SIMDExt.jl`: `construct_transfers` rejects `show_progress` keyword

**File:** `/workspace/GAIO.jl/ext/SIMDExt.jl`

`TransferOperator` passes `show_progress=true/false` to `construct_transfers`,
but the two SIMDExt overloads don't declare that keyword — Julia raises a
`MethodError: unsupported keyword argument` and the CPU benchmark fails.

```bash
# Confirm both signatures are unfixed (prints 2 lines if unfixed)
grep -n "^@inbounds.*function construct_transfers\|^@inbounds @muladd.*function construct_transfers" \
    /workspace/GAIO.jl/ext/SIMDExt.jl

# Patch: add "; show_progress=false, kwargs..." to both 2-argument and 3-argument overloads.
# The signatures look like:
#   function construct_transfers(\n        G::CPUSampledBoxMap{simd}, domain::BoxSet{R,Q,S}\n    )
# We add the keyword before the closing ) on the next line.

python3 - <<'PYEOF'
import re, pathlib

p = pathlib.Path("/workspace/GAIO.jl/ext/SIMDExt.jl")
src = p.read_text()

# Pattern: the two-arg overload (no codomain parameter)
src = re.sub(
    r'(@inbounds function construct_transfers\(\s*\n\s*G::CPUSampledBoxMap\{simd\}, domain::BoxSet\{R,Q,S\}\s*\n\s*\))',
    lambda m: m.group(0).rstrip(')') + '; show_progress::Bool=false, kwargs...\n    )',
    src
)

# Pattern: the three-arg overload (with codomain parameter)
src = re.sub(
    r'(@inbounds @muladd function construct_transfers\(\s*\n\s*G::CPUSampledBoxMap\{simd\}, domain::BoxSet\{R,Q\}, codomain::BoxSet\{U,H\}\s*\n\s*\))',
    lambda m: m.group(0).rstrip(')') + '; show_progress::Bool=false, kwargs...\n    )',
    src
)

p.write_text(src)
print("SIMDExt patch written")
PYEOF

# Verify both signatures now have show_progress
grep -n "show_progress" /workspace/GAIO.jl/ext/SIMDExt.jl
```

#### Recompile GAIO.jl after patching

```bash
julia --project=~/gaio_bench_env -e '
using Pkg
Pkg.precompile()
'
# Takes ~2–5 min (CUDA.jl is already cached, only GAIO is recompiled)
```

#### Verify both fixes

```bash
# CUDAExt: TransferOperator should return non-zero nnz on GPU
julia --project=~/gaio_bench_env -e '
using GAIO, CUDA, SparseArrays
P = GAIO.BoxGrid(GAIO.Box((0.0,0.0,0.0),(5.0,5.0,5.0)), (2,2,2))
fw(x) = (0.2x[1]+x[2]*x[3], -0.4x[2]-0.01x[1]-x[3]*x[2], -x[3]-x[1]*x[2])
f(x)  = GAIO.rk4_flow_map(fw, x, 0.01f0, 20)
F = GAIO.BoxMap(:grid, :gpu, f, P; n_points=(4,4,4))
A = GAIO.BoxSet(P, P)
T = GAIO.TransferOperator(F, A, A)
println("GPU T_op nnz = ", nnz(T.mat), "  (expect > 0)")
' 2>&1

# SIMDExt: TransferOperator with SIMD map should not raise MethodError
julia --project=~/gaio_bench_env -t auto -e '
using GAIO, SIMD, HostCPUFeatures, SparseArrays
P = GAIO.BoxGrid(GAIO.Box((0.0,0.0,0.0),(5.0,5.0,5.0)), (2,2,2))
fw(x) = (0.2x[1]+x[2]*x[3], -0.4x[2]-0.01x[1]-x[3]*x[2], -x[3]-x[1]*x[2])
f(x)  = GAIO.rk4_flow_map(fw, x, 0.01, 20)
F = GAIO.BoxMap(:grid, :simd, f, P; n_points=(4,4,4))
A = GAIO.BoxSet(P, P)
T = GAIO.TransferOperator(F, A, A)
println("SIMD T_op nnz = ", nnz(T.mat), "  (expect > 0)")
' 2>&1
```

---

## 4. Python Verification Checklist

Run these after setup, inside `conda activate gaio`:

### 4a. CUDA sanity check

```bash
python -c "
from numba import cuda
print('Numba CUDA devices:', cuda.gpus.lst)
print('GPU 0:', cuda.get_current_device().name.decode())
"
```

### 4b. GPU binding (one rank per GPU)

```bash
# Each rank should see a different GPU
mpirun -n 8 --bind-to socket python -c "
from mpi4py import MPI
from numba import cuda
rank = MPI.COMM_WORLD.Get_rank()
cuda.select_device(rank)
dev = cuda.get_current_device()
print(f'rank {rank} → GPU {dev.id}: {dev.name.decode()}')
"
```

Expected:
```
rank 0 → GPU 0: NVIDIA A100-PCIE-40GB
rank 1 → GPU 1: NVIDIA A100-PCIE-40GB
...
rank 7 → GPU 7: NVIDIA A100-PCIE-40GB
```

### 4c. MPI correctness / parity check

```bash
mpirun -n 4 --bind-to socket python - <<'EOF'
import numpy as np
from gaio import Box, BoxPartition, BoxSet, SampledBoxMap, relative_attractor
from mpi4py import MPI
from numba import cuda

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
cuda.select_device(rank)

def henon(x): return np.array([1 - 1.4*x[0]**2 + x[1], 0.3*x[0]])
domain = Box(np.array([0.,0.]), np.array([1.5, 0.5]))
P  = BoxPartition(domain, [2,2])
pts = np.stack(np.meshgrid(np.linspace(-1,1,3), np.linspace(-1,1,3)), axis=-1).reshape(-1,2)
F  = SampledBoxMap(henon, domain, pts)

A_dist   = relative_attractor(F, BoxSet.full(P), steps=10, comm=comm)
A_serial = relative_attractor(F, BoxSet.full(P), steps=10, comm=False)

if rank == 0:
    assert np.array_equal(A_dist._keys, A_serial._keys), \
        f"Mismatch: dist={len(A_dist)} serial={len(A_serial)}"
    print(f"[PASS] Distributed == serial: {len(A_dist)} cells")
EOF
```

---

## 5. Running the Benchmarks

Activate the conda env first in every new session:

```bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate gaio
cd /workspace/GAIO.py
```

### Phase 3 — Single-GPU vs CPU (no mpirun needed)

```bash
# ~3–5 min; shows Python → Numba JIT → CUDA speedup
python benchmarks/benchmark_phase3.py --steps 10 --grid-res 2
```

### Phase 3 — GAIO.py vs GAIO.jl (requires Julia setup from §3)

```bash
# NOTE: pass --julia-project pointing at the wrapper env, not /workspace/GAIO.jl directly
python benchmarks/benchmark_vs_julia.py \
    --julia-project ~/gaio_bench_env \
    --steps 10 --grid-res 2 --test-pts 4
```

### Phase 4 — Multi-GPU MPI scaling

```bash
# 8 GPUs — primary target on this instance
mpirun -n 8 --bind-to socket \
    python benchmarks/benchmark_phase4.py \
    --steps 16 --grid-res 4 --test-pts 4

# Expected: ~6–7× speedup for T_op (near-linear at 8 GPUs)
# Note: PCIe A100 (no NVLink) — inter-GPU bandwidth ~32 GB/s vs NVLink ~600 GB/s.
#        MPI communication will be slower than on NVLink nodes; compute scaling
#        is still linear, communication overhead is higher.
```

### Phase 5 — Dynamic Load Balancing

```bash
# Default: runs both "moderate" and "extreme" Lozi scenarios
mpirun -n 8 --bind-to socket \
    python benchmarks/benchmark_phase5.py

# Single scenario:
mpirun -n 8 --bind-to socket \
    python benchmarks/benchmark_phase5.py --scenario moderate

mpirun -n 8 --bind-to socket \
    python benchmarks/benchmark_phase5.py --scenario extreme

# With GPU backend:
mpirun -n 8 --bind-to socket \
    python benchmarks/benchmark_phase5.py --gpu
```

### Saving results

Append `--json results/<name>.json` to any benchmark to save a machine-readable
copy of the output:

```bash
python benchmarks/benchmark_vs_julia.py \
    --julia-project ~/gaio_bench_env \
    --json results/phase3_vs_julia_8xa100_pcie.json

mpirun -n 8 --bind-to socket \
    python benchmarks/benchmark_phase4.py \
    --json results/phase4_8gpu_a100_pcie.json

mpirun -n 8 --bind-to socket \
    python benchmarks/benchmark_phase5.py \
    --json results/phase5_8gpu_a100_pcie.json
```

---

## 6. Instance Selection Guide

| Goal | Recommended template | Notes |
|---|---|---|
| Phase 3 only (CPU/GPU comparison) | 1× A100 or 1× A10 | No MPI needed |
| Phase 3 + Julia comparison | 1× A100 | Julia adds ~15 min setup |
| Phase 4/5 scaling | 4× or 8× A100 PCIe | PCIe, not NVLink — valid results, higher comm overhead |
| Publication-quality 8-GPU run | 8× A100 PCIe | This instance |

> **Cost note:** vast.ai 8× A100 PCIe instances are typically cheaper than
> Lambda's 8× A100 NVLink nodes.  Phase 4/5 results will differ due to
> PCIe vs NVLink topology — clearly label outputs with the GPU SKU.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `conda: command not found` | conda not sourced | `source /opt/miniforge3/etc/profile.d/conda.sh` |
| `mpirun: command not found` | conda env not active or conda-forge openmpi not installed | `conda activate gaio` then check `which mpirun` |
| `ImportError: libmpi.so` | mpi4py linked to wrong libmpi | Reinstall: `MPICC=$(which mpicc) pip install --no-binary mpi4py mpi4py` |
| `PMPI_Init_thread: Unknown error class` | UCX transport mismatch | `export UCX_TLS=tcp,cuda_copy` (disables CUDA IPC, safe fallback) |
| `ModuleNotFoundError: No module named 'gaio'` | GAIO.py not installed | `pip install -e /workspace/GAIO.py/.[gpu,mpi,dev]` |
| Julia `SIMDExt loaded: false` | Using `--project=/workspace/GAIO.jl` directly | Use `--julia-project ~/gaio_bench_env` (wrapper env, §3c) |
| Julia GPU T_op nnz=0 | `CUDAExt.jl` variable shadowing bug | Apply patch from §3d |
| Julia `MethodError: unsupported keyword argument "show_progress"` | `SIMDExt.construct_transfers` missing keyword | Apply patch from §3d |
| Rank segfaults at MPI_Init | CUDA context before MPI | Call `cuda.select_device(rank)` before any GAIO computation |
| `NumbaPerformanceWarning: Grid size 2` | Problem too small for GPU | Use `--steps 14 --grid-res 4` |
| `ompi_info` shows no `cuda` extension | conda-forge openmpi without CUDA | Phase 4/5 still works via CPU staging; GPUDirect not available on PCIe A100 anyway |

---

## 8. Known GAIO.jl Bugs

**Bug 1 — GPU `construct_transfers` always returns nnz=0** (`CUDAExt.jl`):
Line `codomain = BoxSet(P2, S())` overwrites the `codomain` function parameter with an
empty `BoxSet`, so `hit in codomain.set` is always false.
**Fix:** `sed -i '/codomain = BoxSet(P2, S())/d' /workspace/GAIO.jl/ext/CUDAExt.jl`
(see §3d for the full patch procedure).

**Bug 2 — SIMDExt silently not loaded**: Running `julia --project=/workspace/GAIO.jl`
directly causes a circular init dependency.
**Fix:** use the `~/gaio_bench_env` wrapper environment (§3c).

**Bug 3 — `SIMDExt.construct_transfers` rejects `show_progress` keyword**:
The two SIMDExt `construct_transfers` overloads lack the `show_progress` keyword
that `TransferOperator` passes, causing a `MethodError` and silent benchmark failure.
**Fix:** add `; show_progress::Bool=false, kwargs...` to both signatures
(see §3d for the full patch procedure).
