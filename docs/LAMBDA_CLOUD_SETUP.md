# Lambda Cloud Deployment Guide: GAIO.py Multi-GPU

Step-by-step instructions for deploying GAIO.py on a fresh Lambda Labs
GPU instance and running the Phase 4/5 benchmarks.

**Target instance:**
`gpu_8x_a100` (8× A100 80 GB, NVLink) or `gpu_4x_a10` (4× A10 24 GB).
Smaller instances like `gpu_1x_a10` work for single-GPU Phase 3 benchmarks
only.  All steps assume Ubuntu 22.04 with CUDA 12.x pre-installed by Lambda.

---

## 1. What Lambda Provides Out of the Box

Lambda instances ship with:

- CUDA toolkit + driver (typically CUDA 12.2, driver ≥ 525)
- `nvidia-smi` and `nvcc` on PATH
- Miniconda (at `~/miniconda3`)
- OpenMPI **without** CUDA support — must be recompiled (see §3)

Verify the CUDA version before proceeding:

```bash
nvcc --version
nvidia-smi
```

---

## 2. One-Shot Setup Script

Copy, paste, and run the entire block below in a fresh SSH session.
It takes approximately 15–20 minutes on a `gpu_4x_a10` instance.

```bash
#!/usr/bin/env bash
set -euo pipefail

# ── 0. Variables ─────────────────────────────────────────────────────────────
OMPI_VERSION="4.1.6"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
INSTALL_PREFIX="$HOME/opt/openmpi-${OMPI_VERSION}"
GAIO_REPO="$HOME/GAIO.py"
CONDA_ENV="gaio"

# ── 1. System packages ────────────────────────────────────────────────────────
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    build-essential gfortran wget git \
    libucx-dev libibverbs-dev          # UCX for GPUDirect RDMA

# ── 2. Download and compile CUDA-aware OpenMPI 4.1.x ─────────────────────────
# Lambda's default OpenMPI lacks --with-cuda; this build enables
# GPUDirect RDMA (device buffers passed directly to MPI without staging).
cd /tmp
wget -q "https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OMPI_VERSION}.tar.bz2"
tar -xf "openmpi-${OMPI_VERSION}.tar.bz2"
cd "openmpi-${OMPI_VERSION}"

./configure \
    --prefix="${INSTALL_PREFIX}" \
    --with-cuda="${CUDA_HOME}" \
    --with-ucx=/usr \
    --enable-mpirun-prefix-by-default \
    --disable-static \
    2>&1 | tail -5   # show only the final summary lines

make -j"$(nproc)" install 2>&1 | tail -3

# ── 3. Persistent environment variables ───────────────────────────────────────
cat >> ~/.bashrc << EOF

# GAIO.py: CUDA-aware OpenMPI ${OMPI_VERSION}
export PATH="${INSTALL_PREFIX}/bin:\$PATH"
export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"
export OMPI_MCA_opal_cuda_support=1      # enable GPUDirect at runtime
export UCX_TLS=rc,cuda_copy,cuda_ipc     # UCX transport: IB + CUDA peer
EOF

export PATH="${INSTALL_PREFIX}/bin:$PATH"
export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_opal_cuda_support=1
export UCX_TLS=rc,cuda_copy,cuda_ipc

# Verify the build includes CUDA support
mpirun --version
ompi_info | grep -i "cuda\|Built on"

# ── 4. Conda environment ──────────────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh

conda create -y -n "${CONDA_ENV}" python=3.11 \
    -c conda-forge -c nvidia -c defaults \
    numpy scipy numba cudatoolkit=12.2 \
    matplotlib tqdm jupyter ipykernel \
    pytest pytest-cov line_profiler memory_profiler

conda activate "${CONDA_ENV}"

# ── 5. mpi4py — must be compiled against our custom OpenMPI, NOT conda's ─────
# Point the compiler wrappers at the CUDA-aware build from step 2.
MPICC="${INSTALL_PREFIX}/bin/mpicc" \
    pip install --no-binary mpi4py mpi4py

# ── 6. PETSc / SLEPc — distributed eigensolve (Phase 4 eigendecomposition) ──
# petsc4py / slepc4py from conda-forge link against their own PETSc build
# with MPI support.  They do NOT need to link against our custom OpenMPI
# because they only use MPI for Krylov basis distribution (CPU RAM),
# not for GPU-VRAM Allgatherv.  The conda-forge builds are correct here.
conda install -y -c conda-forge petsc4py slepc4py

# ── 7. Install GAIO.py ────────────────────────────────────────────────────────
cd "${GAIO_REPO}"
pip install -e ".[gpu,mpi,hpc,dev]"

# ── 8. One-rank smoke test ────────────────────────────────────────────────────
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

## 3. Verification Checklist

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

## 4. Running the Benchmarks

### Phase 3 — single-GPU acceleration (baseline, no mpirun needed)

```bash
# ~30 s; shows Python → Numba → CUDA speedup
python benchmarks/benchmark_phase3.py --steps 10 --grid-res 2
```

### Phase 4 — multi-GPU MPI scaling (1-GPU vs N-GPU)

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

## 5. Instance Type Selection Guide

| Benchmark goal | Recommended instance | Notes |
|---|---|---|
| Phase 3 single-GPU | `gpu_1x_a10` | Cheapest; Python/Numba/CUDA comparison |
| Phase 4 scaling | `gpu_4x_a10` or `gpu_4x_a100` | Need ≥4 GPUs on same NVLink fabric |
| Phase 5 max imbalance | `gpu_4x_a100` | Larger A100 VRAM fits deeper subdivision |
| Full suite | `gpu_8x_a100` | Reserve for publication-quality numbers |

For a quick Phase 4 + Phase 5 comparison run (~$3–5 total), `gpu_4x_a10`
with `--steps 14 --grid-res 3 --test-pts 4` is a good balance of cost and
statistical significance.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `PMPI_Init_thread: Unknown error class` | OFI fabric mismatch on Lambda | `export UCX_TLS=tcp,cuda_copy` (disables IB, slower but safe) |
| `mpirun: command not found` | PATH not updated | `source ~/.bashrc` or re-login |
| `ImportError: libmpi.so` | mpi4py linked to wrong libmpi | Reinstall: `MPICC=.../mpicc pip install --no-binary mpi4py mpi4py` |
| `NumbaPerformanceWarning: Grid size 2` | Problem too small for GPU | Use `--steps 14 --grid-res 4 --test-pts 5`; warnings already suppressed in benchmarks |
| RDMA probe returns `NO` | Lambda's network stack lacks UCX IB | Normal on A10 nodes; GPU-to-GPU via `cuda_ipc` still available; set `UCX_TLS=cuda_copy,cuda_ipc` |
| `slepc4py` not found | SLEPc optional dependency absent | `conda install -c conda-forge petsc4py slepc4py`; fallback to scipy ARPACK is automatic |
| Rank segfaults at MPI_Init | CUDA context not created before MPI calls | Call `cuda.select_device(rank())` before any GAIO computation |
