# GAIO.py

**GAIO.py** is a Python library for set-oriented analysis of dynamical systems. It is a port of [GAIO.jl](https://github.com/gaioguys/GAIO.jl) by Herwig, Junge, and Dellnitz.

## Why GAIO.py

There are two main reasons for developing GAIO.py as a personal project.
- **Performance parity with Julia.** By leveraging Numba compilation, GAIO.py delivers performance on par with GAIO.jl, a modern HPC language, from pure Python.
- **Multi-GPU scaling.** GAIO.py extends GAIO.jl with MPI-based multi-GPU support. A single `mpiexec -n N` command distributes the domain across N GPUs.

For implementation details, design decisions, and MPI/GPU architecture see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Benchmarks

3D Four-Wing attractor, 4³ = 64 test points per cell.

### Phase 3 — Single GPU vs GAIO.jl

Run: `python benchmarks/benchmark_vs_julia.py --steps 10 --grid-res 4`

| Impl | Threads | Cells | Map (s) | T\_op (s) | Total (s) | nnz | Speedup |
|------|---------|-------|---------|-----------|-----------|-----|---------|
| **numba** | 24 | 24,646 | 0.499 | 0.292 | **0.790** | 145,785 | 1.00× |
| **numba-cuda** | 1 | 24,646 | 0.078 | 0.106 | **0.184** | 145,785 | 4.29× |
| julia-simd† | 24 | 24,646 | 0.768 | 0.397 | 1.165 | 145,786 | 0.68× |
| julia-cuda‡ | 24 | 24,646 | 0.167 | 0.107 | 0.274 | 145,786 | 2.89× |

CPU: numba is **1.47× faster** than julia-simd.
GPU: numba-cuda is **1.49× faster** than julia-cuda (julia-cuda T\_op uses float32; numba-cuda uses float64).
Hardware: AMD Ryzen 9 7845HX (24 threads), NVIDIA RTX 4080 Laptop GPU (12 GB), WSL2.

†julia-simd: run via a custom wrapper environment with a one-line fix for a `show_progress` keyword bug that prevented `SIMDExt` from loading. The fix only improved Julia's CPU performance.
‡julia-cuda: run with a one-line fix for a variable-shadowing bug in `CUDAExt.jl` that caused the GPU `TransferOperator` to silently return an empty matrix. The fix only improved Julia's GPU performance.

### Phase 4 — Multi-GPU MPI scaling

Run `mpiexec -n N python benchmarks/benchmark_phase4.py --steps 16 --grid-res 4`.

| Ranks | Cells | T\_op (s) | Total (s) | Speedup |
|-------|-------|-----------|-----------|---------|
| 1 | — | — | — | 1.0× |
| 4 | — | — | — | — |
| 8 | — | — | — | — |

### Phase 5 — Load-balanced MPI (Lozi attractor)

Run `mpiexec -n N python benchmarks/benchmark_phase5.py`.

| Decomposition | Cells | T\_op (s) | COO imbalance |
|---------------|-------|-----------|---------------|
| static Morton | — | — | ~4.5× (moderate) / ~30× (extreme) |
| weighted Morton | — | — | ~1.0× |

*In Progress*

---

## Setup

### Install

```bash
# Full environment (recommended — includes CUDA, MPI, dev tools)
conda env create -f environment.yml
conda activate gaio
pip install -e ".[gpu,mpi,dev]"
```

CPU-only (no conda required):
```bash
pip install -e .
```

### Run benchmarks

```bash
# Phase 3: GAIO.py vs GAIO.jl (requires Julia — see docs/VAST_AI_SETUP.md §3)
python benchmarks/benchmark_vs_julia.py --steps 10 --grid-res 4

# Phase 4: multi-GPU MPI scaling
mpiexec -n 8 python benchmarks/benchmark_phase4.py --steps 16 --grid-res 4

# Phase 5: load balancing
mpiexec -n 8 python benchmarks/benchmark_phase5.py
```

### Run examples

```bash
python -m gaio.examples.four_wing           # 3D Four-Wing attractor (GPU showcase)
python -m gaio.examples.attractor           # Hénon attractor
python -m gaio.examples.invariant_measure_1d
python -m gaio.examples.almost_invariant_sets
```

### Cloud deployment

For multi-GPU instances (vast.ai or Lambda Cloud), see:
- [docs/VAST_AI_SETUP.md](docs/VAST_AI_SETUP.md)
- [docs/LAMBDA_CLOUD_SETUP.md](docs/LAMBDA_CLOUD_SETUP.md)

---

## References

GAIO.py is a Python port of [GAIO.jl](https://github.com/gaioguys/GAIO.jl). All set-oriented algorithms — subdivision, transfer operator construction, invariant measure computation, Morse decomposition — are directly based on their work.

> Herwig, A., Junge, O., and Dellnitz, M. (2025). *GAIO.jl — A concise Julia package for the global analysis of dynamical systems.* Journal of Open Source Software, 10(116), 9266. https://doi.org/10.21105/joss.09266

> Dellnitz, M. and Junge, O. (2002). *Set Oriented Numerical Methods for Dynamical Systems.* In Handbook of Dynamical Systems, Vol. 2. Elsevier.
