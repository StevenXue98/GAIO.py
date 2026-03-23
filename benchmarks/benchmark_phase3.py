"""
benchmarks/benchmark_phase3.py
================================
Phase 3 backend comparison: pure Python vs Numba CPU vs single CUDA GPU.

Measures the two main computational phases of the GAIO pipeline for the
3-D Four-Wing attractor:

    Phase A — Attractor computation  : ``relative_attractor(F, S, steps=N)``
    Phase B — Transfer operator      : ``TransferOperator(F, A, A)``

across three single-process backends:

    python   — pure Python loop (SampledBoxMap, no Numba)   ← baseline
    cpu      — Numba njit parallel (AcceleratedBoxMap, backend='cpu')
    gpu      — Single CUDA GPU (AcceleratedBoxMap, backend='gpu')

Purpose
-------
Demonstrates the Phase 3 acceleration hierarchy (Python → Numba → CUDA)
on a small, fixed problem.  Kept small so the pure-Python baseline
finishes in under ~30 s on a laptop.

    Run as a single process — NOT under mpiexec.

For MPI multi-GPU scaling benchmarks see ``benchmark_phase4.py``.
For load-imbalance / Phase 5 motivation see ``benchmark_imbalance.py``.

Usage
-----
    # Default: python + cpu + gpu, steps=8, 2³ initial grid
    python benchmarks/benchmark_phase3.py

    # Larger problem (python backend will be slow)
    python benchmarks/benchmark_phase3.py --steps 12 --grid-res 4

    # Save results to JSON
    python benchmarks/benchmark_phase3.py --json results.json

    # Run only specific backends
    python benchmarks/benchmark_phase3.py --backends python cpu

Output
------
    | Backend        |   Cells |  Map (s) | T_op (s) | Total (s) | Speedup |     nnz |
    |----------------|---------|----------|----------|-----------|---------|---------|
    | python         |   1,268 |    4.796 |    2.077 |     6.873 |    1.0× |   4,590 |
    | cpu            |     226 |    0.567 |    0.002 |     0.569 |   12.1× |     526 |
    | gpu            |   1,268 |    0.875 |    0.004 |     0.879 |    7.8× |   4,590 |

(The `python` row is always the speedup baseline.)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional

_RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"


def _json_path(name: str) -> pathlib.Path:
    """Resolve a JSON filename to the results/ directory if no dir is given."""
    p = pathlib.Path(name)
    return _RESULTS_DIR / p if p.parent == pathlib.Path(".") else p

import numpy as np

# Suppress Numba's low-occupancy warning — expected for small benchmark
# problem sizes where the test-point array is too small to fill the GPU.
try:
    from numba.core.errors import NumbaPerformanceWarning
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Four-wing system
# ---------------------------------------------------------------------------

A_FW, B_FW, D_FW = 0.2, -0.01, -0.4


def _four_wing_v(x: np.ndarray) -> np.ndarray:
    return np.array([
        A_FW * x[0] + x[1] * x[2],
        D_FW * x[1] + B_FW * x[0] - x[2] * x[1],
        -x[2] - x[0] * x[1],
    ])


def _unit_pts(res: int = 4) -> np.ndarray:
    """Grid of test points matching GAIO.jl GridBoxMap: k*(2/n)-1 for k=0..n-1."""
    t = np.arange(res, dtype=float) * (2.0 / res) - 1.0
    gx, gy, gz = np.meshgrid(t, t, t, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    backend:      str
    n_ranks:      int
    n_cells:      int           # attractor cells
    map_time:     float         # relative_attractor wall time (s)
    transfer_time: float        # TransferOperator wall time (s)
    total_time:   float         # map_time + transfer_time
    nnz:          int           # T.mat.nnz
    error:        Optional[str] = None

    @property
    def label(self) -> str:
        if self.n_ranks > 1:
            return f"{self.backend} ({self.n_ranks})"
        return self.backend


# ---------------------------------------------------------------------------
# Map factories
# ---------------------------------------------------------------------------

def _make_python_map(domain, unit_pts):
    from gaio import SampledBoxMap, rk4_flow_map
    f = rk4_flow_map(_four_wing_v, step_size=0.01, steps=20)
    return SampledBoxMap(f, domain, unit_pts)


def _make_cpu_map(domain, unit_pts):
    from numba import njit
    from gaio import rk4_flow_map, AcceleratedBoxMap
    from gaio.maps.rk4 import make_njit_rk4_flow_map

    @njit
    def _fw_vfield(x):
        return np.array([
            A_FW * x[0] + x[1] * x[2],
            D_FW * x[1] + B_FW * x[0] - x[2] * x[1],
            -x[2] - x[0] * x[1],
        ])

    f_jit = make_njit_rk4_flow_map(_fw_vfield, step_size=0.01, steps=20)
    f_cpu = rk4_flow_map(_four_wing_v, step_size=0.01, steps=20)
    return AcceleratedBoxMap(f_cpu, domain, unit_pts,
                             f_jit=f_jit, backend="cpu")


def _make_gpu_map(domain, unit_pts, dtype=None):
    from numba import cuda
    from gaio import rk4_flow_map, AcceleratedBoxMap
    from gaio.cuda.rk4_cuda import make_cuda_rk4_flow_map

    @cuda.jit(device=True)
    def _fw_device(x, out):
        out[0] = A_FW * x[0] + x[1] * x[2]
        out[1] = D_FW * x[1] + B_FW * x[0] - x[2] * x[1]
        out[2] = -x[2] - x[0] * x[1]

    f_cpu    = rk4_flow_map(_four_wing_v, step_size=0.01, steps=20)
    f_device = make_cuda_rk4_flow_map(_fw_device, ndim=3, step_size=0.01, steps=20)
    return AcceleratedBoxMap(f_cpu, domain, unit_pts,
                             f_device=f_device, backend="gpu", dtype=dtype)


# ---------------------------------------------------------------------------
# Single-backend runner
# ---------------------------------------------------------------------------

def _run_backend(
    backend_name: str,
    domain,
    unit_pts,
    P,
    steps: int,
    comm=None,
    gpu_dtype=None,
) -> BenchResult:
    """Run the full pipeline for one backend; return a BenchResult."""
    from gaio import BoxSet, relative_attractor, TransferOperator

    n_ranks = 1 if comm is None else comm.Get_size()

    # Build map
    try:
        if backend_name == "python":
            F = _make_python_map(domain, unit_pts)
        elif backend_name == "cpu":
            F = _make_cpu_map(domain, unit_pts)
        elif backend_name in ("gpu", "mpi-gpu"):
            F = _make_gpu_map(domain, unit_pts, dtype=gpu_dtype)
        else:
            raise ValueError(f"Unknown backend: {backend_name}")
    except Exception as exc:
        return BenchResult(backend_name, n_ranks, 0, 0., 0., 0., 0,
                           error=str(exc))

    S = BoxSet.full(P)

    # ── Phase A: attractor ────────────────────────────────────────────────────
    if comm is not None:
        comm.Barrier()
    t0 = time.perf_counter()
    try:
        A = relative_attractor(F, S, steps=steps)
    except Exception as exc:
        return BenchResult(backend_name, n_ranks, 0, 0., 0., 0., 0,
                           error=f"attractor: {exc}")
    if comm is not None:
        comm.Barrier()
    t1 = time.perf_counter()
    map_time = t1 - t0

    # ── Phase B: transfer operator ────────────────────────────────────────────
    if comm is not None:
        comm.Barrier()
    t0 = time.perf_counter()
    try:
        T = TransferOperator(F, A, A, comm=comm)
    except Exception as exc:
        return BenchResult(backend_name, n_ranks, len(A), map_time, 0., map_time, 0,
                           error=f"transfer: {exc}")
    if comm is not None:
        comm.Barrier()
    t1 = time.perf_counter()
    transfer_time = t1 - t0

    return BenchResult(
        backend=backend_name,
        n_ranks=n_ranks,
        n_cells=len(A),
        map_time=map_time,
        transfer_time=transfer_time,
        total_time=map_time + transfer_time,
        nnz=T.mat.nnz,
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _format_table(results: list[BenchResult]) -> str:
    # Baseline for speedup = first successful python result, else first result
    baseline_time = None
    for r in results:
        if r.error is None and r.backend == "python":
            baseline_time = r.total_time
            break
    if baseline_time is None:
        for r in results:
            if r.error is None:
                baseline_time = r.total_time
                break

    header = (
        f"| {'Backend':<14} | {'Cells':>8} | {'Map (s)':>10} "
        f"| {'T_op (s)':>10} | {'Total (s)':>10} | {'Speedup':>8} | {'nnz':>10} |"
    )
    sep = "|" + "|".join(["-" * (w + 2) for w in [14, 8, 10, 10, 10, 8, 10]]) + "|"

    lines = [header, sep]
    for r in results:
        if r.error:
            lines.append(
                f"| {r.label:<14} | {'N/A':>8} | {'N/A':>10} "
                f"| {'N/A':>10} | {'N/A':>10} | {'N/A':>8} | {r.error[:30]:>10} |"
            )
        else:
            spd = f"{baseline_time / r.total_time:.1f}×" if baseline_time else "—"
            lines.append(
                f"| {r.label:<14} | {r.n_cells:>8,} | {r.map_time:>10.3f} "
                f"| {r.transfer_time:>10.3f} | {r.total_time:>10.3f} "
                f"| {spd:>8} | {r.nnz:>10,} |"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="GAIO.py backend benchmark — 3-D Four-Wing attractor"
    )
    parser.add_argument("--steps", type=int, default=8,
                        help="Subdivision steps (default: 8; keep small for fast Python baseline)")
    parser.add_argument("--grid-res", type=int, default=2,
                        help="Grid cells per dimension (default: 2 = 8 initial cells for 3-D)")
    parser.add_argument("--test-pts", type=int, default=4,
                        help="Test points per dimension (default: 4 = 64 pts/cell, matches GAIO.jl GridBoxMap default)")
    parser.add_argument("--backends", nargs="+",
                        choices=["python", "cpu", "gpu"],
                        default=["python", "cpu", "gpu"],
                        help="Backends to benchmark")
    parser.add_argument("--float32", action="store_true",
                        help="Force float32 on GPU kernel (default: auto-detect; A100 uses float64)")
    parser.add_argument("--json", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args(argv)

    from gaio import Box, BoxPartition, cuda_available, numba_available

    center = np.array([0.0, 0.0, 0.0])
    radius = np.array([5.0, 5.0, 5.0])
    domain = Box(center, radius)
    P      = BoxPartition(domain, [args.grid_res] * 3)
    upts   = _unit_pts(args.test_pts)

    backends = list(args.backends)

    # Guard: this benchmark is single-process only
    try:
        from gaio.mpi import is_mpi_active, rank as mpi_rank
        if is_mpi_active():
            if mpi_rank() == 0:
                print("[benchmark_phase3] WARNING: detected mpiexec with "
                      f"{__import__('gaio.mpi', fromlist=['size']).size()} ranks.\n"
                      "  This benchmark is single-process only — each rank runs "
                      "independently and will print duplicate tables.\n"
                      "  Run as:  python benchmarks/benchmark_phase3.py\n"
                      "  For multi-GPU MPI scaling: mpiexec -n P python benchmarks/benchmark_phase4.py\n")
            else:
                return  # silence non-zero ranks
    except Exception:
        pass

    print(f"\nGAIO.py Phase 3 Benchmark  (Python / Numba / CUDA, single process)")
    print(f"  System   : 3-D Four-Wing attractor")
    print(f"  Steps    : {args.steps}  |  Grid: {args.grid_res}³  |  Test pts: {args.test_pts}³ = {len(upts)}")
    print(f"  Backends : {', '.join(backends)}")
    print(f"  (For MPI multi-GPU scaling: mpiexec -n P python benchmarks/benchmark_phase4.py)")
    print()

    results: list[BenchResult] = []

    for backend in backends:
        # Availability checks
        if backend == "cpu" and not numba_available():
            results.append(BenchResult(backend, 1, 0, 0., 0., 0., 0,
                                        error="Numba not installed"))
            continue
        if backend == "gpu" and not cuda_available():
            results.append(BenchResult(backend, 1, 0, 0., 0., 0., 0,
                                        error="CUDA not available"))
            continue

        print(f"  Running '{backend}' …", end="", flush=True)
        gpu_dtype = np.float32 if args.float32 else None
        result = _run_backend(backend, domain, upts, P, args.steps, comm=None,
                              gpu_dtype=gpu_dtype)
        results.append(result)

        if result.error:
            print(f" FAILED: {result.error}")
        else:
            print(f" {result.total_time:.3f} s  ({result.n_cells:,} cells)")

    print()
    print("## Results\n")
    print(_format_table(results))
    print()
    print("*Speedup relative to 'python' baseline.*")
    print("*Map time = relative_attractor; T_op time = TransferOperator.*")

    if args.json:
        out = _json_path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump([asdict(r) for r in results], fh, indent=2)
        print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
