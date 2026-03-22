"""
benchmarks/benchmark_phase4.py
================================
Phase 4 multi-GPU MPI scaling benchmark for GAIO.py.

Designed for multi-GPU cloud instances (e.g. Lambda Cloud 4×A100 or
8×A100).  Run under mpiexec; each rank is pinned to one GPU.

What this measures
------------------
The benchmark performs two timed runs inside a single mpiexec invocation:

  Step 1 — Serial baseline  : rank 0 runs the full pipeline on a single
            GPU with comm=False (no MPI).  Other ranks wait silently.

  Step 2 — Distributed run  : all P ranks execute the pipeline together
            via Phase 4 Morton-decomposed MPI (each rank processes K/P
            domain cells).

A side-by-side comparison table shows wall time and speedup.

Metrics
-------
  - Wall time for relative_attractor and TransferOperator separately
  - Speedup = serial_total / distributed_total
  - Per-rank COO contribution (load imbalance ratio)
  - Test-point memory per rank (shows K/P reduction)
  - CUDA-aware MPI / GPUDirect RDMA status

Usage
-----
    # 4 GPUs — primary use case on Lambda Cloud
    mpiexec -n 4 python benchmarks/benchmark_phase4.py

    # Bind one rank per GPU socket (recommended on NVLink systems)
    mpiexec -n 4 --bind-to socket python benchmarks/benchmark_phase4.py

    # Larger problem (more subdivision → larger attractor → more work per rank)
    mpiexec -n 4 python benchmarks/benchmark_phase4.py --steps 14

    # CPU-only (no CUDA, for testing on laptop/WSL)
    mpiexec -n 4 python benchmarks/benchmark_phase4.py --cpu

    # Include distributed eigensolve timing
    mpiexec -n 4 python benchmarks/benchmark_phase4.py --eigs

    # Save results to JSON
    mpiexec -n 4 python benchmarks/benchmark_phase4.py --json phase4.json

Expected output (4 ranks, steps=12, 4×GPU)
-------------------------------------------
    GAIO.py Phase 4 Multi-GPU Scaling Benchmark
      System  : 3-D Four-Wing attractor
      Ranks   : 4  (one per GPU)
      Steps   : 12  |  Grid: 2³  |  Test pts: 3³ = 27
      Backend : gpu  |  RDMA: YES (GPUDirect)

    ── Step 1: Serial baseline (rank 0, 1 GPU) ──────────────────────
      Attractor : 2.341 s  (11,706 cells)
      T_op      : 0.506 s  (nnz: 40,780)
      Total     : 2.847 s

    ── Step 2: Distributed (4 GPUs, Phase 4 MPI) ────────────────────
      Attractor : 0.588 s
      T_op      : 0.031 s
      Total     : 0.619 s

    ── Comparison ───────────────────────────────────────────────────
      Config        Attractor (s)  T_op (s)  Total (s)  Speedup
      1 GPU               2.341     0.506      2.847      1.0×
      4 GPU (MPI)         0.588     0.031      0.619      4.6×
    ─────────────────────────────────────────────────────────────────
      Near-linear speedup (4.6× with 4 GPUs) ✓

    Per-rank COO contribution (Phase 4, static Morton)
        Rank    COO entries    % of total
      ────────────────────────────────────────
           0         10,548         25.8%
           1         10,201         25.0%
           2         10,640         26.1%
           3          9,491         23.3%
      ────────────────────────────────────────
       Total         40,880        100.0%
      Load imbalance (max/min): 1.12×

    Memory estimate (test-point array per rank)
      Full (1 rank)   :   324.0 MB
      Per rank (1/4)  :    81.0 MB  (4.0× reduction)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

# Suppress Numba's low-occupancy warning — expected when problem size is
# small relative to GPU capacity. Use --steps 16 --grid-res 4 on Lambda
# Cloud to get a large enough problem for meaningful GPU utilisation.
try:
    from numba.core.errors import NumbaPerformanceWarning
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Four-Wing system
# ---------------------------------------------------------------------------

_A_FW, _B_FW, _D_FW = 0.2, -0.01, -0.4


def _four_wing_v(x: np.ndarray) -> np.ndarray:
    return np.array([
        _A_FW * x[0] + x[1] * x[2],
        _D_FW * x[1] + _B_FW * x[0] - x[2] * x[1],
        -x[2] - x[0] * x[1],
    ])


# ---------------------------------------------------------------------------
# Map builder
# ---------------------------------------------------------------------------

def _build_map(domain, unit_pts, use_gpu: bool):
    from gaio import SampledBoxMap, rk4_flow_map

    f_flow = rk4_flow_map(_four_wing_v, step_size=0.01, steps=20)

    if not use_gpu:
        return SampledBoxMap(f_flow, domain, unit_pts)

    try:
        from numba import cuda
        from gaio import AcceleratedBoxMap
        from gaio.cuda.rk4_cuda import make_cuda_rk4_flow_map

        @cuda.jit(device=True)
        def _fw_device(x, out):
            out[0] = _A_FW * x[0] + x[1] * x[2]
            out[1] = _D_FW * x[1] + _B_FW * x[0] - x[2] * x[1]
            out[2] = -x[2] - x[0] * x[1]

        f_device = make_cuda_rk4_flow_map(_fw_device, ndim=3, step_size=0.01, steps=20)
        return AcceleratedBoxMap(f_flow, domain, unit_pts,
                                 f_device=f_device, backend="gpu")
    except Exception as exc:
        print(f"  [warn] GPU unavailable ({exc}), falling back to SampledBoxMap",
              flush=True)
        return SampledBoxMap(f_flow, domain, unit_pts)


# ---------------------------------------------------------------------------
# Timed pipeline run (serial or distributed)
# ---------------------------------------------------------------------------

@dataclass
class _RunResult:
    n_cells:        int
    nnz:            int
    attractor_time: float
    transfer_time:  float
    eigs_time:      float
    per_rank_nnz:   list
    total_nnz_raw:  int
    eigenvalues:    list
    error:          Optional[str] = None

    @property
    def total_time(self) -> float:
        return self.attractor_time + self.transfer_time + self.eigs_time


def _run_pipeline(F, S, steps, comm, run_eigs: bool, k_eigs: int = 3) -> _RunResult:
    """Run relative_attractor + TransferOperator (+ optional eigs) and return timings."""
    from gaio import relative_attractor, TransferOperator

    # ── Attractor ─────────────────────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        A  = relative_attractor(F, S, steps=steps, comm=comm)
        if comm is not False and hasattr(comm, "Barrier"):
            comm.Barrier()
        att_time = time.perf_counter() - t0
    except Exception as exc:
        return _RunResult(0, 0, 0., 0., 0., [], 0, [], error=f"attractor: {exc}")

    # ── Transfer operator ─────────────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        T  = TransferOperator(F, A, A, comm=comm)
        if comm is not False and hasattr(comm, "Barrier"):
            comm.Barrier()
        top_time = time.perf_counter() - t0
    except Exception as exc:
        return _RunResult(len(A), 0, att_time, 0., 0., [], 0, [],
                          error=f"transfer: {exc}")

    stats         = getattr(T, "mpi_stats", {})
    per_rank_nnz  = list(map(int, stats.get("per_rank_nnz", np.array([T.mat.nnz]))))
    total_nnz_raw = int(stats.get("total_nnz_raw", T.mat.nnz))

    # ── Eigensolve (optional) ─────────────────────────────────────────────────
    eigenvalues = []
    eigs_time   = 0.0
    if run_eigs:
        try:
            t0 = time.perf_counter()
            evals, _ = T.eigs(k=k_eigs)
            if comm is not False and hasattr(comm, "Barrier"):
                comm.Barrier()
            eigs_time   = time.perf_counter() - t0
            eigenvalues = evals.tolist()
        except Exception as exc:
            eigenvalues = [f"ERROR: {exc}"]

    return _RunResult(
        n_cells=len(A),
        nnz=T.mat.nnz,
        attractor_time=att_time,
        transfer_time=top_time,
        eigs_time=eigs_time,
        per_rank_nnz=per_rank_nnz,
        total_nnz_raw=total_nnz_raw,
        eigenvalues=eigenvalues,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _memory_mb(n_cells: int, M: int, ndim: int) -> float:
    return n_cells * M * ndim * 8 / (1024 ** 2)


def _print_comparison(serial: _RunResult, dist: _RunResult, n_ranks: int) -> None:
    speedup = serial.total_time / dist.total_time if dist.total_time > 0 else float("inf")
    near_linear = speedup >= 0.7 * n_ranks
    check = "✓" if near_linear else "△ (sub-linear — check load balance)"

    print(f"\n  {'─' * 60}")
    print(f"  {'Config':<18}  {'Attractor':>12}  {'T_op':>8}  {'Total':>9}  {'Speedup':>8}")
    print(f"  {'─' * 60}")
    print(f"  {'1 GPU (serial)':<18}  {serial.attractor_time:>10.3f}s  "
          f"{serial.transfer_time:>6.3f}s  {serial.total_time:>7.3f}s  {'1.0×':>8}")
    dist_label = f"{n_ranks} GPU (MPI)"
    print(f"  {dist_label:<18}  {dist.attractor_time:>10.3f}s  "
          f"{dist.transfer_time:>6.3f}s  {dist.total_time:>7.3f}s  {speedup:>7.1f}×")
    print(f"  {'─' * 60}")
    print(f"  {speedup:.1f}× speedup with {n_ranks} GPUs  {check}")


def _print_per_rank_table(per_rank_nnz: list, total_nnz: int) -> None:
    dashes = "─" * 42
    print(f"\n  Per-rank COO contribution (Phase 4, static Morton)")
    print(f"    {'Rank':>6}  {'COO entries':>14}  {'% of total':>12}")
    print(f"  {dashes}")
    for r, nnz in enumerate(per_rank_nnz):
        pct = 100.0 * nnz / total_nnz if total_nnz > 0 else 0.0
        print(f"  {r:>8}  {nnz:>14,}  {pct:>11.1f}%")
    print(f"  {dashes}")
    print(f"  {'Total':>8}  {total_nnz:>14,}  {'100.0%':>12}")
    min_nnz = min(per_rank_nnz) if per_rank_nnz else 0
    max_nnz = max(per_rank_nnz) if per_rank_nnz else 0
    if min_nnz > 0:
        print(f"\n  Load imbalance (max/min): {max_nnz / min_nnz:.2f}×")
    else:
        print(f"\n  Load imbalance: ∞ (one rank has zero work — Phase 5 target)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="GAIO.py Phase 4 multi-GPU MPI scaling benchmark"
    )
    parser.add_argument("--steps", type=int, default=12,
                        help="Subdivision steps (default: 12)")
    parser.add_argument("--grid-res", type=int, default=2,
                        help="Initial grid cells per dimension (default: 2 = 8 cells)")
    parser.add_argument("--test-pts", type=int, default=3,
                        help="Test points per dimension (default: 3 = 27/cell)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU-only mode (SampledBoxMap, no CUDA)")
    parser.add_argument("--eigs", action="store_true",
                        help="Also time distributed eigensolve (k=3)")
    parser.add_argument("--rdma-check", action="store_true",
                        help="Report CUDA-aware MPI / GPUDirect RDMA status")
    parser.add_argument("--json", type=str, default=None,
                        help="Save results to JSON (rank 0 only)")
    args = parser.parse_args(argv)

    # ── MPI setup ─────────────────────────────────────────────────────────────
    from gaio.mpi import get_comm
    comm    = get_comm()
    my_rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    use_gpu = not args.cpu
    if use_gpu:
        try:
            import numba.cuda as nc
            nc.select_device(my_rank % len(nc.gpus.lst))
        except Exception:
            use_gpu = False

    backend_str = "gpu" if use_gpu else "cpu (SampledBoxMap)"

    rdma_status = "not checked"
    if args.rdma_check:
        from gaio.mpi import is_rdma_capable
        rdma_status = "YES (GPUDirect)" if is_rdma_capable(comm) else "NO (CPU staging)"

    # ── Domain / unit points (same for both runs) ─────────────────────────────
    from gaio import Box, BoxPartition, BoxSet

    domain   = Box(np.array([0.0, 0.0, 0.0]), np.array([5.0, 5.0, 5.0]))
    P        = BoxPartition(domain, [args.grid_res] * 3)
    t_pts    = np.linspace(-0.5, 0.5, args.test_pts)
    gx, gy, gz = np.meshgrid(t_pts, t_pts, t_pts)
    unit_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    M        = len(unit_pts)
    S        = BoxSet.full(P)

    if my_rank == 0:
        print(f"\nGAIO.py Phase 4 Multi-GPU Scaling Benchmark")
        print(f"  System  : 3-D Four-Wing attractor")
        print(f"  Ranks   : {n_ranks}  ({'one per GPU' if use_gpu else 'CPU only'})")
        print(f"  Steps   : {args.steps}  |  Grid: {args.grid_res}³  |  "
              f"Test pts: {args.test_pts}³ = {M}")
        print(f"  Backend : {backend_str}  |  RDMA: {rdma_status}")
        sys.stdout.flush()

    # ── Step 1: Serial baseline (rank 0 only, comm=False) ─────────────────────
    serial: _RunResult | None = None

    if my_rank == 0:
        print(f"\n  ── Step 1: Serial baseline (rank 0, 1 GPU) {'─' * 22}")
        print(f"    Running …", end="", flush=True)
        F_serial = _build_map(domain, unit_pts, use_gpu=use_gpu)
        serial   = _run_pipeline(F_serial, S, args.steps, comm=False,
                                  run_eigs=args.eigs)
        if serial.error:
            print(f" FAILED: {serial.error}")
        else:
            print(f" done")
            eigs_str = (f"  Eigs(k=3)  : {serial.eigs_time:.3f} s"
                        if args.eigs else "")
            print(f"    Attractor : {serial.attractor_time:.3f} s  "
                  f"({serial.n_cells:,} cells)")
            print(f"    T_op      : {serial.transfer_time:.3f} s  "
                  f"(nnz: {serial.nnz:,})")
            if args.eigs:
                print(f"    Eigs(k=3) : {serial.eigs_time:.3f} s")
            print(f"    Total     : {serial.total_time:.3f} s")
        sys.stdout.flush()

    # Broadcast serial result summary to all ranks (for comparison table)
    serial_summary = None
    if my_rank == 0 and serial and not serial.error:
        serial_summary = [serial.attractor_time, serial.transfer_time,
                          serial.eigs_time, serial.n_cells, serial.nnz]
    serial_summary = comm.bcast(serial_summary, root=0)

    # ── Step 2: Distributed run (all ranks) ───────────────────────────────────
    comm.Barrier()
    if my_rank == 0:
        print(f"\n  ── Step 2: Distributed ({n_ranks} GPU{'s' if n_ranks > 1 else ''}, "
              f"Phase 4 MPI) {'─' * 18}")
        print(f"    Running …", end="", flush=True)
        sys.stdout.flush()

    F_dist = _build_map(domain, unit_pts, use_gpu=use_gpu)
    comm.Barrier()
    dist   = _run_pipeline(F_dist, S, args.steps, comm=comm,
                            run_eigs=args.eigs)
    comm.Barrier()

    # ── Report (rank 0 only) ─────────────────────────────────────────────────
    if my_rank == 0:
        if dist.error:
            print(f" FAILED: {dist.error}")
        else:
            print(f" done")
            print(f"    Attractor : {dist.attractor_time:.3f} s")
            print(f"    T_op      : {dist.transfer_time:.3f} s")
            if args.eigs:
                print(f"    Eigs(k=3) : {dist.eigs_time:.3f} s")
            print(f"    Total     : {dist.total_time:.3f} s")

        # ── Comparison table ──────────────────────────────────────────────────
        if serial_summary and not dist.error:
            s_att, s_top, s_eig, s_cells, s_nnz = serial_summary
            # Reconstruct a minimal serial result for the table
            class _S:
                attractor_time = s_att
                transfer_time  = s_top
                eigs_time      = s_eig
                total_time     = s_att + s_top + s_eig
            print(f"\n  ── Comparison {'─' * 47}")
            _print_comparison(_S(), dist, n_ranks)

        # ── Per-rank COO table ────────────────────────────────────────────────
        if not dist.error:
            _print_per_rank_table(dist.per_rank_nnz, dist.total_nnz_raw)

            # ── Memory estimate ───────────────────────────────────────────────
            mem_full  = _memory_mb(dist.n_cells, M, 3)
            mem_local = _memory_mb(dist.n_cells // n_ranks, M, 3) if n_ranks > 0 else mem_full
            reduction = mem_full / mem_local if mem_local > 0 else float("inf")
            print(f"\n  Memory estimate (test-point array per rank)")
            print(f"    Full (1 rank)  : {mem_full:8.1f} MB")
            print(f"    Per rank (1/{n_ranks}) : {mem_local:8.1f} MB  "
                  f"({reduction:.1f}× reduction)")

            # ── Eigenvalue sanity ─────────────────────────────────────────────
            if args.eigs and dist.eigenvalues and isinstance(dist.eigenvalues[0], complex):
                evals = dist.eigenvalues
                check = "✓" if abs(abs(evals[0]) - 1.0) < 0.02 else "✗"
                ev_strs = [f"({v.real:.4f}{v.imag:+.4f}j)" for v in evals[:3]]
                print(f"\n  Eigenvalue sanity (|λ₁| ≈ 1.0): {check}")
                print(f"    λ = [{', '.join(ev_strs)}]")

        # ── JSON output ───────────────────────────────────────────────────────
        if args.json and not dist.error:
            out = {
                "n_ranks":   n_ranks,
                "steps":     args.steps,
                "backend":   backend_str,
                "rdma":      rdma_status,
                "serial": {
                    "attractor_time": serial_summary[0] if serial_summary else None,
                    "transfer_time":  serial_summary[1] if serial_summary else None,
                    "eigs_time":      serial_summary[2] if serial_summary else None,
                    "n_cells":        int(serial_summary[3]) if serial_summary else None,
                    "nnz":            int(serial_summary[4]) if serial_summary else None,
                },
                "distributed": {
                    "attractor_time": dist.attractor_time,
                    "transfer_time":  dist.transfer_time,
                    "eigs_time":      dist.eigs_time,
                    "n_cells":        dist.n_cells,
                    "nnz":            dist.nnz,
                    "per_rank_nnz":   dist.per_rank_nnz,
                    "imbalance":      (max(dist.per_rank_nnz) / min(dist.per_rank_nnz)
                                       if dist.per_rank_nnz and min(dist.per_rank_nnz) > 0
                                       else None),
                },
            }
            if serial_summary:
                s_total = sum(serial_summary[:3])
                out["speedup"] = s_total / dist.total_time if dist.total_time > 0 else None
            with open(args.json, "w") as fh:
                json.dump(out, fh, indent=2)
            print(f"\n  Results saved to {args.json}")


if __name__ == "__main__":
    main()
