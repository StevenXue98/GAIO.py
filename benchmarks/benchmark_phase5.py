"""
benchmarks/benchmark_phase5.py
================================
Phase 5 benchmark: static Morton decomposition (Phase 4) vs. adaptive
weighted decomposition (Phase 5) on non-uniform attractors.

The benchmark runs the same TransferOperator construction twice on the
same attractor:

    Frame 0 — Phase 4 path  : ``partition_weights=None`` (uniform K/P split)
    Frame 1 — Phase 5 path  : ``partition_weights=T0.partition_weights``
                               (weighted prefix-sum split, balanced by density)

For each frame it records:
    - Wall time for TransferOperator construction
    - Per-rank COO contributions (``mpi_stats["per_rank_nnz"]``)
    - Load imbalance ratio (max / min per-rank nnz)

Then prints:
    - Side-by-side comparison table
    - Imbalance reduction (%)
    - Speedup achieved by Phase 5 (time[0] / time[1])

For near-uniform attractors (e.g., Four-Wing) Phase 5 overhead is small
and speedup approaches 1.0 — this is the correct "no regression" case.
For highly non-uniform attractors (Hénon, Ikeda) Phase 5 should reduce
imbalance by 50–90% and deliver 1.5–5× T_op speedup.

Run as a single mpiexec invocation:

    mpiexec -n 4 python benchmarks/benchmark_phase5.py --map henon --steps 12
    mpiexec -n 4 python benchmarks/benchmark_phase5.py --map ikeda --steps 10
    mpiexec -n 4 python benchmarks/benchmark_phase5.py --map fourwing --steps 8

    # Run all three maps in sequence
    mpiexec -n 4 python benchmarks/benchmark_phase5.py --all-maps

    # Save results to JSON
    mpiexec -n 4 python benchmarks/benchmark_phase5.py --map henon --json results.json

GPU usage (bind one rank per GPU):
    mpiexec -n 4 --bind-to socket python benchmarks/benchmark_phase5.py --gpu --map henon --steps 14
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

# Suppress Numba low-occupancy warning for small benchmark problems
try:
    from numba.core.errors import NumbaPerformanceWarning
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Map definitions
# ─────────────────────────────────────────────────────────────────────────────

# Hénon map: a=1.4, b=0.3.  Horseshoe attractor — extreme imbalance target.
_HENON_A, _HENON_B = 1.4, 0.3

def _henon(x: np.ndarray) -> np.ndarray:
    return np.array([1.0 - _HENON_A * x[0]**2 + x[1], _HENON_B * x[0]])


# Ikeda map: μ=0.9.  Off-centre spiral — moderate imbalance target.
_IKEDA_MU = 0.9

def _ikeda(x: np.ndarray) -> np.ndarray:
    t = 0.4 - 6.0 / (1.0 + x[0]**2 + x[1]**2)
    c, s = np.cos(t), np.sin(t)
    return np.array([
        1.0 + _IKEDA_MU * (x[0] * c - x[1] * s),
              _IKEDA_MU * (x[0] * s + x[1] * c),
    ])


# Four-Wing ODE: near-uniform control case.
_A_FW, _B_FW, _D_FW = 0.2, -0.01, -0.4

def _four_wing_v(x: np.ndarray) -> np.ndarray:
    return np.array([
        _A_FW * x[0] + x[1] * x[2],
        _D_FW * x[1] + _B_FW * x[0] - x[2] * x[1],
        -x[2] - x[0] * x[1],
    ])


# Lozi map: a=1.7, b=0.5.  Piecewise-linear analogue of Hénon.
# The attractor is a thin angular filament concentrated in one
# quadrant of the domain — the Morton split puts most hot cells
# on one or two ranks, producing dramatic (5–30×) imbalance at
# moderate K when the domain is large relative to the attractor.
_LOZI_A, _LOZI_B = 1.7, 0.5

def _lozi(x: np.ndarray) -> np.ndarray:
    return np.array([
        1.0 - _LOZI_A * abs(x[0]) + x[1],
        _LOZI_B * x[0],
    ])


# Registry: name → (fn, center, radius, label, ndim, map_type)
_MAP_REGISTRY = {
    "henon": (
        _henon,
        np.array([0.0, 0.0]),
        np.array([1.5, 0.5]),
        "Hénon map  (a=1.4, b=0.3)  — fractal horseshoe",
        2, "discrete",
    ),
    "ikeda": (
        _ikeda,
        np.array([1.0, 0.0]),
        np.array([1.5, 1.5]),
        "Ikeda map  (μ=0.9)          — off-centre spiral",
        2, "discrete",
    ),
    "lozi": (
        _lozi,
        np.array([0.0, 0.0]),
        np.array([1.5, 0.75]),
        "Lozi map   (a=1.7, b=0.5)  — thin filament, extreme imbalance",
        2, "discrete",
    ),
    "fourwing": (
        _four_wing_v,
        np.array([0.0, 0.0, 0.0]),
        np.array([5.0, 5.0, 5.0]),
        "Four-Wing ODE               — near-uniform (control)",
        3, "ode",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Map construction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _unit_pts(ndim: int, res: int) -> np.ndarray:
    t = np.linspace(-1.0, 1.0, res)
    grids = np.meshgrid(*([t] * ndim), indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)


def _build_map(map_name: str, domain, unit_pts: np.ndarray, use_gpu: bool):
    fn, center, radius, label, ndim, map_type = _MAP_REGISTRY[map_name]

    if use_gpu:
        from gaio.cuda.accelerated_map import AcceleratedBoxMap
        from gaio import rk4_flow_map
        if map_type == "ode":
            f_cpu = rk4_flow_map(fn, step_size=0.01, steps=20)
            try:
                from numba import cuda as _cuda
                from gaio.cuda.rk4_cuda import make_cuda_rk4_flow_map
                # Build a device function lazily; the cpu map is used as fallback
                return AcceleratedBoxMap(f_cpu, domain, unit_pts,
                                         f_jit=None, backend="gpu")
            except Exception:
                pass
        # Discrete map — use CPU backend (GPU backend requires device kernel)
        from numba import njit
        @njit
        def _fn_jit(x):
            return fn(x)
        from gaio import SampledBoxMap
        return SampledBoxMap(fn, domain, unit_pts)

    from gaio import SampledBoxMap, rk4_flow_map
    if map_type == "ode":
        f = rk4_flow_map(fn, step_size=0.01, steps=20)
    else:
        f = fn
    return SampledBoxMap(f, domain, unit_pts)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameResult:
    frame:        int           # 0 = Phase 4, 1 = Phase 5
    label:        str           # "Phase 4" / "Phase 5"
    n_cells:      int           # attractor cells
    t_op_time:    float         # TransferOperator wall time (s)
    nnz:          int           # mat.nnz after dedup + normalise
    total_nnz_raw: int          # sum(per_rank_nnz)
    per_rank_nnz: list          # per-rank COO contributions
    imbalance:    float         # max/min ratio
    error:        Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Core benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_map_benchmark(
    map_name: str,
    steps: int,
    attractor_steps: Optional[int],
    grid_res: int,
    test_pts: int,
    n_frames: int,
    use_gpu: bool,
    comm,
    n_trials: int = 5,
    domain_scale: float = 1.0,
) -> list[FrameResult]:
    from gaio import Box, BoxPartition, BoxSet, TransferOperator, relative_attractor
    from gaio.mpi.load_balance import compute_imbalance

    fn, center, radius, label, ndim, map_type = _MAP_REGISTRY[map_name]
    my_rank = comm.Get_rank()

    # domain_scale > 1.0 inflates the domain beyond the attractor footprint.
    # This keeps fringe cells (low hit rate) alongside core attractor cells
    # (high hit rate), producing a genuine hot/cold split that stresses
    # Phase 5 load balancing even at modest K.
    domain   = Box(center, radius * domain_scale)
    P        = BoxPartition(domain, [grid_res] * ndim)
    upts     = _unit_pts(ndim, test_pts)

    try:
        F = _build_map(map_name, domain, upts, use_gpu)
    except Exception as exc:
        err = str(exc)
        return [FrameResult(0, "Phase 4", 0, 0., 0, 0, [], float("inf"), error=err),
                FrameResult(1, "Phase 5", 0, 0., 0, 0, [], float("inf"), error=err)]

    # ── Compute attractor once (shared by both frames) ────────────────────────
    # attractor_steps < steps → partially converged; retains fringe cells that
    # have fewer test-point hits than core cells, increasing COO imbalance.
    _asteps = attractor_steps if attractor_steps is not None else steps
    S = BoxSet.full(P)
    try:
        A = relative_attractor(F, S, steps=_asteps, comm=comm)
    except Exception as exc:
        err = f"attractor: {exc}"
        return [FrameResult(0, "Phase 4", 0, 0., 0, 0, [], float("inf"), error=err),
                FrameResult(1, "Phase 5", 0, 0., 0, 0, [], float("inf"), error=err)]

    # ── Warmup: two untimed Phase 4 runs to warm Python/NumPy internals ─────
    # Without warmup, Phase 4 pays cold-start costs (allocator, dispatch cache,
    # OS page faults) while Phase 5 benefits from a warm system — producing
    # wildly inflated "speedup" numbers unrelated to load balancing.
    for _ in range(2):
        try:
            TransferOperator(F, A, A, comm=comm, partition_weights=None)
        except Exception:
            pass

    results: list[FrameResult] = []
    partition_weights = None

    for frame in range(min(n_frames, 2)):  # frame 0 = Phase 4, frame 1 = Phase 5
        frame_label = "Phase 4 (static Morton)" if frame == 0 else "Phase 5 (weighted split)"
        T = None
        trial_times: list[float] = []

        for trial in range(n_trials):
            comm.Barrier()
            t0 = time.perf_counter()
            try:
                T = TransferOperator(F, A, A, comm=comm,
                                     partition_weights=partition_weights)
            except Exception as exc:
                results.append(FrameResult(
                    frame, frame_label, len(A), 0., 0, 0, [], float("inf"),
                    error=str(exc)
                ))
                T = None
                break
            comm.Barrier()
            trial_times.append(time.perf_counter() - t0)

        if T is None:
            continue

        elapsed = float(np.median(trial_times))
        nnz_arr = T.mpi_stats["per_rank_nnz"]
        imb     = compute_imbalance(nnz_arr)

        results.append(FrameResult(
            frame         = frame,
            label         = frame_label,
            n_cells       = len(A),
            t_op_time     = elapsed,
            nnz           = T.mat.nnz,
            total_nnz_raw = int(nnz_arr.sum()),
            per_rank_nnz  = nnz_arr.tolist(),
            imbalance     = float(imb) if not np.isinf(imb) else -1.0,
        ))

        # Phase 5: carry weights forward
        partition_weights = T.partition_weights

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Report formatting
# ─────────────────────────────────────────────────────────────────────────────

def _print_per_rank_table(label: str, per_rank_nnz: list) -> None:
    P = len(per_rank_nnz)
    total = sum(per_rank_nnz)
    print(f"  {label}")
    print(f"    {'Rank':>6}  {'COO entries':>14}  {'% of total':>10}")
    print(f"    {'-'*6}  {'-'*14}  {'-'*10}")
    for r, n in enumerate(per_rank_nnz):
        pct = 100.0 * n / total if total > 0 else 0.0
        print(f"    {r:>6}  {n:>14,}  {pct:>9.1f}%")
    print(f"    {'Total':>6}  {total:>14,}  {'100.0%':>10}")


def _print_comparison(
    map_name: str,
    map_label: str,
    r0: FrameResult,
    r1: FrameResult,
) -> None:
    n_ranks = len(r0.per_rank_nnz)

    print()
    print("=" * 70)
    print(f"  Map: {map_label}")
    print(f"  Ranks: {n_ranks}  |  Cells: {r0.n_cells:,}  |  mat.nnz: {r0.nnz:,}")
    print("=" * 70)
    print()

    # Per-rank tables
    _print_per_rank_table("Phase 4 — static Morton (uniform K/P split):",
                          r0.per_rank_nnz)
    print()
    _print_per_rank_table("Phase 5 — weighted split (balanced by hit density):",
                          r1.per_rank_nnz)
    print()

    # Summary comparison
    imb0 = r0.imbalance if r0.imbalance >= 0 else float("inf")
    imb1 = r1.imbalance if r1.imbalance >= 0 else float("inf")
    imb0_str = f"{imb0:.2f}×" if np.isfinite(imb0) else "∞ (idle rank)"
    imb1_str = f"{imb1:.2f}×" if np.isfinite(imb1) else "∞ (idle rank)"

    if np.isfinite(imb0) and np.isfinite(imb1) and imb0 > 0:
        imb_drop_pct = 100.0 * (imb0 - imb1) / imb0
        imb_summary  = f"{imb_drop_pct:+.1f}% (reduction)"
    else:
        imb_summary = "N/A"

    speedup = r0.t_op_time / r1.t_op_time if r1.t_op_time > 0 else float("nan")

    header = f"| {'':22} | {'Phase 4':>12} | {'Phase 5':>12} | {'Δ':>14} |"
    sep    = "|" + "|".join(["-" * 24, "-" * 14, "-" * 14, "-" * 16]) + "|"
    print(header)
    print(sep)
    print(f"| {'T_op wall time (s)':<22} | {r0.t_op_time:>12.3f} | {r1.t_op_time:>12.3f} | {f'{speedup:.2f}× speedup':>14} |")
    print(f"| {'Load imbalance (max/min)':<22} | {imb0_str:>12} | {imb1_str:>12} | {imb_summary:>14} |")
    print(f"| {'total_nnz_raw':<22} | {r0.total_nnz_raw:>12,} | {r1.total_nnz_raw:>12,} | {'':>14} |")
    print()

    # Verdict
    # Wall-time speedup from load balancing is only meaningful when per-rank
    # computation >> synchronization overhead.  Below ~5 ms/rank (K < ~10K
    # cells on CPU), timing noise dominates.  The per-rank COO table above is
    # the authoritative load-balance signal at small problem sizes.
    _too_small = r0.t_op_time < 0.005  # < 5 ms: noise-dominated timing
    if _too_small:
        print(f"  Phase 5 COO balance: {imb_summary}  (timing noise-dominated at K={r0.n_cells:,}; "
              f"wall-time speedup visible at K ≫ 10K / GPU scale)")
    elif r1.t_op_time > 0 and speedup > 1.05:
        print(f"  Phase 5 speedup:  {speedup:.2f}×  ({imb_summary})")
    elif r1.t_op_time > 0 and speedup >= 0.95:
        print(f"  Phase 5 overhead: ≈ {(1-speedup)*100:.1f}%  (expected for near-uniform attractors)")
    else:
        print(f"  Phase 5 time ratio: {speedup:.2f}×")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="GAIO.py Phase 5 load-balancing benchmark"
    )
    parser.add_argument("--map", choices=list(_MAP_REGISTRY.keys()),
                        nargs="+", default=["henon"],
                        help="One or more maps to benchmark "
                             "(default: henon; e.g. --map henon ikeda)")
    parser.add_argument("--all-maps", action="store_true",
                        help="Run all three maps in sequence (shorthand for "
                             "--map henon ikeda fourwing)")
    parser.add_argument("--steps", type=int, default=14,
                        help="Subdivision steps (default: 14; use ≥14 for visible imbalance)")
    parser.add_argument("--grid-res", type=int, default=2,
                        help="Grid cells per dimension for initial covering (default: 2)")
    parser.add_argument("--test-pts", type=int, default=3,
                        help="Test points per dimension per cell (default: 3 → 9/27 pts)")
    parser.add_argument("--n-frames", type=int, default=2,
                        help="Frames to run: 1 = Phase 4 only; 2 = Phase 4 + Phase 5 (default: 2)")
    parser.add_argument("--n-trials", type=int, default=5,
                        help="Timed trials per frame; median reported (default: 5)")
    parser.add_argument("--domain-scale", type=float, default=1.5,
                        help="Inflate domain radius by this factor (default: 1.5). "
                             "Values > 1 include fringe cells outside the attractor, "
                             "increasing hot/cold COO imbalance.")
    parser.add_argument("--attractor-steps", type=int, default=None,
                        help="Subdivision steps for relative_attractor (default: --steps). "
                             "Use fewer steps than --steps to retain fringe cells with "
                             "variable hit rates, producing more dramatic COO imbalance.")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU backend (requires AcceleratedBoxMap)")
    parser.add_argument("--json", type=str, default=None,
                        help="Save results to JSON file (rank 0 only)")
    args = parser.parse_args(argv)

    # ── MPI setup ─────────────────────────────────────────────────────────────
    from gaio.mpi.comm import get_comm
    comm     = get_comm()
    my_rank  = comm.Get_rank()
    n_ranks  = comm.Get_size()

    if my_rank == 0:
        print()
        print("GAIO.py Phase 5 Benchmark — Static vs. Adaptive Load Balancing")
        print(f"  Ranks     : {n_ranks}")
        print(f"  Steps     : {args.steps}  |  Grid: {args.grid_res}^d  |  Test pts: {args.test_pts}^d")
        print(f"  Backend   : {'GPU (AcceleratedBoxMap)' if args.gpu else 'CPU (SampledBoxMap)'}")
        print(f"  Frames    : {args.n_frames}  (0=Phase 4, 1=Phase 5)")
        print(f"  Timing    : median of {args.n_trials} trials + 2 warmup runs")
        print(f"  Domain    : {args.domain_scale:.1f}× attractor radius")
        _asteps = args.attractor_steps or args.steps
        _conv   = "converged" if _asteps >= args.steps else f"partial ({_asteps} steps)"
        print(f"  Attractor : {_asteps} steps  [{_conv}]")
        print()

    # GPU: bind each rank to its own device
    if args.gpu:
        try:
            from numba import cuda as _cuda
            _cuda.select_device(my_rank % _cuda.gpus.lst.__len__()
                                if hasattr(_cuda.gpus, 'lst') else my_rank)
        except Exception:
            pass

    maps_to_run = list(_MAP_REGISTRY.keys()) if args.all_maps else args.map

    all_results: dict = {}

    for map_name in maps_to_run:
        fn, center, radius, label, ndim, map_type = _MAP_REGISTRY[map_name]

        if my_rank == 0:
            print(f"── {label} ──")
            print(f"   domain: center={center.tolist()}, radius={radius.tolist()}")
            print(f"   Running Phase 4 ...", end="", flush=True)

        results = _run_map_benchmark(
            map_name        = map_name,
            steps           = args.steps,
            attractor_steps = args.attractor_steps,
            grid_res        = args.grid_res,
            test_pts        = args.test_pts,
            n_frames        = args.n_frames,
            use_gpu         = args.gpu,
            comm            = comm,
            n_trials        = args.n_trials,
            domain_scale    = args.domain_scale,
        )

        if my_rank == 0:
            for r in results:
                if r.error:
                    print(f"\n   [{r.label}] FAILED: {r.error}")
                elif r.frame == 0:
                    imb_str = f"{r.imbalance:.2f}×" if r.imbalance >= 0 else "∞"
                    print(f" {r.t_op_time:.3f}s  (imbalance={imb_str})")
                    if args.n_frames > 1:
                        print(f"   Running Phase 5 ...", end="", flush=True)
                elif r.frame == 1:
                    imb_str = f"{r.imbalance:.2f}×" if r.imbalance >= 0 else "∞"
                    print(f" {r.t_op_time:.3f}s  (imbalance={imb_str})")

            if len(results) >= 2 and results[0].error is None and results[1].error is None:
                _print_comparison(map_name, label, results[0], results[1])
            elif len(results) == 1 and results[0].error is None:
                # Phase 4 only
                r = results[0]
                imb_str = f"{r.imbalance:.2f}×" if r.imbalance >= 0 else "∞"
                print(f"\n  Phase 4 only: T_op={r.t_op_time:.3f}s  imbalance={imb_str}")
                _print_per_rank_table("Per-rank COO (Phase 4):", r.per_rank_nnz)
                print()

        all_results[map_name] = [asdict(r) for r in results]

    if my_rank == 0 and args.json:
        with open(args.json, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"Results saved to {args.json}")


if __name__ == "__main__":
    main()
