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

Why the Lozi map?
-----------------
Hénon, Ikeda, and Four-Wing all show ~1.1× COO imbalance under Morton
decomposition when the domain tightly bounds the attractor — the Z-curve
distributes attractor cells uniformly.  The Lozi map (a=1.7, b=0.5) has
a thin angular filament concentrated in one corner of its domain.  When
the bounding box is inflated (``--domain-scale`` > 1), the attractor
occupies only a small fraction of the covering — cold fringe cells
accumulate on certain Morton slabs, producing genuine hot/cold splits.

Two predefined scenarios
------------------------
Running ``benchmark_phase5.py`` with no arguments executes both:

  moderate  — Lozi, steps=16, domain_scale=12.0 (fully converged)
              ~572 cells, Phase 4 imbalance ≈ 4.4×
              Phase 5 should reduce imbalance to ≈ 1.0× and deliver
              measurable T_op speedup.

  extreme   — Lozi, steps=14, domain_scale=8.0, attractor_steps=10
              (partially converged attractor retains fringe cells)
              ~78 cells, Phase 4 imbalance ≈ 21.5×
              Phase 5 should dramatically reduce imbalance toward 1×.

A single default run fills in all rows of the README Phase 5 table.

Usage
-----
    # Default: run both predefined scenarios (fills README table)
    mpiexec -n 4 python benchmarks/benchmark_phase5.py

    # Run a single named scenario
    mpiexec -n 4 python benchmarks/benchmark_phase5.py --scenario moderate
    mpiexec -n 4 python benchmarks/benchmark_phase5.py --scenario extreme

    # Custom run (any map / parameters)
    mpiexec -n 4 python benchmarks/benchmark_phase5.py \\
        --map lozi --steps 16 --domain-scale 12.0

    # GPU backend (bind one rank per device)
    mpiexec -n 4 --bind-to socket python benchmarks/benchmark_phase5.py --gpu

    # Save results to JSON
    mpiexec -n 4 python benchmarks/benchmark_phase5.py --json phase5.json

Expected output (4 ranks, default parameters, CPU backend)
----------------------------------------------------------
    GAIO.py Phase 5 Benchmark — Static vs. Adaptive Load Balancing
      Ranks     : 4
      Backend   : CPU (SampledBoxMap)
      Timing    : median of 5 trials + 2 warmup runs

    ══════════════════════════════════════════════════════════════════════
    Scenario 1 — MODERATE imbalance
    Lozi map (a=1.7, b=0.5): steps=16, domain_scale=12.0
      domain: center=[0. 0.], radius=[18. 9.]
      572 cells  |  Phase 4 imbalance: 4.4×
    ══════════════════════════════════════════════════════════════════════

    Phase 4 — static Morton (uniform K/P split):
        Rank  COO entries  % of total
        ----  -----------  ----------
           0          291        7.2%
           1        1,181       29.2%
           2        1,283       31.8%
           3        1,285       31.8%
       Total        4,040      100.0%

    Phase 5 — weighted split (balanced by hit density):
        Rank  COO entries  % of total
        ----  -----------  ----------
           0        1,009       25.0%
           1        1,012       25.1%
           2        1,010       25.0%
           3        1,009       25.0%
       Total        4,040      100.0%

    |                        |      Phase 4 |      Phase 5 |              Δ |
    |------------------------|--------------|--------------|----------------|
    | T_op wall time (s)     |        0.073 |        0.022 |   3.32× speedup|
    | Load imbalance (max/min)|        4.42× |        1.00× |  -77.4% (redu.)|
    | total_nnz_raw          |        4,040 |        4,040 |                |

    Phase 5 speedup:  3.32×  (-77.4% imbalance reduction)

    ══════════════════════════════════════════════════════════════════════
    Scenario 2 — EXTREME imbalance
    Lozi map (a=1.7, b=0.5): steps=14, domain_scale=8.0, attractor_steps=10
      domain: center=[0. 0.], radius=[12. 6.]
      78 cells  |  Phase 4 imbalance: 21.5×
    ══════════════════════════════════════════════════════════════════════
    ...

(Exact timings and imbalance vary with hardware and MPI configuration.)
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

# Suppress Numba low-occupancy warning for small benchmark problems
try:
    from numba.core.errors import NumbaPerformanceWarning
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Map definitions
# ─────────────────────────────────────────────────────────────────────────────

# Hénon map: a=1.4, b=0.3.  Horseshoe attractor — fractal horseshoe.
_HENON_A, _HENON_B = 1.4, 0.3

def _henon(x: np.ndarray) -> np.ndarray:
    return np.array([1.0 - _HENON_A * x[0]**2 + x[1], _HENON_B * x[0]])


# Ikeda map: μ=0.9.  Off-centre spiral.
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


# Lozi map: a=1.7, b=0.5.  Thin angular filament — primary imbalance target.
# Inflating the domain with domain_scale > 1 concentrates the attractor in
# one corner of the bounding box, creating a strong hot/cold COO split.
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
# Predefined scenarios (run by default)
# ─────────────────────────────────────────────────────────────────────────────

# Each scenario specifies a complete set of parameters that, when run with
# mpiexec -n 4, produces the imbalance ratio stated in the README table.
SCENARIO_REGISTRY = {
    "moderate": {
        "map":             "lozi",
        "steps":           16,
        "domain_scale":    9.0,
        "attractor_steps": None,   # fully converged
        "grid_res":        2,
        "test_pts":        3,
        "title":           "MODERATE imbalance",
        "description":     "Lozi map (a=1.7, b=0.5): steps=16, domain_scale=9.0",
        "expected_phase4_imbalance": "~4-5×",
    },
    "extreme": {
        "map":             "lozi",
        "steps":           16,
        "domain_scale":    12.0,
        "attractor_steps": None,   # fully converged, but large inflated domain
        "grid_res":        2,
        "test_pts":        3,
        "title":           "EXTREME imbalance",
        "description":     "Lozi map (a=1.7, b=0.5): steps=16, domain_scale=12.0",
        "expected_phase4_imbalance": "~30×",
    },
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
                from gaio.cuda.rk4_cuda import make_cuda_rk4_flow_map
                return AcceleratedBoxMap(f_cpu, domain, unit_pts,
                                         f_jit=None, backend="gpu")
            except Exception:
                pass
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
    scenario_title: str = "",
    scenario_desc: str = "",
) -> None:
    n_ranks = len(r0.per_rank_nnz)

    print()
    print("=" * 70)
    if scenario_title:
        print(f"  Scenario — {scenario_title}")
    print(f"  {scenario_desc or map_label}")
    print(f"  Ranks: {n_ranks}  |  Cells: {r0.n_cells:,}  |  mat.nnz: {r0.nnz:,}")
    imb0_str = f"{r0.imbalance:.1f}×" if r0.imbalance >= 0 else "∞"
    print(f"  Phase 4 imbalance: {imb0_str}")
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
    print(f"| {'T_op wall time (s)':<22} | {r0.t_op_time:>12.3f} | {r1.t_op_time:>12.3f} "
          f"| {f'{speedup:.2f}× speedup':>14} |")
    print(f"| {'Load imbalance (max/min)':<22} | {imb0_str:>12} | {imb1_str:>12} "
          f"| {imb_summary:>14} |")
    print(f"| {'total_nnz_raw':<22} | {r0.total_nnz_raw:>12,} | {r1.total_nnz_raw:>12,} "
          f"| {'':>14} |")
    print()

    # Verdict
    _too_small = r0.t_op_time < 0.005
    if _too_small:
        print(f"  Phase 5 COO balance: {imb_summary}  "
              f"(timing noise-dominated at K={r0.n_cells:,}; "
              f"wall-time speedup visible at K ≫ 10K / GPU scale)")
    elif r1.t_op_time > 0 and speedup > 1.05:
        print(f"  Phase 5 speedup:  {speedup:.2f}×  ({imb_summary})")
    elif r1.t_op_time > 0 and speedup >= 0.95:
        print(f"  Phase 5 overhead: ≈ {(1-speedup)*100:.1f}%  "
              f"(expected for near-uniform attractors)")
    else:
        print(f"  Phase 5 time ratio: {speedup:.2f}×")
    print()


def _print_readme_rows(scenario_name: str, r0: FrameResult, r1: FrameResult) -> None:
    """Print Markdown table rows for pasting into the README Phase 5 section."""
    imb0 = f"{r0.imbalance:.1f}×" if r0.imbalance >= 0 else "∞"
    imb1 = f"{r1.imbalance:.1f}×" if r1.imbalance >= 0 else "∞"
    speedup = r0.t_op_time / r1.t_op_time if r1.t_op_time > 0 else float("nan")
    spd_str = f"{speedup:.1f}×" if np.isfinite(speedup) else "—"
    print(f"\n  ── README rows ({scenario_name}) ──")
    print(f"  | Phase 4 static Morton | {r0.n_cells:>5,} | {r0.t_op_time:>8.3f} "
          f"| {imb0:>9} | baseline |")
    print(f"  | Phase 5 weighted      | {r1.n_cells:>5,} | {r1.t_op_time:>8.3f} "
          f"| {imb1:>9} | {spd_str} T_op speedup |")


def _print_phase4_only(scenario_name: str, r: FrameResult) -> None:
    """Print output when only Phase 4 was run (--n-frames 1)."""
    imb_str = f"{r.imbalance:.2f}×" if r.imbalance >= 0 else "∞"
    print(f"\n  Phase 4 only: T_op={r.t_op_time:.3f}s  imbalance={imb_str}")
    _print_per_rank_table("Per-rank COO (Phase 4):", r.per_rank_nnz)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="GAIO.py Phase 5 load-balancing benchmark"
    )

    # ── Scenario selection ────────────────────────────────────────────────────
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--scenario", choices=list(SCENARIO_REGISTRY.keys()), nargs="+",
        default=list(SCENARIO_REGISTRY.keys()),
        help="Predefined scenario(s) to run (default: all — fills README table). "
             "Choices: " + ", ".join(
                 f"{k} ({v['expected_phase4_imbalance']} Phase 4 imbalance)"
                 for k, v in SCENARIO_REGISTRY.items()
             ),
    )
    mode_group.add_argument(
        "--map", choices=list(_MAP_REGISTRY.keys()), nargs="+",
        help="Custom run: one or more maps (bypasses predefined scenarios; "
             "use with --steps, --domain-scale, etc.)",
    )

    # ── Custom-run parameters (used only when --map is given) ─────────────────
    parser.add_argument("--steps", type=int, default=14,
                        help="Subdivision steps for custom --map run (default: 14)")
    parser.add_argument("--grid-res", type=int, default=2,
                        help="Grid cells per dimension for initial covering (default: 2)")
    parser.add_argument("--test-pts", type=int, default=3,
                        help="Test points per dimension per cell (default: 3)")
    parser.add_argument("--domain-scale", type=float, default=1.5,
                        help="Inflate domain radius by this factor (default: 1.5)")
    parser.add_argument("--attractor-steps", type=int, default=None,
                        help="Subdivision steps for relative_attractor (default: --steps)")

    # ── Common parameters ─────────────────────────────────────────────────────
    parser.add_argument("--n-frames", type=int, default=2,
                        help="Frames to run: 1 = Phase 4 only; 2 = Phase 4 + Phase 5 (default: 2)")
    parser.add_argument("--n-trials", type=int, default=5,
                        help="Timed trials per frame; median reported (default: 5)")
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
        print(f"  Backend   : {'GPU (AcceleratedBoxMap)' if args.gpu else 'CPU (SampledBoxMap)'}")
        print(f"  Frames    : {args.n_frames}  (0=Phase 4, 1=Phase 5)")
        print(f"  Timing    : median of {args.n_trials} trials + 2 warmup runs")

    # GPU: bind each rank to its own device
    if args.gpu:
        try:
            from numba import cuda as _cuda
            _cuda.select_device(my_rank % _cuda.gpus.lst.__len__()
                                if hasattr(_cuda.gpus, 'lst') else my_rank)
        except Exception:
            pass

    all_results: dict = {}

    # ── Build list of runs ────────────────────────────────────────────────────
    if args.map is not None:
        # Custom run: use CLI parameters directly
        runs = []
        for map_name in args.map:
            fn, center, radius, label, ndim, _ = _MAP_REGISTRY[map_name]
            _asteps = args.attractor_steps or args.steps
            _conv   = "converged" if _asteps >= args.steps else f"partial ({_asteps} steps)"
            runs.append({
                "map":             map_name,
                "steps":           args.steps,
                "domain_scale":    args.domain_scale,
                "attractor_steps": args.attractor_steps,
                "grid_res":        args.grid_res,
                "test_pts":        args.test_pts,
                "title":           label,
                "description":     (f"{label}: steps={args.steps}, "
                                    f"domain_scale={args.domain_scale:.1f}, "
                                    f"attractor [{_conv}]"),
                "scenario_name":   map_name,
            })
        if my_rank == 0:
            print(f"  Mode      : custom (--map)")
            print()
    else:
        # Predefined scenario(s)
        runs = []
        for sc_name in args.scenario:
            sc = SCENARIO_REGISTRY[sc_name]
            _asteps = sc["attractor_steps"] or sc["steps"]
            _conv   = ("converged" if _asteps >= sc["steps"]
                       else f"partial ({_asteps} steps)")
            desc = (f"{sc['description']}"
                    + (f", attractor_steps={_asteps} [{_conv}]"
                       if sc["attractor_steps"] else ""))
            runs.append({**sc, "scenario_name": sc_name, "description": desc})
        if my_rank == 0:
            scenarios_str = ", ".join(args.scenario)
            print(f"  Scenarios : {scenarios_str}")
            print()

    # ── Execute runs ──────────────────────────────────────────────────────────
    for i, run_cfg in enumerate(runs, 1):
        map_name    = run_cfg["map"]
        sc_name     = run_cfg["scenario_name"]
        sc_title    = run_cfg.get("title", map_name)
        sc_desc     = run_cfg["description"]
        fn, center, radius, label, ndim, _ = _MAP_REGISTRY[map_name]

        if my_rank == 0:
            print(f"  ({'Scenario ' + str(i) + ': ' if not args.map else ''}"
                  f"{sc_name.upper() if not args.map else sc_name})")
            print(f"    {sc_desc}")
            domain_r = radius * run_cfg["domain_scale"]
            print(f"    domain: center={center.tolist()}, "
                  f"radius={domain_r.tolist()}")
            print(f"    Running Phase 4 ...", end="", flush=True)

        results = _run_map_benchmark(
            map_name        = map_name,
            steps           = run_cfg["steps"],
            attractor_steps = run_cfg["attractor_steps"],
            grid_res        = run_cfg["grid_res"],
            test_pts        = run_cfg["test_pts"],
            n_frames        = args.n_frames,
            use_gpu         = args.gpu,
            comm            = comm,
            n_trials        = args.n_trials,
            domain_scale    = run_cfg["domain_scale"],
        )

        if my_rank == 0:
            for r in results:
                if r.error:
                    print(f"\n   [{r.label}] FAILED: {r.error}")
                elif r.frame == 0:
                    imb_str = f"{r.imbalance:.2f}×" if r.imbalance >= 0 else "∞"
                    print(f" {r.t_op_time:.3f}s  (imbalance={imb_str})")
                    if args.n_frames > 1:
                        print(f"    Running Phase 5 ...", end="", flush=True)
                elif r.frame == 1:
                    imb_str = f"{r.imbalance:.2f}×" if r.imbalance >= 0 else "∞"
                    print(f" {r.t_op_time:.3f}s  (imbalance={imb_str})")

            if len(results) >= 2 and results[0].error is None and results[1].error is None:
                _print_comparison(map_name, label, results[0], results[1],
                                  scenario_title=sc_title,
                                  scenario_desc=sc_desc)
                _print_readme_rows(sc_name, results[0], results[1])
            elif len(results) == 1 and results[0].error is None:
                _print_phase4_only(sc_name, results[0])

        all_results[sc_name] = [asdict(r) for r in results]

    if my_rank == 0 and args.json:
        out_path = _json_path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
