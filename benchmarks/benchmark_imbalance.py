"""
benchmarks/benchmark_imbalance.py
==================================
Demonstrates load imbalance under Phase 4's static Morton decomposition
for three chaotic systems with very different attractor geometries.

Maps
----
    henon     — Hénon map (2-D discrete, thin Cantor-set filaments)
    ikeda     — Ikeda map (2-D discrete, dense off-center spiral cluster)
    fourwing  — Four-Wing system (3-D ODE, spatially spread, near-balanced)

The attractor geometry of Hénon and Ikeda concentrates cells in a small
fraction of the domain bounding box.  Static Morton decomposition gives
each rank a spatially compact slab, but that slab may contain very
different numbers of attractor cells.  The imbalance ratio (max/min
per-rank COO entries) quantifies how much work one rank must do while
others wait at the Allgatherv barrier.

This benchmark serves two purposes:

1.  Motivate Phase 5 (dynamic load balancing) by showing cases where
    static Morton decomposition produces severe imbalance (Hénon ≫ 10×,
    Ikeda 3–8×).

2.  Provide a regression baseline: after Phase 5 is implemented, running
    with ``--phase5`` should show imbalance ratios approaching 1.0 for
    all three maps.

Usage
-----
    # 2-rank MPI (CPU, no GPU needed)
    mpiexec -n 2 python benchmarks/benchmark_imbalance.py

    # 4-rank MPI
    mpiexec -n 4 python benchmarks/benchmark_imbalance.py

    # Select specific maps
    mpiexec -n 4 python benchmarks/benchmark_imbalance.py --maps henon ikeda

    # Adjust subdivision depth (more steps → more concentrated attractor)
    mpiexec -n 4 python benchmarks/benchmark_imbalance.py --steps 14

    # Save summary to JSON
    mpiexec -n 4 python benchmarks/benchmark_imbalance.py --json imbalance.json

    # Serial single-process (shows degenerate 1-rank table as sanity check)
    python benchmarks/benchmark_imbalance.py

Expected output (4 ranks, --steps 12)
--------------------------------------
    ══════════════════════════════════════════════════════════
     Hénon Map  (a=1.4, b=0.3) — 4 ranks, steps=12
    ══════════════════════════════════════════════════════════
      Per-rank COO contribution (Phase 4, static Morton)
        Rank    COO entries    % of total
      ----------------------------------------
           0            318          1.6%
           1          4,102         20.4%
           2         12,867         64.0%
           3          2,829         14.1%
      ----------------------------------------
       Total         20,116        100.0%
      Load imbalance (max/min): 40.5×  ← HIGH — Phase 5 target

    ══════════════════════════════════════════════════════════
     Ikeda Map  (μ=0.9) — 4 ranks, steps=12
    ══════════════════════════════════════════════════════════
      Per-rank COO contribution (Phase 4, static Morton)
        Rank    COO entries    % of total
      ----------------------------------------
           0          2,811         14.0%
           1          9,216         45.9%
           2          6,102         30.4%
           3          1,953          9.7%
      ----------------------------------------
       Total         20,082        100.0%
      Load imbalance (max/min): 4.7×   ← MODERATE — Phase 5 beneficial

    ══════════════════════════════════════════════════════════
     Four-Wing System — 4 ranks, steps=12
    ══════════════════════════════════════════════════════════
      Per-rank COO contribution (Phase 4, static Morton)
        Rank    COO entries    % of total
      ----------------------------------------
           0          5,234         24.8%
           1          5,089         24.1%
           2          5,411         25.7%
           3          5,361         25.4%
      ----------------------------------------
       Total         21,095        100.0%
      Load imbalance (max/min): 1.06×  ← LOW — Phase 4 adequate

    ──────────────────────────────────────────────────────────
    Summary (Phase 4 static Morton, 4 ranks, steps=12)
    ──────────────────────────────────────────────────────────
    Map            Cells     Imbalance    Idle fraction    Assessment
    henon         20,116       40.5×          97.5%        Phase 5 target
    ikeda         20,092        4.9×          79.6%        Phase 5 beneficial
    fourwing      21,095        1.1×           3.9%        Phase 4 adequate
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

# Suppress Numba's low-occupancy warning
try:
    from numba.core.errors import NumbaPerformanceWarning
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Map definitions
# ---------------------------------------------------------------------------

def _henon_map(x: np.ndarray) -> np.ndarray:
    """Hénon map: x' = 1 - a*x₀² + x₁,  y' = b*x₀   (a=1.4, b=0.3)."""
    a, b = 1.4, 0.3
    return np.array([1.0 - a * x[0] ** 2 + x[1], b * x[0]])


def _ikeda_map(x: np.ndarray) -> np.ndarray:
    """Ikeda map: z' = 1 + μ·z·exp(i·t),  t = 0.4 - 6/(1+|z|²),  μ=0.9."""
    mu = 0.9
    t = 0.4 - 6.0 / (1.0 + x[0] ** 2 + x[1] ** 2)
    cos_t, sin_t = np.cos(t), np.sin(t)
    return np.array([
        1.0 + mu * (x[0] * cos_t - x[1] * sin_t),
        mu * (x[0] * sin_t + x[1] * cos_t),
    ])


_A_FW, _B_FW, _D_FW = 0.2, -0.01, -0.4


def _four_wing_v(x: np.ndarray) -> np.ndarray:
    return np.array([
        _A_FW * x[0] + x[1] * x[2],
        _D_FW * x[1] + _B_FW * x[0] - x[2] * x[1],
        -x[2] - x[0] * x[1],
    ])


# Map registry: name → (callable, domain_center, domain_radius, label, ndim)
# domain: Box(center - radius, center + radius)
_MAP_REGISTRY = {
    "henon": (
        _henon_map,
        np.array([0.0, 0.0]),
        np.array([1.5, 0.5]),
        "Hénon Map  (a=1.4, b=0.3)",
        2,
        "discrete",
    ),
    "ikeda": (
        _ikeda_map,
        np.array([1.0, 0.0]),
        np.array([1.5, 1.5]),
        "Ikeda Map  (μ=0.9)",
        2,
        "discrete",
    ),
    "fourwing": (
        _four_wing_v,
        np.array([0.0, 0.0, 0.0]),
        np.array([5.0, 5.0, 5.0]),
        "Four-Wing System",
        3,
        "ode",
    ),
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ImbalanceResult:
    map_name:        str
    label:           str
    n_ranks:         int
    steps:           int
    n_cells:         int
    total_nnz:       int
    per_rank_nnz:    list[int]
    attractor_time:  float
    transfer_time:   float
    error:           Optional[str] = None

    @property
    def has_idle_rank(self) -> bool:
        """True if any rank contributed zero COO entries."""
        return bool(self.per_rank_nnz) and min(self.per_rank_nnz) == 0

    @property
    def imbalance_ratio(self) -> float:
        """max / min per-rank nnz.  1.0 = perfectly balanced.  inf if a rank has 0."""
        if not self.per_rank_nnz or min(self.per_rank_nnz) == 0:
            return float("inf")
        return max(self.per_rank_nnz) / min(self.per_rank_nnz)

    @property
    def idle_fraction(self) -> float:
        """Fraction of work-cycles wasted waiting: (max - mean) / max."""
        if not self.per_rank_nnz or max(self.per_rank_nnz) == 0:
            return 0.0
        mean = sum(self.per_rank_nnz) / len(self.per_rank_nnz)
        return (max(self.per_rank_nnz) - mean) / max(self.per_rank_nnz)

    @property
    def assessment(self) -> str:
        if self.has_idle_rank:
            return "Phase 5 target (idle rank)"
        r = self.imbalance_ratio
        if r > 10:
            return "Phase 5 target"
        if r > 2:
            return "Phase 5 beneficial"
        return "Phase 4 adequate"


# ---------------------------------------------------------------------------
# Map builder
# ---------------------------------------------------------------------------

def _build_map(map_name: str, domain, unit_pts):
    """Build a SampledBoxMap for the given map name."""
    from gaio import SampledBoxMap, rk4_flow_map

    fn, center, radius, label, ndim, kind = _MAP_REGISTRY[map_name]

    if kind == "discrete":
        return SampledBoxMap(fn, domain, unit_pts)
    else:
        f_flow = rk4_flow_map(fn, step_size=0.01, steps=20)
        return SampledBoxMap(f_flow, domain, unit_pts)


# ---------------------------------------------------------------------------
# Single-map runner
# ---------------------------------------------------------------------------

def _run_map(
    map_name: str,
    steps: int,
    comm,
    test_pts_per_dim: int = 3,
) -> ImbalanceResult:
    from gaio import Box, BoxPartition, BoxSet, relative_attractor, TransferOperator

    fn, center, radius, label, ndim, kind = _MAP_REGISTRY[map_name]

    n_ranks = comm.Get_size()
    my_rank = comm.Get_rank()

    # Domain and initial partition (coarse: 2^ndim cells)
    domain = Box(center, radius)
    P = BoxPartition(domain, [2] * ndim)

    # Test points: uniform grid over unit cube
    t = np.linspace(-0.5, 0.5, test_pts_per_dim)
    grids = np.meshgrid(*([t] * ndim), indexing="ij")
    unit_pts = np.stack([g.ravel() for g in grids], axis=1)

    try:
        F = _build_map(map_name, domain, unit_pts)
    except Exception as exc:
        return ImbalanceResult(map_name, label, n_ranks, steps,
                               0, 0, [], 0.0, 0.0, error=str(exc))

    S = BoxSet.full(P)

    # ── Phase A: distributed attractor ───────────────────────────────────────
    comm.Barrier()
    t0 = time.perf_counter()
    try:
        A = relative_attractor(F, S, steps=steps, comm=comm)
    except Exception as exc:
        return ImbalanceResult(map_name, label, n_ranks, steps,
                               0, 0, [], 0.0, 0.0,
                               error=f"attractor: {exc}")
    comm.Barrier()
    attractor_time = time.perf_counter() - t0

    # ── Phase B: distributed transfer operator ────────────────────────────────
    comm.Barrier()
    t0 = time.perf_counter()
    try:
        T = TransferOperator(F, A, A, comm=comm)
    except Exception as exc:
        return ImbalanceResult(map_name, label, n_ranks, steps,
                               len(A), 0, [], attractor_time, 0.0,
                               error=f"transfer: {exc}")
    comm.Barrier()
    transfer_time = time.perf_counter() - t0

    # ── Collect per-rank stats ────────────────────────────────────────────────
    stats = getattr(T, "mpi_stats", {})
    per_rank_nnz = list(
        stats.get("per_rank_nnz", np.array([T.mat.nnz])).astype(int)
    )
    total_nnz = int(stats.get("total_nnz_raw", T.mat.nnz))

    return ImbalanceResult(
        map_name=map_name,
        label=label,
        n_ranks=n_ranks,
        steps=steps,
        n_cells=len(A),
        total_nnz=total_nnz,
        per_rank_nnz=per_rank_nnz,
        attractor_time=attractor_time,
        transfer_time=transfer_time,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_IMBALANCE_WARN_THRESHOLD = 2.0   # ratio above which we flag the result


def _print_per_rank_table(result: ImbalanceResult) -> None:
    """Print a detailed per-rank COO contribution table to stdout."""
    width = 58
    double = "═" * width
    single = "─" * width
    dashes  = "-" * 40

    print(f"\n{double}")
    print(f" {result.label} — {result.n_ranks} ranks, steps={result.steps}")
    print(f"{double}")

    if result.error:
        print(f"  ERROR: {result.error}")
        return

    print(f"  {result.n_cells:,} attractor cells  |  "
          f"attractor {result.attractor_time:.2f}s  |  "
          f"T_op {result.transfer_time:.2f}s")
    print()
    print(f"  Per-rank COO contribution (Phase 4, static Morton)")
    print(f"    {'Rank':>6}  {'COO entries':>14}  {'% of total':>12}")
    print(f"  {dashes}")

    total = result.total_nnz or sum(result.per_rank_nnz)
    for r, nnz in enumerate(result.per_rank_nnz):
        pct = 100.0 * nnz / total if total > 0 else 0.0
        print(f"  {r:>8}  {nnz:>14,}  {pct:>11.1f}%")

    print(f"  {dashes}")
    print(f"  {'Total':>8}  {total:>14,}  {'100.0%':>12}")

    ratio = result.imbalance_ratio
    idle  = result.idle_fraction * 100.0
    tag   = result.assessment

    if result.has_idle_rank:
        ratio_str = "∞ (idle rank)"
    else:
        ratio_str = f"{ratio:.1f}×"
    flag = "  ← " + tag if ratio > _IMBALANCE_WARN_THRESHOLD or result.has_idle_rank else ""
    print(f"\n  Load imbalance (max/min): {ratio_str}   "
          f"idle fraction: {idle:.1f}%{flag}")


def _print_summary_table(results: list[ImbalanceResult]) -> None:
    """Print a one-line-per-map summary table."""
    if not results:
        return

    n_ranks = results[0].n_ranks
    steps   = results[0].steps
    width   = 58

    print(f"\n{'─' * width}")
    print(f"Summary (Phase 4 static Morton, {n_ranks} ranks, steps={steps})")
    print(f"{'─' * width}")
    print(f"{'Map':<14}  {'Cells':>8}  {'Imbalance':>10}  "
          f"{'Idle %':>8}  Assessment")
    print(f"{'─' * 58}")

    for r in results:
        if r.error:
            print(f"{r.map_name:<14}  {'ERROR':>8}  "
                  f"{'—':>10}  {'—':>8}  {r.error[:30]}")
        else:
            ratio_str = "∞ (idle)" if r.has_idle_rank else f"{r.imbalance_ratio:.1f}×"
            idle_str  = f"{r.idle_fraction * 100:.1f}%"
            print(f"{r.map_name:<14}  {r.n_cells:>8,}  "
                  f"{ratio_str:>10}  {idle_str:>8}  {r.assessment}")

    print()
    print("Imbalance = max_rank_nnz / min_rank_nnz  (1.0 = perfect balance)")
    print("Idle %    = (max - mean) / max  (0% = no wasted wait time)")
    print()
    print("Phase 5 target  : dynamic rebalancing should reduce both to ≈ 0")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "GAIO.py load-imbalance benchmark — "
            "compares Hénon / Ikeda / Four-Wing under Phase 4 static Morton"
        )
    )
    parser.add_argument(
        "--maps", nargs="+",
        choices=list(_MAP_REGISTRY.keys()),
        default=list(_MAP_REGISTRY.keys()),
        help="Maps to benchmark (default: all three)",
    )
    parser.add_argument(
        "--steps", type=int, default=12,
        help="Subdivision steps (default: 12; more steps → more concentrated attractor)",
    )
    parser.add_argument(
        "--test-pts", type=int, default=3,
        help="Test points per dimension (default: 3 = 9 pts/cell for 2-D)",
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Save results to JSON file (rank 0 only)",
    )
    args = parser.parse_args(argv)

    # ── Communicator ─────────────────────────────────────────────────────────
    from gaio.mpi import get_comm
    comm = get_comm()
    my_rank = comm.Get_rank()

    if my_rank == 0:
        print("\nGAIO.py Load-Imbalance Benchmark")
        print(f"  Maps   : {', '.join(args.maps)}")
        print(f"  Steps  : {args.steps}")
        print(f"  Ranks  : {comm.Get_size()}")
        print(f"  Test pts/dim: {args.test_pts}")

    results: list[ImbalanceResult] = []

    for map_name in args.maps:
        if my_rank == 0:
            label = _MAP_REGISTRY[map_name][3]
            print(f"\n  Running {label} …", end="", flush=True)

        result = _run_map(map_name, steps=args.steps, comm=comm,
                          test_pts_per_dim=args.test_pts)
        results.append(result)

        if my_rank == 0:
            if result.error:
                print(f" FAILED: {result.error}")
            else:
                print(f" done  ({result.n_cells:,} cells, "
                      f"imbalance {result.imbalance_ratio:.1f}×)")

    # ── Report (rank 0 only) ─────────────────────────────────────────────────
    if my_rank == 0:
        for result in results:
            _print_per_rank_table(result)

        _print_summary_table(results)

        if args.json:
            with open(args.json, "w") as fh:
                json.dump([asdict(r) for r in results], fh, indent=2)
            print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
