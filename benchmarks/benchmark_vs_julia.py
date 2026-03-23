"""
benchmarks/benchmark_vs_julia.py
==================================
Side-by-side benchmark: GAIO.py vs GAIO.jl — Four-Wing attractor.

Default run reproduces the README Phase 3 benchmark table (steps=10, 4³ test pts):

    python benchmarks/benchmark_vs_julia.py

Produces a 4-row table:

    | Impl          | Threads | Cells | Map (s) | T_op (s) | Total (s) | nnz | Speedup |
    |---------------|---------|-------|---------|----------|-----------|-----|---------|
    | numba         |      30 | 4,687 |    1.64 |     0.18 |      1.82 | ... |    1.0× |
    | numba-cuda    |       1 | 4,687 |    0.65 |     0.13 |      0.78 | ... |    2.3× |
    | julia-default |      30 | 4,687 |    0.99 |     0.80 |      1.78 | ... |    1.0× |
    | julia-cuda    |       1 | 4,687 |    0.06 |     0.55 |      0.61 | ... |    2.9× |

    (Values from README — actual timings depend on hardware.)

Backends:

  Python side:
    numba       — Numba @njit(parallel=True, fastmath=True), all CPU threads
    numba-cuda  — Numba @cuda.jit single GPU

  Julia side (via subprocess, gaio_julia_benchmark.jl):
    julia-simd    — BoxMap(:grid, :simd, ...) if SIMD.jl installed, else FLoops default
    julia-cuda    — BoxMap(:grid, :gpu, ...) requires CUDA.jl installed

  "julia-default" in the label means SIMD.jl was not found and FLoops was used.
  Install SIMD.jl to get "julia-simd" (fastest Julia CPU):
    julia --project=<GAIO.jl dir> -e 'using Pkg; Pkg.add("SIMD")'

Usage
-----
    # Default: numba + numba-cuda vs julia (cpu + gpu), steps=10 (README config)
    python benchmarks/benchmark_vs_julia.py

    # CPU-only comparison (skip GPU rows)
    python benchmarks/benchmark_vs_julia.py --python-backends numba --julia-backends cpu

    # Skip Julia entirely
    python benchmarks/benchmark_vs_julia.py --no-julia

    # Larger problem
    python benchmarks/benchmark_vs_julia.py --steps 14 --grid-res 4

    # Install Julia deps (SIMD.jl + CUDA.jl) before running
    python benchmarks/benchmark_vs_julia.py --setup-julia

    # Save results
    python benchmarks/benchmark_vs_julia.py --json results/vs_julia.json

Notes
-----
• Both sides warm up the JIT before timing (--n-warmup runs, default 2).
  Reported time = median of --n-trials timed runs (default 3).
• Julia GPU (julia-cuda): uses float32 end-to-end for GPU efficiency.
  GAIO.jl's construct_transfers GPU path sometimes returns nnz=0 (known bug);
  if detected, the note column flags it and T_op time is unreliable.
• Speedup baseline = numba (first Python CPU result), or first successful result.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

try:
    from numba.core.errors import NumbaPerformanceWarning
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
except ImportError:
    pass


# ── Paths ─────────────────────────────────────────────────────────────────────

_REPO_ROOT        = pathlib.Path(__file__).parent.parent
_RESULTS_DIR      = _REPO_ROOT / "results"
_JULIA_SCRIPT     = pathlib.Path(__file__).parent / "gaio_julia_benchmark.jl"
_DEFAULT_JL_PROJ  = _REPO_ROOT / "references" / "GAIO.jl-master"


def _find_julia() -> Optional[str]:
    candidates = [
        shutil.which("julia"),
        str(pathlib.Path.home() / "julia-1.10.8" / "bin" / "julia"),
        str(pathlib.Path.home() / "julia-1.11.0" / "bin" / "julia"),
        "/usr/local/bin/julia",
        "/usr/bin/julia",
    ]
    for p in candidates:
        if p and pathlib.Path(p).exists():
            return p
    return None


# ── Display name mapping ──────────────────────────────────────────────────────
# Internal backend keys → human-readable labels for the comparison table.

_DISPLAY = {
    "python":      "python",
    "numba":       "numba",
    "numba-cuda":  "numba-cuda",
}


# ── Four-Wing map ─────────────────────────────────────────────────────────────

A_FW, B_FW, D_FW = 0.2, -0.01, -0.4


def _four_wing_v(x: np.ndarray) -> np.ndarray:
    return np.array([
        A_FW * x[0] + x[1] * x[2],
        D_FW * x[1] + B_FW * x[0] - x[2] * x[1],
        -x[2] - x[0] * x[1],
    ])


def _unit_pts(res: int = 4) -> np.ndarray:
    """Grid matching GAIO.jl GridBoxMap: k*(2/n)-1 for k=0..n-1."""
    t = np.arange(res, dtype=float) * (2.0 / res) - 1.0
    gx, gy, gz = np.meshgrid(t, t, t, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    impl:       str        # "numba", "numba-cuda", "julia-default", "julia-simd", "julia-cuda"
    n_threads:  int        # CPU threads (1 for GPU)
    n_cells:    int
    map_time:   float      # relative_attractor wall time (s)
    t_op_time:  float      # TransferOperator wall time (s)
    total_time: float
    nnz:        int
    note:       str  = ""
    error:      Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ── Python backend factories ──────────────────────────────────────────────────

def _make_numba_map(domain, unit_pts):
    from numba import njit
    from gaio import rk4_flow_map, AcceleratedBoxMap
    from gaio.maps.rk4 import make_njit_rk4_flow_map_scalar3

    # Scalar vfield: takes 3 floats, returns 3-tuple — zero heap allocation
    # inside the RK4 loop.  All stage vectors live in CPU registers.
    @njit(fastmath=True)
    def _fw_vfield_s(x0, x1, x2):
        return (
            A_FW * x0 + x1 * x2,
            D_FW * x1 + B_FW * x0 - x2 * x1,
            -x2 - x0 * x1,
        )

    f_jit = make_njit_rk4_flow_map_scalar3(_fw_vfield_s, step_size=0.01, steps=20)
    f_cpu = rk4_flow_map(_four_wing_v, step_size=0.01, steps=20)
    return AcceleratedBoxMap(f_cpu, domain, unit_pts, f_jit=f_jit, backend="cpu")


def _make_numba_cuda_map(domain, unit_pts):
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
    return AcceleratedBoxMap(f_cpu, domain, unit_pts, f_device=f_device, backend="gpu")


def _make_python_map(domain, unit_pts):
    from gaio import SampledBoxMap, rk4_flow_map
    return SampledBoxMap(
        rk4_flow_map(_four_wing_v, step_size=0.01, steps=20), domain, unit_pts
    )


_MAP_FACTORIES = {
    "python":     _make_python_map,
    "numba":      _make_numba_map,
    "numba-cuda": _make_numba_cuda_map,
}


def _run_python_backend(name: str, domain, unit_pts, P, steps: int,
                        n_trials: int, n_warmup: int) -> BenchResult:
    from gaio import BoxSet, relative_attractor, TransferOperator

    try:
        F = _MAP_FACTORIES[name](domain, unit_pts)
    except Exception as exc:
        return BenchResult(name, 0, 0, 0., 0., 0., 0, error=str(exc))

    n_threads = 1
    if name == "numba":
        try:
            import numba
            n_threads = int(os.environ.get(
                "NUMBA_NUM_THREADS",
                getattr(numba.config, "NUMBA_NUM_THREADS", os.cpu_count() or 1)
            ))
        except Exception:
            n_threads = os.cpu_count() or 1

    # Warmup: forces Numba JIT compilation and warms OS page cache
    for _ in range(n_warmup):
        try:
            A_w = relative_attractor(F, BoxSet.full(P), steps=steps)
            TransferOperator(F, A_w, A_w)
        except Exception:
            pass

    att_times, top_times = [], []
    n_cells_last = nnz_last = 0

    for _ in range(n_trials):
        t0 = time.perf_counter()
        try:
            A = relative_attractor(F, BoxSet.full(P), steps=steps)
        except Exception as exc:
            return BenchResult(name, n_threads, 0, 0., 0., 0., 0,
                               error=f"attractor: {exc}")
        att_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        try:
            T = TransferOperator(F, A, A)
        except Exception as exc:
            return BenchResult(name, n_threads, len(A),
                               float(np.median(att_times)), 0., 0., 0,
                               error=f"transfer: {exc}")
        top_times.append(time.perf_counter() - t0)
        n_cells_last = len(A)
        nnz_last     = T.mat.nnz

    att_med = float(np.median(att_times))
    top_med = float(np.median(top_times))
    return BenchResult(
        impl=name, n_threads=n_threads, n_cells=n_cells_last,
        map_time=att_med, t_op_time=top_med,
        total_time=att_med + top_med, nnz=nnz_last,
    )


# ── Julia runner ──────────────────────────────────────────────────────────────

def _run_julia(julia_exec: str, julia_project: str, backend: str,
               steps: int, grid_res: int, test_pts: int,
               n_trials: int, n_warmup: int) -> BenchResult:
    """Launch gaio_julia_benchmark.jl with --backend cpu|gpu, parse JSON."""
    cmd = [
        julia_exec, "-t", "auto",
        f"--project={julia_project}",
        str(_JULIA_SCRIPT),
        "--backend",  backend,
        "--steps",    str(steps),
        "--grid-res", str(grid_res),
        "--test-pts", str(test_pts),
        "--n-trials", str(n_trials),
        "--n-warmup", str(n_warmup),
    ]
    impl_label = "julia-cuda" if backend == "gpu" else "julia-?"
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except FileNotFoundError:
        return BenchResult(impl_label, 0, 0, 0., 0., 0., 0,
                           error=f"Julia not found: {julia_exec}")
    except subprocess.TimeoutExpired:
        return BenchResult(impl_label, 0, 0, 0., 0., 0., 0,
                           error="Julia timed out (>900 s)")

    # Find the JSON line (last line starting with '{')
    json_line = None
    for line in reversed(proc.stdout.splitlines()):
        if line.strip().startswith("{"):
            json_line = line.strip()
            break

    if json_line is None:
        stderr_tail = (proc.stderr or "")[-600:]
        return BenchResult(impl_label, 0, 0, 0., 0., 0., 0,
                           error=f"No JSON from Julia. stderr: {stderr_tail}")

    try:
        d = json.loads(json_line)
    except json.JSONDecodeError as exc:
        return BenchResult(impl_label, 0, 0, 0., 0., 0., 0,
                           error=f"JSON parse error: {exc}\nRaw: {json_line}")

    if d.get("error"):
        return BenchResult(impl_label, d.get("n_threads", 0), 0, 0., 0., 0., 0,
                           error=d["error"])

    # Build display label from the backend the Julia script actually used
    jl_backend = d.get("backend", "?")
    impl = f"julia-{jl_backend}"   # e.g. "julia-simd", "julia-default", "julia-cuda"

    return BenchResult(
        impl       = impl,
        n_threads  = d.get("n_threads", 0),
        n_cells    = d.get("n_cells", 0),
        map_time   = d.get("attractor_time", 0.),
        t_op_time  = d.get("t_op_time", 0.),
        total_time = d.get("total_time", 0.),
        nnz        = d.get("nnz", 0),
        note       = d.get("note", ""),
    )


# ── Julia dependency setup ────────────────────────────────────────────────────

def _setup_julia_deps(julia_exec: str, julia_project: str) -> None:
    """Install SIMD.jl and CUDA.jl into the Julia project environment."""
    print("  Installing Julia dependencies (SIMD.jl, CUDA.jl) ...")
    print("  Note: CUDA.jl is large (~500 MB); this may take several minutes.")
    cmd = [
        julia_exec,
        f"--project={julia_project}",
        "-e", 'using Pkg; Pkg.add(["SIMD", "CUDA"]); Pkg.precompile()',
    ]
    proc = subprocess.run(cmd, timeout=900)
    if proc.returncode == 0:
        print("  Julia deps installed successfully.")
    else:
        print("  WARNING: Julia dep install returned non-zero exit code.")


# ── Table formatting ──────────────────────────────────────────────────────────

def _format_table(results: list[BenchResult], baseline_impl: str) -> str:
    baseline_time: Optional[float] = None
    for r in results:
        if r.ok and r.impl == baseline_impl:
            baseline_time = r.total_time
            break
    if baseline_time is None:
        for r in results:
            if r.ok:
                baseline_time = r.total_time
                break

    # Column widths
    cw = [15, 8, 8, 10, 10, 10, 8, 8, 36]
    header = (
        f"| {'Impl':<{cw[0]}} | {'Threads':>{cw[1]}} | {'Cells':>{cw[2]}} "
        f"| {'Map (s)':>{cw[3]}} | {'T_op (s)':>{cw[4]}} | {'Total (s)':>{cw[5]}} "
        f"| {'nnz':>{cw[6]}} | {'Speedup':>{cw[7]}} | {'Note':<{cw[8]}} |"
    )
    sep = "|" + "|".join("-" * (w + 2) for w in cw) + "|"
    rows = [header, sep]

    for r in results:
        note_str = r.note[:cw[8]] if r.note else ""
        if r.error:
            err_str = r.error[:50]
            rows.append(
                f"| {r.impl:<{cw[0]}} | {'—':>{cw[1]}} | {'—':>{cw[2]}} "
                f"| {'—':>{cw[3]}} | {'—':>{cw[4]}} | {'—':>{cw[5]}} "
                f"| {'—':>{cw[6]}} | {'ERR':>{cw[7]}} | {err_str:<{cw[8]}} |"
            )
        else:
            speedup = (f"{baseline_time / r.total_time:.2f}×"
                       if baseline_time and r.total_time > 0 else "—")
            thr_str = str(r.n_threads) if r.n_threads > 0 else "—"
            rows.append(
                f"| {r.impl:<{cw[0]}} | {thr_str:>{cw[1]}} | {r.n_cells:>{cw[2]},} "
                f"| {r.map_time:>{cw[3]}.3f} | {r.t_op_time:>{cw[4]}.3f} "
                f"| {r.total_time:>{cw[5]}.3f} | {r.nnz:>{cw[6]},} "
                f"| {speedup:>{cw[7]}} | {note_str:<{cw[8]}} |"
            )
    return "\n".join(rows)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="GAIO.py vs GAIO.jl benchmark — Four-Wing attractor (README config)"
    )
    # ── Problem size (defaults reproduce README Phase 3 table) ────────────────
    parser.add_argument("--steps", type=int, default=10,
                        help="Subdivision steps (default: 10 — matches README benchmark)")
    parser.add_argument("--grid-res", type=int, default=2,
                        help="Initial grid cells per dimension (default: 2)")
    parser.add_argument("--test-pts", type=int, default=4,
                        help="Test points per dimension (default: 4 → 4³=64 pts)")
    # ── Python backends ───────────────────────────────────────────────────────
    parser.add_argument("--python-backends", nargs="+",
                        choices=["python", "numba", "numba-cuda"],
                        default=["numba", "numba-cuda"],
                        help="Python backends to run (default: numba numba-cuda)")
    # ── Julia backends ────────────────────────────────────────────────────────
    parser.add_argument("--no-julia", action="store_true",
                        help="Skip all Julia benchmarks")
    parser.add_argument("--julia-backends", nargs="+",
                        choices=["cpu", "gpu"],
                        default=["cpu", "gpu"],
                        help="Julia backends: cpu (FLoops/SIMD), gpu (CUDA) (default: both)")
    parser.add_argument("--julia-exec", type=str, default=None,
                        help="Path to julia binary (auto-detected if omitted)")
    parser.add_argument("--julia-project", type=str,
                        default=str(_DEFAULT_JL_PROJ),
                        help=f"GAIO.jl project dir (default: {_DEFAULT_JL_PROJ})")
    # ── Timing ────────────────────────────────────────────────────────────────
    parser.add_argument("--n-trials", type=int, default=3,
                        help="Timed trials per backend — median reported (default: 3)")
    parser.add_argument("--n-warmup", type=int, default=2,
                        help="Warmup runs before timing to amortise JIT (default: 2)")
    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument("--setup-julia", action="store_true",
                        help="Install SIMD.jl + CUDA.jl into the Julia project first")
    parser.add_argument("--json", type=str, default=None,
                        help="Save all results to JSON file")
    args = parser.parse_args(argv)

    # ── Julia binary ──────────────────────────────────────────────────────────
    julia_exec = None
    if not args.no_julia:
        julia_exec = args.julia_exec or _find_julia()
        if julia_exec is None:
            print("Julia not found — Julia rows will be skipped.")
            print("  Use --julia-exec /path/to/julia to specify the binary.")

    # ── Optional Julia dep install ────────────────────────────────────────────
    if args.setup_julia and julia_exec:
        _setup_julia_deps(julia_exec, args.julia_project)

    print()
    print("GAIO.py vs GAIO.jl — Four-Wing Attractor (README Phase 3 Config)")
    print(f"  steps={args.steps}  grid_res={args.grid_res}  "
          f"test_pts={args.test_pts}³={args.test_pts**3}  "
          f"trials={args.n_trials}  warmup={args.n_warmup}")
    print()

    from gaio import Box, BoxPartition
    domain = Box(np.zeros(3), np.full(3, 5.0))
    P      = BoxPartition(domain, [args.grid_res] * 3)
    upts   = _unit_pts(args.test_pts)

    results: list[BenchResult] = []

    # ── Python backends ───────────────────────────────────────────────────────
    for backend in args.python_backends:
        print(f"  [{backend}] running ...", end="", flush=True)
        r = _run_python_backend(backend, domain, upts, P,
                                steps=args.steps, n_trials=args.n_trials,
                                n_warmup=args.n_warmup)
        if r.ok:
            print(f" {r.total_time:.3f}s  ({r.n_cells:,} cells)")
        else:
            print(f" FAILED: {r.error}")
        results.append(r)

    # ── Julia backends ────────────────────────────────────────────────────────
    if not args.no_julia and julia_exec:
        for jl_backend in args.julia_backends:
            label = "julia-cuda" if jl_backend == "gpu" else "julia-cpu"
            print(f"  [{label}] running Julia ({julia_exec}) ...", flush=True)
            if jl_backend == "gpu":
                print("           (note: Julia GPU T_op may return nnz=0 — known bug)")
            print("           warmup + timed runs ...", end="", flush=True)
            r = _run_julia(
                julia_exec, args.julia_project, jl_backend,
                steps=args.steps, grid_res=args.grid_res, test_pts=args.test_pts,
                n_trials=args.n_trials, n_warmup=args.n_warmup,
            )
            if r.ok:
                print(f" {r.total_time:.3f}s  ({r.n_cells:,} cells, "
                      f"{r.n_threads} threads, impl={r.impl})")
                if r.note:
                    print(f"           NOTE: {r.note}")
            else:
                print(f" FAILED: {r.error}")
            results.append(r)

    # ── Comparison table ──────────────────────────────────────────────────────
    # Speedup baseline: numba > python > first successful result
    baseline = next(
        (r.impl for r in results if r.ok and r.impl == "numba"), None
    ) or next(
        (r.impl for r in results if r.ok and r.impl == "python"), None
    ) or next(
        (r.impl for r in results if r.ok), None
    ) or "numba"

    print()
    print("=" * 100)
    print("  Four-Wing Attractor — GAIO.py vs GAIO.jl")
    print(f"  steps={args.steps}  grid_res={args.grid_res}  "
          f"test_pts={args.test_pts}³  |  speedup baseline: {baseline}")
    print("=" * 100)
    print(_format_table(results, baseline_impl=baseline))
    print()

    # ── Head-to-head CPU and GPU breakdown ────────────────────────────────────
    py_cpu  = next((r for r in results if r.impl == "numba"      and r.ok), None)
    py_gpu  = next((r for r in results if r.impl == "numba-cuda" and r.ok), None)
    jl_cpu  = next((r for r in results
                    if r.impl.startswith("julia-") and "cuda" not in r.impl and r.ok), None)
    jl_gpu  = next((r for r in results if r.impl == "julia-cuda" and r.ok), None)

    def _vs(a_time: float, b_time: float, a_label: str, b_label: str) -> tuple[str, str]:
        """Return (multiplier_str, verdict) — always expressed as 'a is Nx faster/slower than b'."""
        if b_time <= 0:
            return "—", "?"
        ratio = b_time / a_time   # > 1 means a is faster
        if ratio > 1.05:
            return f"{ratio:.2f}×", "faster than"
        elif ratio < 0.95:
            return f"{1.0 / ratio:.2f}×", "slower than"
        else:
            return f"{ratio:.2f}×", "on par with"

    if py_cpu and jl_cpu:
        mul, verdict = _vs(py_cpu.total_time, jl_cpu.total_time, "numba", jl_cpu.impl)
        map_mul, map_v = _vs(py_cpu.map_time, jl_cpu.map_time, "numba", jl_cpu.impl)
        top_mul, top_v = _vs(py_cpu.t_op_time, jl_cpu.t_op_time, "numba", jl_cpu.impl)
        print(f"  CPU: numba is {mul} {verdict} {jl_cpu.impl}")
        print(f"       map  : numba={py_cpu.map_time:.3f}s  {jl_cpu.impl}={jl_cpu.map_time:.3f}s  "
              f"({map_mul} {map_v})")
        print(f"       T_op : numba={py_cpu.t_op_time:.3f}s  {jl_cpu.impl}={jl_cpu.t_op_time:.3f}s  "
              f"({top_mul} {top_v})")
        print()

    if py_gpu and jl_gpu and jl_gpu.nnz > 0:
        mul, verdict = _vs(py_gpu.total_time, jl_gpu.total_time, "numba-cuda", "julia-cuda")
        map_mul, map_v = _vs(py_gpu.map_time, jl_gpu.map_time, "numba-cuda", "julia-cuda")
        print(f"  GPU: numba-cuda is {mul} {verdict} julia-cuda")
        print(f"       map  : numba-cuda={py_gpu.map_time:.3f}s  julia-cuda={jl_gpu.map_time:.3f}s  "
              f"({map_mul} {map_v})")
        print(f"       (note: julia-cuda T_op uses float32; numba-cuda uses float64)")
        print()
    elif jl_gpu and jl_gpu.nnz == 0:
        print(f"  GPU: julia-cuda T_op returned nnz=0 (known GAIO.jl bug) — "
              f"GPU comparison limited to attractor map time only")
        if py_gpu:
            map_mul, map_v = _vs(py_gpu.map_time, jl_gpu.map_time, "numba-cuda", "julia-cuda")
            print(f"       map-only: numba-cuda is {map_mul} {map_v} julia-cuda")
        print()

    # ── JSON export ───────────────────────────────────────────────────────────
    if args.json:
        out = pathlib.Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump({"config": vars(args), "results": [asdict(r) for r in results]},
                      fh, indent=2)
        print(f"  Results saved to {out}")


if __name__ == "__main__":
    main()
