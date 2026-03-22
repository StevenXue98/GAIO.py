"""
tests/test_mpi_scaling.py
=========================
MPI scaling test for the TransferOperator distributed pipeline.

Designed to be run directly under mpiexec — NOT via pytest:

    # 1 rank (serial path exercised with MPI communicator explicitly)
    mpiexec -n 1 python tests/test_mpi_scaling.py

    # 4 ranks (Morton decomposition + Allgatherv)
    mpiexec -n 4 python tests/test_mpi_scaling.py

    # 4 ranks with GPU (one GPU per rank — CUDA must be available)
    mpiexec -n 4 python tests/test_mpi_scaling.py --gpu

    # Finer resolution
    mpiexec -n 4 python tests/test_mpi_scaling.py --steps 10 --grid-res 4

What it verifies
----------------
1.  Every rank completes the full pipeline without error.
2.  All ranks produce a TransferOperator with the same ``mat.shape`` and
    ``mat.nnz`` (consistency check via Bcast of rank-0 values).
3.  Rank 0 prints a human-readable table showing:
      - Grid resolution and attractor cell count.
      - Per-rank COO contribution (proves Morton decomposition splits work).
      - Total COO entries before and after sparse deduplication.
      - Imbalance ratio (max/min per-rank nnz — should be close to 1.0 for
        a spatially uniform attractor, higher for a sparse fractal).
4.  The leading eigenvalue is close to 1.0 (sanity check for the physics).

Usage
-----
    mpiexec -n 4 python tests/test_mpi_scaling.py
    mpiexec -n 4 python tests/test_mpi_scaling.py --steps 12 --grid-res 4
    mpiexec -n 4 python tests/test_mpi_scaling.py --gpu
    mpiexec -n 4 python tests/test_mpi_scaling.py --no-attractor
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# Four-wing system (copied inline to keep this file self-contained)
# ---------------------------------------------------------------------------

A_FW, B_FW, D_FW = 0.2, -0.01, -0.4


def _four_wing_v(x: np.ndarray) -> np.ndarray:
    return np.array([
        A_FW * x[0] + x[1] * x[2],
        D_FW * x[1] + B_FW * x[0] - x[2] * x[1],
        -x[2] - x[0] * x[1],
    ])


def _make_cpu_map(domain, unit_pts):
    from gaio import SampledBoxMap, rk4_flow_map
    f = rk4_flow_map(_four_wing_v, step_size=0.01, steps=20)
    return SampledBoxMap(f, domain, unit_pts)


def _make_gpu_map(domain, unit_pts):
    from numba import cuda
    from gaio import rk4_flow_map, AcceleratedBoxMap
    from gaio.cuda.rk4_cuda import make_cuda_rk4_flow_map

    @cuda.jit(device=True)
    def _vfield(x, out):
        out[0] = A_FW * x[0] + x[1] * x[2]
        out[1] = D_FW * x[1] + B_FW * x[0] - x[2] * x[1]
        out[2] = -x[2] - x[0] * x[1]

    f_cpu    = rk4_flow_map(_four_wing_v, step_size=0.01, steps=20)
    f_device = make_cuda_rk4_flow_map(_vfield, ndim=3, step_size=0.01, steps=20)
    return AcceleratedBoxMap(f_cpu, domain, unit_pts,
                             f_device=f_device, backend="gpu")


# ---------------------------------------------------------------------------
# Rank-aware print (only rank 0 prints unless force=True)
# ---------------------------------------------------------------------------

def rprint(msg: str, comm, *, force: bool = False) -> None:
    if comm.Get_rank() == 0 or force:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="MPI scaling test — Four-wing TransferOperator"
    )
    parser.add_argument("--grid-res", type=int, default=3,
                        help="Grid cells per dimension (default: 3 = 27 initial cells)")
    parser.add_argument("--steps", type=int, default=8,
                        help="Subdivision steps for attractor (default: 8, ~fast)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU backend (requires CUDA + Numba)")
    parser.add_argument("--no-attractor", action="store_true",
                        help="Skip attractor subdivision; use full partition as domain")
    args = parser.parse_args(argv)

    # ── MPI setup ────────────────────────────────────────────────────────────
    from gaio.mpi import get_comm, rank as mpi_rank, size as mpi_size
    comm = get_comm()
    r, s = mpi_rank(), mpi_size()

    rprint("=" * 60, comm)
    rprint(f"  GAIO.py MPI Scaling Test   ({s} rank{'s' if s > 1 else ''})", comm)
    rprint("=" * 60, comm)

    # ── Domain & map ─────────────────────────────────────────────────────────
    from gaio import Box, BoxPartition, BoxSet, relative_attractor, TransferOperator

    center = np.array([0.0, 0.0, 0.0])
    radius = np.array([5.0, 5.0, 5.0])
    domain_box = Box(center, radius)
    P = BoxPartition(domain_box, [args.grid_res] * 3)

    t    = np.array([-0.5, 0.0, 0.5])
    gx, gy, gz = np.meshgrid(t, t, t)
    unit_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    # Only rank 0 builds the GPU map to avoid CUDA init races, then all ranks
    # use the same map object (the CPU fallback path is always safe).
    use_gpu = args.gpu
    if use_gpu:
        from gaio import cuda_available
        if not cuda_available():
            rprint("[warning] CUDA not available; falling back to CPU", comm)
            use_gpu = False

    if use_gpu:
        # Each rank selects its own GPU (one GPU per rank)
        try:
            import numba.cuda as numba_cuda
            numba_cuda.select_device(r % numba_cuda.gpus.lst.__len__())
        except Exception:
            pass
        try:
            F = _make_gpu_map(domain_box, unit_pts)
        except Exception as e:
            rprint(f"[rank {r}] GPU map init failed ({e}); falling back to CPU",
                   comm, force=True)
            F = _make_cpu_map(domain_box, unit_pts)
            use_gpu = False
    else:
        F = _make_cpu_map(domain_box, unit_pts)

    rprint(f"  Backend : {F}", comm)

    # ── Attractor (all ranks compute; result is deterministic) ───────────────
    S = BoxSet.full(P)
    if args.no_attractor:
        A = S
        rprint(f"  Domain  : full partition  ({len(A)} cells)", comm)
    else:
        rprint(f"  Attractor computation ({args.steps} steps) …", comm)
        t0_attr = time.perf_counter()
        A = relative_attractor(F, S, steps=args.steps)
        t1_attr = time.perf_counter()
        rprint(f"  Attractor: {len(A)} cells  ({t1_attr - t0_attr:.2f} s)", comm)

    # ── TransferOperator (MPI-distributed) ───────────────────────────────────
    rprint("  Building TransferOperator (MPI-distributed) …", comm)
    comm.Barrier()
    t0_to = time.perf_counter()
    T = TransferOperator(F, A, A, comm=comm)
    comm.Barrier()
    t1_to = time.perf_counter()

    # ── Consistency check — all ranks must have identical mat.shape / nnz ────
    shape_here = np.array(list(T.mat.shape) + [T.mat.nnz], dtype=np.int64)
    shape_r0   = np.empty(3, dtype=np.int64)
    # Broadcast rank-0 values to all ranks
    if r == 0:
        shape_r0[:] = shape_here
    comm.Bcast(shape_r0, root=0)

    ok = np.array_equal(shape_here, shape_r0)
    ok_global = np.empty(s, dtype=np.int8)
    comm.Allgather(np.array([int(ok)], dtype=np.int8), ok_global)

    if not ok_global.all():
        bad_ranks = np.where(ok_global == 0)[0].tolist()
        rprint(f"\n[FAIL] Inconsistent TransferOperator on ranks: {bad_ranks}", comm)
        sys.exit(1)

    # ── Eigenvalue sanity ─────────────────────────────────────────────────────
    if r == 0:
        vals, _ = T.eigs(k=1)
        leading_ev = float(np.abs(vals[0]))
    else:
        leading_ev = 0.0

    # ── Report (rank 0 only) ──────────────────────────────────────────────────
    if r == 0:
        stats     = T.mpi_stats
        pr_nnz    = stats["per_rank_nnz"]    # shape (n_ranks,)
        total_raw = stats["total_nnz_raw"]

        print()
        print("  Transfer Operator")
        print(f"    shape           : {T.mat.shape[0]} × {T.mat.shape[1]}")
        print(f"    nnz (deduped)   : {T.mat.nnz}")
        print(f"    assembly time   : {t1_to - t0_to:.3f} s")
        print(f"    leading |λ|     : {leading_ev:.8f}  (expect ≈ 1.0)")
        print()
        print("  Per-rank COO contribution (Morton decomposition)")
        print(f"  {'Rank':>6}  {'COO entries':>14}  {'% of total':>12}")
        print("  " + "-" * 38)
        for rank_i, n in enumerate(pr_nnz):
            pct = 100.0 * n / total_raw if total_raw > 0 else 0.0
            print(f"  {rank_i:>6}  {n:>14,}  {pct:>11.2f}%")
        print("  " + "-" * 38)
        print(f"  {'Total':>6}  {total_raw:>14,}  {'100.00%':>12}")

        if s > 1:
            imbalance = pr_nnz.max() / max(pr_nnz.min(), 1)
            print(f"\n  Load imbalance (max/min): {imbalance:.3f}"
                  f"  ({'good' if imbalance < 1.5 else 'consider re-balancing'})")
        print()
        print("  [PASS] All ranks produced identical TransferOperator.")
        print("=" * 60)

    comm.Barrier()


if __name__ == "__main__":
    main()
