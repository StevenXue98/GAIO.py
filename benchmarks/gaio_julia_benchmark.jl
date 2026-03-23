"""
benchmarks/gaio_julia_benchmark.jl
====================================
Standalone GAIO.jl timing script for the Four-Wing attractor benchmark.
Outputs a single JSON line to stdout — parsed by benchmark_vs_julia.py.

Usage:
    julia -t auto --project=<GAIO.jl dir> benchmarks/gaio_julia_benchmark.jl
    julia -t auto --project=<GAIO.jl dir> benchmarks/gaio_julia_benchmark.jl \\
        --backend gpu --steps 10 --grid-res 2 --test-pts 4

Flags:
    --steps N       Subdivision steps for relative_attractor (default: 10)
    --grid-res N    Grid cells per dimension in initial BoxGrid (default: 2)
    --test-pts N    Test points per dimension per cell (default: 4 → 4³=64 pts)
    --n-trials N    Timed trials — median is reported (default: 3)
    --n-warmup N    JIT warmup runs before timing (default: 2)
    --backend STR   "cpu"  → tries :simd first, falls back to FLoops :grid
                    "gpu"  → :grid with CUDA (requires CUDA.jl installed)

Output (one JSON line, to stdout):
    {"backend":"simd","n_threads":30,"n_cells":4687,"nnz":27391,
     "attractor_time":0.85,"t_op_time":0.80,"total_time":1.65,
     "trials":[...],"error":null}

Notes on Julia GPU (--backend gpu):
    • GAIO.jl GPU path runs in float32 end-to-end; float32 literals are used
      for the map definition to avoid silent float64 promotion.
    • The construct_transfers GPU path is known to sometimes return nnz=0
      (codomain rebinding bug).  This script detects that and reports it in
      the JSON "note" field while still emitting valid timing data.
    • Install CUDA.jl first: julia --project=<dir> -e 'using Pkg; Pkg.add("CUDA")'
    • Install SIMD.jl for fastest CPU: julia --project=<dir> -e 'using Pkg; Pkg.add("SIMD")'
"""

using GAIO
using SparseArrays

# ── Optional extensions — loaded at module level to avoid Julia world-age issues ─
# @eval using X inside a function body makes X visible at module scope, but
# calling X.functional() in the *same* function runs at the pre-@eval world age,
# so Julia raises MethodError.  Loading here (once, at startup) fixes that.

const _SIMD_LOADED = try
    @eval using SIMD
    true
catch
    false
end

const _CUDA_LOADED = try
    @eval using CUDA
    true
catch
    false
end

# ── Argument parsing ──────────────────────────────────────────────────────────

function parse_args()
    d = Dict{String,Any}(
        "steps"    => 10,
        "grid_res" => 2,
        "test_pts" => 4,
        "n_trials" => 3,
        "n_warmup" => 2,
        "backend"  => "cpu",
    )
    i = 1
    while i <= length(ARGS)
        key = ARGS[i]
        val = i + 1 <= length(ARGS) ? ARGS[i+1] : nothing
        if     key == "--steps"    && !isnothing(val); d["steps"]    = parse(Int, val); i += 2
        elseif key == "--grid-res" && !isnothing(val); d["grid_res"] = parse(Int, val); i += 2
        elseif key == "--test-pts" && !isnothing(val); d["test_pts"] = parse(Int, val); i += 2
        elseif key == "--n-trials" && !isnothing(val); d["n_trials"] = parse(Int, val); i += 2
        elseif key == "--n-warmup" && !isnothing(val); d["n_warmup"] = parse(Int, val); i += 2
        elseif key == "--backend"  && !isnothing(val); d["backend"]  = val;             i += 2
        else;                                                                            i += 1
        end
    end
    return d
end

# ── Four-Wing vector field (float64 — CPU path) ───────────────────────────────

const _a64, _b64, _d64 = 0.2,    -0.01,    -0.4
_fw64((x, y, z)) = (_a64*x + y*z,  _d64*y + _b64*x - z*y,  -z - x*y)
_fw_f64(x) = rk4_flow_map(_fw64, x, 0.01,  20)

# ── Four-Wing vector field (float32 — GPU path) ───────────────────────────────
# Julia GPU kernels run in float32; explicit f32 literals prevent silent
# float64 promotion that would slow down the GPU path.

const _a32, _b32, _d32 = 0.2f0, -0.01f0, -0.4f0
_fw32((x, y, z)) = (_a32*x + y*z,  _d32*y + _b32*x - z*y,  -z - x*y)
_fw_f32(x) = rk4_flow_map(_fw32, x, 0.01f0, 20)

# ── BoxMap construction ───────────────────────────────────────────────────────

function make_boxmap_cpu(P, n_pts::Int)
    npts = ntuple(_ -> n_pts, 3)
    if _SIMD_LOADED
        # Try several calling conventions — SIMDExt dispatch varies across GAIO.jl versions.
        # Some versions accept n_points keyword; others require it to be omitted.
        for call in [
            () -> BoxMap(:simd, _fw_f64, P; n_points=npts),
            () -> BoxMap(:grid, :simd, _fw_f64, P; n_points=npts),
            () -> BoxMap(:simd, _fw_f64, P),
            () -> BoxMap(:grid, :simd, _fw_f64, P),
        ]
            try
                F = call()
                return F, "simd"
            catch
                # try next convention
            end
        end
        # SIMDExt installed but none of the dispatch forms worked — fall through to default
    end
    F = BoxMap(:grid, _fw_f64, P; n_points=npts)
    return F, "default"
end

function make_boxmap_gpu(P, n_pts::Int)
    npts = ntuple(_ -> n_pts, 3)
    _CUDA_LOADED || error(
        "CUDA.jl not installed. Run:\n" *
        "  julia --project=<GAIO.jl dir> -e 'using Pkg; Pkg.add(\"CUDA\")'\n" *
        "or: python benchmarks/benchmark_vs_julia.py --setup-julia"
    )
    CUDA.functional() || error("CUDA.functional() = false — no GPU visible to CUDA.jl")
    F = BoxMap(:grid, :gpu, _fw_f32, P; n_points=npts)
    return F, "cuda"
end

# ── JSON helpers (no JSON3 dependency) ───────────────────────────────────────

_jstr(x::AbstractString) = "\"$(escape_string(x))\""
_jstr(x::Number)         = string(x)
_jstr(x::Nothing)        = "null"
_jstr(x::Vector)         = "[" * join(_jstr.(x), ",") * "]"

function emit_json(; kw...)
    pairs = ["$(_jstr(string(k))):$(_jstr(v))" for (k, v) in kw]
    println("{" * join(pairs, ",") * "}")
    flush(stdout)
end

# ── Main ──────────────────────────────────────────────────────────────────────

function main()
    cfg = parse_args()
    steps    = cfg["steps"]
    grid_res = cfg["grid_res"]
    test_pts = cfg["test_pts"]
    n_trials = cfg["n_trials"]
    n_warmup = cfg["n_warmup"]
    backend  = cfg["backend"]

    center = (0.0, 0.0, 0.0)
    radius = (5.0, 5.0, 5.0)
    P = BoxGrid(Box(center, radius), ntuple(_ -> grid_res, 3))

    F, actual_backend = try
        if backend == "gpu"
            make_boxmap_gpu(P, test_pts)
        else
            make_boxmap_cpu(P, test_pts)
        end
    catch e
        emit_json(backend=backend, n_threads=Threads.nthreads(),
                  n_cells=0, nnz=0,
                  attractor_time=0.0, t_op_time=0.0, total_time=0.0,
                  trials=Float64[], note="", error=string(e))
        return
    end

    # ── JIT warmup ───────────────────────────────────────────────────────────
    # For GPU backend, also warm up the CPU fallback T_op path (used when nnz=0)
    _warmup_cpu_top = backend == "gpu" ? BoxMap(:grid, _fw_f64, P; n_points=ntuple(_ -> test_pts, 3)) : nothing
    for _ in 1:n_warmup
        S = cover(P, :)
        A = relative_attractor(F, S, steps=steps)
        T_w = TransferOperator(F, A, A)
        if backend == "gpu" && nnz(T_w.mat) == 0 && !isnothing(_warmup_cpu_top)
            TransferOperator(_warmup_cpu_top, A, A)
        end
    end

    # ── Timed trials ─────────────────────────────────────────────────────────
    attractor_times = Float64[]
    t_op_times      = Float64[]
    n_cells_last    = 0
    nnz_last        = 0
    note            = ""

    # CPU fallback boxmap — used when GPU T_op returns nnz=0 (known GAIO.jl CUDAExt bug)
    _cpu_fallback_map(P_ref, n_pts) = BoxMap(:grid, _fw_f64, P_ref; n_points=ntuple(_ -> n_pts, 3))

    for _ in 1:n_trials
        S = cover(P, :)
        t_att = @elapsed A = relative_attractor(F, S, steps=steps)
        t_top = @elapsed T = TransferOperator(F, A, A)
        push!(attractor_times, t_att)
        n_cells_last = length(A)
        this_nnz     = nnz(T.mat)

        # GPU construct_transfers bug: codomain overwritten with empty set → nnz=0.
        # Fall back to CPU BoxMap for T_op so T_op timing remains meaningful.
        if this_nnz == 0 && backend == "gpu"
            F_cpu = _cpu_fallback_map(P, test_pts)
            t_top = @elapsed T = TransferOperator(F_cpu, A, A)
            this_nnz = nnz(T.mat)
            if isempty(note)
                note = "GPU map + CPU T_op (GPU construct_transfers nnz=0 bug)"
            end
        end

        push!(t_op_times, t_top)
        nnz_last = this_nnz
    end

    sort!(attractor_times)
    sort!(t_op_times)
    med(v) = v[div(length(v), 2) + 1]

    att_med = med(attractor_times)
    top_med = med(t_op_times)
    all_totals = [attractor_times[i] + t_op_times[i] for i in 1:n_trials]

    emit_json(
        backend        = actual_backend,
        n_threads      = Threads.nthreads(),
        n_cells        = n_cells_last,
        nnz            = nnz_last,
        attractor_time = att_med,
        t_op_time      = top_med,
        total_time     = att_med + top_med,
        trials         = all_totals,
        note           = note,
        error          = nothing,
    )
end

main()
