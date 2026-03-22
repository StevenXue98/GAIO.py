"""
gaio/cuda/rk4_cuda.py
======================
CUDA device function factory for classical 4th-order Runge-Kutta integration.

Mirrors the Python :func:`gaio.maps.rk4.rk4_flow_map` interface, but produces
a ``@cuda.jit(device=True)`` flow-map device function that can be passed
directly to :class:`~gaio.cuda.AcceleratedBoxMap`.

Usage pattern
-------------
Python (CPU) side::

    from gaio.maps.rk4 import rk4_flow_map

    def vfield(x):          # returns np.ndarray
        return np.array([...])

    flow = rk4_flow_map(vfield, step_size=0.05, steps=5)
    F = SampledBoxMap(flow, domain, unit_pts)

CUDA (GPU) side::

    from numba import cuda
    from gaio.cuda.rk4_cuda import make_cuda_rk4_flow_map
    from gaio.cuda import AcceleratedBoxMap

    @cuda.jit(device=True)
    def vfield_device(x, out):      # writes f(x) into out, returns nothing
        out[0] = ...
        out[1] = ...

    flow_device = make_cuda_rk4_flow_map(vfield_device, ndim=3,
                                          step_size=0.05, steps=5)
    F = AcceleratedBoxMap(flow, domain, unit_pts,
                          f_device=flow_device, backend='auto')

Requirements on ``vfield_device``
----------------------------------
* Must be decorated with ``@numba.cuda.jit(device=True)``.
* Must use the **output-parameter pattern**: accept two 1-D device-array
  views ``(x, out)`` and write the result ``f(x)`` into ``out`` in-place.
  Must not return a value (Numba CUDA device functions cannot return
  heap-allocated arrays).

How it works
------------
``make_cuda_rk4_flow_map`` is a Python factory that captures ``vfield_device``,
``ndim``, ``step_size``, and ``steps`` by closure and applies the
``@cuda.jit(device=True)`` decorator to a new inner function.  Because
``ndim`` and ``steps`` are Python integers in the closure, Numba treats
them as compile-time constants — the inner ``for i in range(ndim)`` loops
are fully unrolled and the ``for _ in range(steps)`` loop is compiled
to a tight GPU loop.  ``vfield_device`` is inlined into the PTX via
Numba's closure mechanism (same strategy as :func:`make_map_kernel`).

Intermediate RK4 arrays (k1, k2, k3, k4, tmp) are allocated with
``cuda.local.array`` — they live in GPU registers or L1 scratchpad,
never in global memory.  For ``ndim ≤ 6`` this costs only ~40 bytes
of register file per thread.
"""
from __future__ import annotations


def make_cuda_rk4_flow_map(vfield_device, ndim: int,
                            step_size: float, steps: int):
    """
    Build a ``@cuda.jit(device=True)`` RK4 flow-map device function.

    Parameters
    ----------
    vfield_device : ``@cuda.jit(device=True)`` callable
        Vector field with the output-parameter signature::

            vfield_device(x: 1-D device array, out: 1-D device array) -> None

        Writes ``f(x)`` into ``out``.
    ndim : int
        State-space dimension (must equal ``x.shape[0]``).
    step_size : float
        RK4 step size (dt).  Total integration time = ``step_size * steps``.
    steps : int
        Number of RK4 steps to apply.

    Returns
    -------
    ``@cuda.jit(device=True)`` callable
        Flow-map device function with the same output-parameter signature::

            flow_device(x: 1-D device array, out: 1-D device array) -> None

        Writes ``Φ^T(x)`` (the state after ``step_size * steps`` time) into
        ``out``.  Pass this as ``f_device`` to
        :class:`~gaio.cuda.AcceleratedBoxMap`.

    Examples
    --------
    >>> from numba import cuda
    >>> from gaio.cuda.rk4_cuda import make_cuda_rk4_flow_map
    >>> @cuda.jit(device=True)
    ... def linear_vfield(x, out):
    ...     out[0] = -x[0]      # x' = -x  →  x(t) = e^{-t} x₀
    >>> flow_d = make_cuda_rk4_flow_map(linear_vfield, ndim=1,
    ...                                  step_size=0.1, steps=10)
    """
    try:
        import numba
        from numba import cuda
    except ImportError as exc:
        raise ImportError(
            "make_cuda_rk4_flow_map requires 'numba' with CUDA support.  "
            "Install with: conda install numba cudatoolkit"
        ) from exc

    @cuda.jit(device=True)
    def rk4_flow_device(x, out):
        # ── Intermediate local arrays (GPU registers / L1 scratchpad) ─────
        # cuda.local.array shape must be a compile-time constant; capturing
        # `ndim` from the closure makes it one for Numba's JIT compiler.
        k1  = cuda.local.array(ndim, numba.float64)
        k2  = cuda.local.array(ndim, numba.float64)
        k3  = cuda.local.array(ndim, numba.float64)
        k4  = cuda.local.array(ndim, numba.float64)
        tmp = cuda.local.array(ndim, numba.float64)

        # Initialise running state in `out` (reuse output buffer as state)
        for i in range(ndim):
            out[i] = x[i]

        for _ in range(steps):
            # k1 = f(state)
            vfield_device(out, k1)
            # k2 = f(state + dt/2 · k1)
            for i in range(ndim):
                tmp[i] = out[i] + (step_size * 0.5) * k1[i]
            vfield_device(tmp, k2)
            # k3 = f(state + dt/2 · k2)
            for i in range(ndim):
                tmp[i] = out[i] + (step_size * 0.5) * k2[i]
            vfield_device(tmp, k3)
            # k4 = f(state + dt · k3)
            for i in range(ndim):
                tmp[i] = out[i] + step_size * k3[i]
            vfield_device(tmp, k4)
            # RK4 update: state += (dt/6)(k1 + 2k2 + 2k3 + k4)
            for i in range(ndim):
                out[i] += (step_size / 6.0) * (
                    k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]
                )

    return rk4_flow_device
