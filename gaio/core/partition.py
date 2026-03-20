"""
gaio/core/partition.py
======================
BoxPartition — a uniform Cartesian grid discretisation of a Box domain.

Key design: flat ``int64`` keys
--------------------------------
Every cell in the grid is identified by a single ``int64`` value obtained
via  ``np.ravel_multi_index(multi_index, dims)``  (C / row-major order).
The inverse is  ``np.unravel_index(flat_key, dims)``.

Why flat keys?
  * A plain Python ``int`` or a contiguous ``int64`` array is the *only*
    cell-identifier representation that passes safely into Numba JIT
    functions and MPI send/recv buffers.  Tuple keys require Python object
    boxing that Numba cannot handle in ``nopython`` mode.
  * All vectorised operations (containment tests, transfer-matrix row
    extraction, MPI scatter/gather) work naturally on 1-D integer arrays.

Cell geometry
-------------
For a domain  B = [lo, hi)  divided into ``dims[i]`` cells along axis i:

    cell_width[i]  = (hi[i] - lo[i]) / dims[i]   = 2 * domain.radius[i] / dims[i]
    cell_radius[i] = cell_width[i] / 2            = domain.radius[i] / dims[i]

For flat key ``k`` with multi-index  ``m = unravel_index(k, dims)``:

    center[i] = lo[i] + cell_width[i] * (m[i] + 0.5)
              = lo[i] + 2 * cell_radius[i] * (m[i] + 0.5)
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .box import Box, F64, I64


class BoxPartition:
    """
    Uniform Cartesian grid of  prod(dims)  cells covering *domain*.

    Parameters
    ----------
    domain : Box
        The spatial domain to be partitioned.
    dims : array_like of int, shape (n,)
        Number of grid cells per dimension.  All values must be ≥ 1.

    Attributes
    ----------
    domain : Box
    dims : NDArray[I64], shape (n,)
    cell_radius : NDArray[F64], shape (n,)
        Half-width of each grid cell.

    Examples
    --------
    >>> import numpy as np
    >>> from gaio.core.box import Box
    >>> domain = Box([0.0, 0.0], [1.0, 1.0])
    >>> p = BoxPartition(domain, [4, 4])
    >>> p.size
    16
    >>> p.key_to_box(0)
    Box(center=[-0.75, -0.75], radius=[0.25, 0.25])
    >>> p.key_to_box(5)
    Box(center=[-0.25, 0.25], radius=[0.25, 0.25])
    >>> p.point_to_key(np.array([0.1, -0.3]))
    9
    """

    __slots__ = ("domain", "dims", "cell_radius")

    def __init__(
        self,
        domain: Box,
        dims: NDArray[I64] | list[int] | tuple[int, ...],
    ) -> None:
        d = np.ascontiguousarray(dims, dtype=I64)
        if d.ndim != 1:
            raise ValueError("dims must be a 1-D array.")
        if d.shape[0] != domain.ndim:
            raise ValueError(
                f"dims has length {d.shape[0]} but domain has {domain.ndim} dimensions."
            )
        if np.any(d < 1):
            raise ValueError("All dims values must be ≥ 1.")

        self.domain: Box = domain
        self.dims: NDArray[I64] = d
        # cell half-widths: domain.radius / dims  (strictly positive)
        self.cell_radius: NDArray[F64] = domain.radius / d.astype(F64)

    # ------------------------------------------------------------------
    # Shape properties
    # ------------------------------------------------------------------

    @property
    def ndim(self) -> int:
        """Spatial dimension of the domain."""
        return self.domain.ndim

    @property
    def size(self) -> int:
        """Total number of cells  = prod(dims)."""
        return int(np.prod(self.dims))

    @property
    def cell_volume(self) -> float:
        """Volume of one cell  = prod(2 * cell_radius)."""
        return float(np.prod(2.0 * self.cell_radius))

    # ------------------------------------------------------------------
    # Key ↔ multi-index conversions
    # ------------------------------------------------------------------

    def key_to_multi(self, key: int | NDArray[I64]) -> NDArray[I64]:
        """
        Convert flat key(s) to multi-index array(s).

        Parameters
        ----------
        key : int or ndarray of int64
            Flat key(s) in  [0, size).

        Returns
        -------
        ndarray, shape (n,)  or (N, n)
            Multi-index array(s).
        """
        scalar = np.isscalar(key)
        k = np.asarray(key, dtype=I64)
        multi = np.stack(np.unravel_index(k.ravel(), self.dims), axis=-1).astype(I64)
        return multi[0] if scalar else multi

    def multi_to_key(self, multi: NDArray[I64]) -> int | NDArray[I64]:
        """
        Convert multi-index array(s) to flat key(s).

        Parameters
        ----------
        multi : ndarray, shape (n,) or (N, n)
            Multi-index (or batch of multi-indices).

        Returns
        -------
        int or ndarray of int64
        """
        m = np.asarray(multi, dtype=I64)
        scalar = m.ndim == 1
        m2d = m[None, :] if scalar else m  # (N, n)
        keys = np.ravel_multi_index(m2d.T, self.dims).astype(I64)
        return int(keys[0]) if scalar else keys

    # ------------------------------------------------------------------
    # Key ↔ spatial conversions
    # ------------------------------------------------------------------

    def key_to_box(self, key: int) -> Box:
        """
        Return the Box corresponding to flat *key*.

        The returned box has:
          center = lo + cell_width * (multi + 0.5)
          radius = cell_radius

        where  cell_width = 2 * cell_radius.

        Raises
        ------
        IndexError
            If *key* is outside  [0, size).
        """
        k = int(key)
        if k < 0 or k >= self.size:
            raise IndexError(
                f"Key {k} is out of bounds for partition of size {self.size}."
            )
        multi = self.key_to_multi(k).astype(F64)
        center = self.domain.lo + 2.0 * self.cell_radius * (multi + 0.5)
        return Box(center, self.cell_radius.copy())

    def point_to_key(self, point: NDArray[F64]) -> int | None:
        """
        Return the flat key of the cell containing *point*, or ``None``
        if *point* is outside the domain.

        Uses half-open containment  [lo, hi)  matching GAIO.jl semantics.
        """
        p = np.asarray(point, dtype=F64)
        if not self.domain.contains_point(p):
            return None
        # multi_idx[i] = floor((p[i] - lo[i]) / cell_width[i])
        multi = np.floor(
            (p - self.domain.lo) / (2.0 * self.cell_radius)
        ).astype(I64)
        # Clamp to valid range (handles floating-point boundary edge cases)
        multi = np.clip(multi, I64(0), self.dims - I64(1))
        return self.multi_to_key(multi)

    def point_to_key_batch(self, points: NDArray[F64]) -> NDArray[I64]:
        """
        Vectorised version of :meth:`point_to_key` for an (N, n) point array.

        Points outside the domain are assigned key  -1.

        Parameters
        ----------
        points : ndarray, shape (N, n)

        Returns
        -------
        keys : ndarray of int64, shape (N,)
            -1 for out-of-domain points.
        """
        N = points.shape[0]
        keys = np.full(N, -1, dtype=I64)

        lo = self.domain.lo          # (n,)
        hi = self.domain.hi          # (n,)
        cw = 2.0 * self.cell_radius  # cell widths, (n,)

        in_domain = np.all((points >= lo) & (points < hi), axis=1)  # (N,)
        p_in = points[in_domain]  # (M, n)

        if p_in.shape[0] == 0:
            return keys

        multi = np.floor((p_in - lo) / cw).astype(I64)  # (M, n)
        multi = np.clip(multi, I64(0), self.dims - I64(1))
        keys[in_domain] = np.ravel_multi_index(multi.T, self.dims).astype(I64)
        return keys

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    def subdivide(self, dim: int) -> BoxPartition:
        """
        Return a new BoxPartition with resolution doubled along *dim*.
        The domain is unchanged.
        """
        new_dims = self.dims.copy()
        new_dims[dim] *= I64(2)
        return BoxPartition(self.domain, new_dims)

    def subdivide_all(self) -> BoxPartition:
        """Double resolution in every dimension simultaneously."""
        return BoxPartition(self.domain, self.dims * I64(2))

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def all_keys(self) -> NDArray[I64]:
        """Return a C-contiguous array  [0, 1, ..., size-1]  of all flat keys."""
        return np.arange(self.size, dtype=I64)

    def keys_in_box(self, query: Box) -> NDArray[I64]:
        """
        Return all flat keys whose corresponding cell intersects *query*.

        This is a brute-force O(size) scan.  Phase 3 will replace this
        with a CUDA kernel for high-dimensional grids.
        """
        if not self.domain.intersects(query):
            return np.empty(0, dtype=I64)

        # Compute multi-index range from query bounds
        lo_m = np.floor(
            (np.maximum(query.lo, self.domain.lo) - self.domain.lo)
            / (2.0 * self.cell_radius)
        ).astype(I64)
        hi_m = np.ceil(
            (np.minimum(query.hi, self.domain.hi) - self.domain.lo)
            / (2.0 * self.cell_radius)
        ).astype(I64)

        lo_m = np.clip(lo_m, I64(0), self.dims - I64(1))
        hi_m = np.clip(hi_m, I64(1), self.dims)

        # Build cartesian product of ranges per dimension
        ranges = [np.arange(lo_m[i], hi_m[i], dtype=I64) for i in range(self.ndim)]
        mesh = np.stack(
            [g.ravel() for g in np.meshgrid(*ranges, indexing="ij")], axis=1
        )  # (M, n)
        return np.ravel_multi_index(mesh.T, self.dims).astype(I64)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        dims_str = " × ".join(str(d) for d in self.dims.tolist())
        return f"BoxPartition([{dims_str}], domain={self.domain})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoxPartition):
            return NotImplemented
        return self.domain == other.domain and np.array_equal(self.dims, other.dims)

    def __hash__(self) -> int:
        return hash((self.domain, self.dims.tobytes()))
