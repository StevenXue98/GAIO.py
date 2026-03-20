"""
gaio/core/box.py
================
Foundational Box (axis-aligned hyperrectangle) data structure.

Design constraints
------------------
* ``center`` and ``radius`` are always ``float64``, 1-D, C-contiguous.
  These invariants are enforced on construction so that every downstream
  Numba JIT function can rely on them without defensive casting.
* A Box represents the *half-open* interval  [c - r, c + r)  in each
  dimension, consistent with GAIO.jl and standard set-oriented numerics.
* No mutable state after construction — all operations return new Box
  objects so that BoxSet / MPI code can safely share references.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Canonical dtypes — import these in every module that touches raw arrays
# ---------------------------------------------------------------------------
F64 = np.float64
I64 = np.int64


class Box:
    """
    An *n*-dimensional axis-aligned box  B = [c - r, c + r).

    Parameters
    ----------
    center : array_like, shape (n,)
        Centre coordinates.  Stored as ``float64``.
    radius : array_like, shape (n,)
        Half-widths in each dimension.  Every component must be > 0.

    Examples
    --------
    >>> import numpy as np
    >>> b = Box([0.0, 0.0], [1.0, 1.0])
    >>> b.ndim
    2
    >>> b.volume
    4.0
    >>> b.contains_point(np.array([0.5, -0.5]))
    True
    >>> b.contains_point(np.array([1.0, 0.0]))   # right boundary is open
    False
    """

    __slots__ = ("center", "radius")

    def __init__(
        self,
        center: NDArray[F64] | list | tuple,
        radius: NDArray[F64] | list | tuple,
    ) -> None:
        c = np.ascontiguousarray(center, dtype=F64)
        r = np.ascontiguousarray(radius, dtype=F64)

        if c.ndim != 1 or r.ndim != 1:
            raise ValueError("center and radius must be 1-D arrays.")
        if c.shape != r.shape:
            raise ValueError(
                f"center (shape {c.shape}) and radius (shape {r.shape}) must match."
            )
        if np.any(r <= 0.0):
            raise ValueError("All radius components must be strictly positive.")

        self.center: NDArray[F64] = c
        self.radius: NDArray[F64] = r

    # ------------------------------------------------------------------
    # Read-only derived properties
    # ------------------------------------------------------------------

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return int(self.center.shape[0])

    @property
    def lo(self) -> NDArray[F64]:
        """Lower corner  c - r  (inclusive boundary)."""
        return self.center - self.radius

    @property
    def hi(self) -> NDArray[F64]:
        """Upper corner  c + r  (exclusive boundary)."""
        return self.center + self.radius

    @property
    def volume(self) -> float:
        """Lebesgue measure of the box."""
        return float(np.prod(2.0 * self.radius))

    @property
    def widths(self) -> NDArray[F64]:
        """Side lengths  2r  per dimension."""
        return 2.0 * self.radius

    # ------------------------------------------------------------------
    # Spatial queries
    # ------------------------------------------------------------------

    def contains_point(self, point: NDArray[F64]) -> bool:
        """Return True if *point* ∈ [lo, hi)  (half-open in every dimension)."""
        p = np.asarray(point, dtype=F64)
        return bool(np.all(p >= self.lo) and np.all(p < self.hi))

    def contains_box(self, other: Box) -> bool:
        """Return True if *other* ⊆ self (closed containment)."""
        return bool(np.all(other.lo >= self.lo) and np.all(other.hi <= self.hi))

    def intersects(self, other: Box) -> bool:
        """Return True if the two boxes have non-empty intersection."""
        return bool(np.all(self.lo < other.hi) and np.all(other.lo < self.hi))

    def intersection(self, other: Box) -> Box:
        """
        Return the intersection box, or raise ValueError if it is empty.

        Raises
        ------
        ValueError
            If the boxes do not overlap.
        """
        lo = np.maximum(self.lo, other.lo)
        hi = np.minimum(self.hi, other.hi)
        if np.any(lo >= hi):
            raise ValueError("Boxes do not intersect.")
        c = (lo + hi) * 0.5
        r = (hi - lo) * 0.5
        return Box(c, r)

    def bounding_box(self, other: Box) -> Box:
        """Return the smallest box containing both self and other."""
        lo = np.minimum(self.lo, other.lo)
        hi = np.maximum(self.hi, other.hi)
        c = (lo + hi) * 0.5
        r = (hi - lo) * 0.5
        return Box(c, r)

    # ------------------------------------------------------------------
    # Coordinate transforms (used by BoxMap sampling)
    # ------------------------------------------------------------------

    def rescale(self, unit_point: NDArray[F64]) -> NDArray[F64]:
        """
        Map a point from the unit box  [-1, 1]^n  into box coordinates.

        This is the primary way BoxMap implementations generate test points
        inside a cell.

        Parameters
        ----------
        unit_point : ndarray, shape (n,) or (N, n)
            Point(s) in  [-1, 1]^n.

        Returns
        -------
        ndarray
            Corresponding point(s) in  [lo, hi).
        """
        p = np.asarray(unit_point, dtype=F64)
        return self.center + p * self.radius

    def normalize(self, point: NDArray[F64]) -> NDArray[F64]:
        """
        Map a point from box coordinates into  [-1, 1]^n.

        Inverse of :meth:`rescale`.
        """
        p = np.asarray(point, dtype=F64)
        return (p - self.center) / self.radius

    # ------------------------------------------------------------------
    # Subdivision
    # ------------------------------------------------------------------

    def subdivide(self, dim: int) -> tuple[Box, Box]:
        """
        Bisect the box along *dim*, returning (left_child, right_child).

        Parameters
        ----------
        dim : int
            Dimension index in  [0, ndim).

        Returns
        -------
        left_child, right_child : Box
            Two boxes of equal size whose union equals self.
        """
        r = self.radius.copy()
        r[dim] *= 0.5
        c_lo = self.center.copy()
        c_hi = self.center.copy()
        c_lo[dim] -= r[dim]
        c_hi[dim] += r[dim]
        return Box(c_lo, r), Box(c_hi, r)

    def subdivide_all(self) -> list[Box]:
        """
        Bisect along every dimension, returning 2^n children covering self.
        """
        boxes: list[Box] = [self]
        for dim in range(self.ndim):
            boxes = [child for b in boxes for child in b.subdivide(dim)]
        return boxes

    # ------------------------------------------------------------------
    # Operator overloads
    # ------------------------------------------------------------------

    def __contains__(self, point: object) -> bool:
        return self.contains_point(np.asarray(point, dtype=F64))

    def __and__(self, other: Box) -> Box:
        return self.intersection(other)

    def __or__(self, other: Box) -> Box:
        return self.bounding_box(other)

    def __le__(self, other: Box) -> bool:
        """A <= B  iff  A ⊆ B."""
        return self.contains_box.__func__(other, self)  # type: ignore[attr-defined]

    def __ge__(self, other: Box) -> bool:
        """A >= B  iff  A ⊇ B."""
        return self.contains_box(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Box):
            return NotImplemented
        return bool(
            np.array_equal(self.center, other.center)
            and np.array_equal(self.radius, other.radius)
        )

    def __hash__(self) -> int:
        return hash((self.center.tobytes(), self.radius.tobytes()))

    def __repr__(self) -> str:
        return f"Box(center={self.center.tolist()}, radius={self.radius.tolist()})"
