"""
gaio/core/boxset.py
===================
BoxSet — a finite, mutable-by-rebuild collection of cells from a BoxPartition.

Representation
--------------
Internally the active cells are stored as a **sorted, unique** ``int64``
NumPy array of flat keys.  This representation was chosen because:

* NumPy ``np.union1d / intersect1d / setdiff1d`` run in  O(n log n)  and
  produce sorted output — ideal for repeated set operations in subdivision
  loops.
* A contiguous ``int64`` array is directly passable to Numba CUDA kernels
  and MPI send/recv buffers (Phase 3 / Phase 4).
* ``np.searchsorted`` gives O(log n) membership tests without a hash table.

All set operations return **new** BoxSet objects; the original is never
mutated.  This makes it safe to share references across MPI ranks or GPU
memory without explicit copying.

Key invariant
-------------
``self._keys`` is always ``dtype=int64``, 1-D, C-contiguous, sorted, unique.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .box import Box, F64, I64
from .partition import BoxPartition


def _validated_keys(keys: NDArray | Iterable | int, size: int) -> NDArray[I64]:
    """Return sorted unique int64 array; raise if any key is out-of-range."""
    arr = np.unique(np.asarray(list(keys) if not hasattr(keys, "__len__") else keys, dtype=I64))
    if arr.size > 0 and (arr[0] < 0 or arr[-1] >= size):
        bad = arr[(arr < 0) | (arr >= size)]
        raise ValueError(
            f"Keys {bad[:5].tolist()} are out of range [0, {size})."
        )
    return arr


class BoxSet:
    """
    A finite set of grid cells drawn from a single :class:`BoxPartition`.

    Parameters
    ----------
    partition : BoxPartition
        The underlying grid.
    keys : array_like of int
        Flat cell keys.  Duplicates are silently removed; the array is
        sorted on construction.

    Examples
    --------
    >>> import numpy as np
    >>> from gaio.core.box import Box
    >>> from gaio.core.partition import BoxPartition
    >>> domain = Box([0.0, 0.0], [1.0, 1.0])
    >>> p = BoxPartition(domain, [4, 4])
    >>> s = BoxSet(p, np.arange(8))          # first 8 cells
    >>> len(s)
    8
    >>> 3 in s
    True
    >>> 9 in s
    False
    """

    __slots__ = ("partition", "_keys")

    def __init__(
        self,
        partition: BoxPartition,
        keys: NDArray[I64] | list[int] | range,
    ) -> None:
        self.partition: BoxPartition = partition
        raw = np.ascontiguousarray(keys, dtype=I64)
        self._keys: NDArray[I64] = np.unique(raw)

        if self._keys.size > 0 and (
            self._keys[0] < 0 or self._keys[-1] >= partition.size
        ):
            bad = self._keys[(self._keys < 0) | (self._keys >= partition.size)]
            raise ValueError(
                f"Keys {bad[:5].tolist()} are outside [0, {partition.size})."
            )

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def full(cls, partition: BoxPartition) -> BoxSet:
        """Return the BoxSet containing *all* cells of *partition*."""
        return cls(partition, partition.all_keys())

    @classmethod
    def empty(cls, partition: BoxPartition) -> BoxSet:
        """Return an empty BoxSet on *partition*."""
        return cls(partition, np.empty(0, dtype=I64))

    @classmethod
    def cover(cls, partition: BoxPartition, points: NDArray[F64]) -> BoxSet:
        """
        Return the smallest BoxSet whose cells collectively cover *points*.

        Points outside the domain are silently ignored.

        Parameters
        ----------
        partition : BoxPartition
        points : ndarray, shape (N, n)
        """
        keys = partition.point_to_key_batch(points)
        return cls(partition, keys[keys >= 0])

    @classmethod
    def from_box(cls, partition: BoxPartition, query: Box) -> BoxSet:
        """Return all cells that intersect *query*."""
        return cls(partition, partition.keys_in_box(query))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def keys(self) -> NDArray[I64]:
        """Sorted unique flat-key array (read-only view)."""
        return self._keys

    def __len__(self) -> int:
        return int(self._keys.size)

    def is_empty(self) -> bool:
        return self._keys.size == 0

    # ------------------------------------------------------------------
    # Membership
    # ------------------------------------------------------------------

    def __contains__(self, key: int) -> bool:
        """O(log n) membership test via binary search."""
        k = I64(key)
        idx = int(np.searchsorted(self._keys, k))
        return idx < len(self._keys) and self._keys[idx] == k

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self):
        """Iterate over flat keys in sorted order."""
        return iter(self._keys)

    def boxes(self):
        """Yield (key, Box) pairs for every active cell."""
        for k in self._keys:
            yield int(k), self.partition.key_to_box(int(k))

    # ------------------------------------------------------------------
    # Set algebra — all O(n log n), all return new BoxSet
    # ------------------------------------------------------------------

    def _check_compat(self, other: BoxSet) -> None:
        if self.partition != other.partition:
            raise ValueError(
                "Cannot combine BoxSets from different partitions.  "
                "Call .repartition() first if you need to merge sets "
                "at different resolutions."
            )

    def union(self, other: BoxSet) -> BoxSet:
        """A ∪ B"""
        self._check_compat(other)
        return BoxSet(self.partition, np.union1d(self._keys, other._keys))

    def intersection(self, other: BoxSet) -> BoxSet:
        """A ∩ B"""
        self._check_compat(other)
        return BoxSet(self.partition, np.intersect1d(self._keys, other._keys))

    def difference(self, other: BoxSet) -> BoxSet:
        """A \\ B"""
        self._check_compat(other)
        return BoxSet(self.partition, np.setdiff1d(self._keys, other._keys))

    def symmetric_difference(self, other: BoxSet) -> BoxSet:
        """A △ B  =  (A ∪ B) \\ (A ∩ B)"""
        self._check_compat(other)
        return BoxSet(
            self.partition,
            np.setxor1d(self._keys, other._keys),
        )

    def issubset(self, other: BoxSet) -> bool:
        """Return True if self ⊆ other."""
        self._check_compat(other)
        return np.setdiff1d(self._keys, other._keys).size == 0

    def issuperset(self, other: BoxSet) -> bool:
        """Return True if self ⊇ other."""
        return other.issubset(self)

    # Operator aliases
    def __or__(self, other: BoxSet) -> BoxSet:
        return self.union(other)

    def __and__(self, other: BoxSet) -> BoxSet:
        return self.intersection(other)

    def __sub__(self, other: BoxSet) -> BoxSet:
        return self.difference(other)

    def __xor__(self, other: BoxSet) -> BoxSet:
        return self.symmetric_difference(other)

    def __le__(self, other: BoxSet) -> bool:
        return self.issubset(other)

    def __ge__(self, other: BoxSet) -> bool:
        return self.issuperset(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoxSet):
            return NotImplemented
        return self.partition == other.partition and np.array_equal(
            self._keys, other._keys
        )

    # ------------------------------------------------------------------
    # Batch spatial accessors (vectorised — Numba-friendly shapes)
    # ------------------------------------------------------------------

    def centers(self) -> NDArray[F64]:
        """
        Return an (N, n) C-contiguous float64 array of cell centres.

        This is the primary way algorithms extract geometry from a BoxSet
        without touching Python-level Box objects.
        """
        if self._keys.size == 0:
            return np.empty((0, self.partition.ndim), dtype=F64)
        multi = np.stack(
            np.unravel_index(self._keys, self.partition.dims), axis=1
        ).astype(F64)  # (N, n)
        return np.ascontiguousarray(
            self.partition.domain.lo
            + 2.0 * self.partition.cell_radius * (multi + 0.5)
        )

    def cell_radius(self) -> NDArray[F64]:
        """
        Return the uniform cell half-width vector, shape (n,).

        All cells in a BoxPartition share the same radius, so this is a
        scalar (broadcast) property of the set, not per-cell.
        """
        return self.partition.cell_radius.copy()

    def bounds(self) -> tuple[NDArray[F64], NDArray[F64]]:
        """
        Return (lo, hi) bounding box of the *union* of all active cells.
        Returns (None, None) for an empty set.
        """
        if self._keys.size == 0:
            return (
                np.full(self.partition.ndim, np.inf, dtype=F64),
                np.full(self.partition.ndim, -np.inf, dtype=F64),
            )
        c = self.centers()
        r = self.cell_radius()
        return c.min(axis=0) - r, c.max(axis=0) + r

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    def subdivide(self, dim: int) -> BoxSet:
        """
        Return the corresponding BoxSet on a 2× finer partition along *dim*.

        Each active cell maps to exactly 2 children.  The spatial coverage
        is identical — the representation just becomes finer.

        Parameters
        ----------
        dim : int
            Dimension to refine.
        """
        new_partition = self.partition.subdivide(dim)
        if self._keys.size == 0:
            return BoxSet.empty(new_partition)

        # multi-indices on old partition: (N, n)
        old_multi = self.key_to_multi_batch()  # (N, n)

        # Each cell along `dim` spawns two children: 2*m and 2*m+1
        child_multi = np.repeat(old_multi, 2, axis=0)  # (2N, n)
        child_multi[0::2, dim] *= I64(2)
        child_multi[1::2, dim] = child_multi[1::2, dim] * I64(2) + I64(1)

        new_keys = np.ravel_multi_index(child_multi.T, new_partition.dims).astype(I64)
        return BoxSet(new_partition, new_keys)

    def subdivide_all(self) -> BoxSet:
        """Refine in every dimension simultaneously (2^n children per cell)."""
        result = self
        for dim in range(self.partition.ndim):
            result = result.subdivide(dim)
        return result

    def repartition(self, new_partition: BoxPartition) -> BoxSet:
        """
        Transfer this BoxSet to *new_partition* by spatial overlap.

        Useful when comparing sets at different resolutions or after a
        global refinement step.  A cell in new_partition is included if
        its centre lies inside any active cell of self.
        """
        if self._keys.size == 0:
            return BoxSet.empty(new_partition)
        c = self.centers()           # (N, n) centres of current active cells
        r = self.cell_radius()       # (n,) half-widths of current cells

        # For each new cell, check if its centre is covered by self
        new_centers = BoxSet.full(new_partition).centers()  # (M, n)
        r_new = new_partition.cell_radius

        # Expand current cell bounds slightly to catch shared edges
        eps = np.minimum(r, r_new) * 1e-10

        covered_keys: list[int] = []
        for j, nc in enumerate(new_centers):
            # Is nc inside any active cell?  Vectorised over N.
            lo = c - r - eps
            hi = c + r + eps
            if np.any(np.all((nc >= lo) & (nc < hi), axis=1)):
                covered_keys.append(j)

        return BoxSet(new_partition, np.array(covered_keys, dtype=I64))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def key_to_multi_batch(self) -> NDArray[I64]:
        """Return (N, n) multi-index array for all active keys."""
        return np.stack(
            np.unravel_index(self._keys, self.partition.dims), axis=1
        ).astype(I64)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BoxSet({len(self)} cells on {self.partition})"
        )

    def __hash__(self) -> int:
        return hash((self.partition, self._keys.tobytes()))
