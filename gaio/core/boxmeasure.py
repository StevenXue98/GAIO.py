"""
gaio/core/boxmeasure.py
=======================
BoxMeasure — a discrete signed measure on a BoxPartition, stored as parallel
sorted ``int64`` key / ``float64`` weight arrays.

Correspondence with GAIO.jl
----------------------------
``BoxMeasure`` ↔ ``BoxMeasure{B,K,V,P,D}`` in ``src/boxmeasure.jl``.

Julia stores weights in an ``OrderedDict{key → weight}``.  Python uses two
parallel NumPy arrays (like BoxSet for keys) so that:

* scipy sparse matrix × measure is a plain ``mat @ weight_vec`` call.
* MPI scatter / gather of weights is a ``float64`` buffer send (Phase 4).
* Numba can access weights without Python object overhead (Phase 3).

The support is always a subset of ``partition``'s key space.  Keys outside
the support have weight 0 (uniform default for getitem).

Key invariant
-------------
``self._keys`` is always ``dtype=int64``, 1-D, C-contiguous, sorted, unique.
``self._weights`` is ``dtype=float64``, same length as ``self._keys``.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .box import Box, F64, I64
from .partition import BoxPartition
from .boxset import BoxSet


class BoxMeasure:
    """
    A discrete signed measure on a :class:`BoxPartition`.

    Each active cell carries a scalar weight; cells outside the support
    implicitly have weight 0.  Arithmetic is the usual vector-space
    operations on the weight array (addition, scalar multiplication, etc.).

    Parameters
    ----------
    partition : BoxPartition
    keys : array_like of int64
        Sorted unique flat keys for cells in the support.
    weights : array_like of float64
        Corresponding weights (same length as ``keys``).

    Examples
    --------
    >>> import numpy as np
    >>> from gaio.core.box import Box
    >>> from gaio.core.partition import BoxPartition
    >>> from gaio.core.boxset import BoxSet
    >>> from gaio.core.boxmeasure import BoxMeasure
    >>> domain = Box([0.0, 0.0], [1.0, 1.0])
    >>> P = BoxPartition(domain, [4, 4])
    >>> B = BoxSet.full(P)
    >>> mu = BoxMeasure.uniform(B)
    >>> len(mu)
    16
    >>> mu.total()
    16.0
    """

    __slots__ = ("partition", "_keys", "_weights")

    def __init__(
        self,
        partition: BoxPartition,
        keys: NDArray[I64],
        weights: NDArray[F64],
    ) -> None:
        self.partition: BoxPartition = partition
        self._keys: NDArray[I64] = np.ascontiguousarray(keys, dtype=I64)
        self._weights: NDArray[F64] = np.ascontiguousarray(weights, dtype=F64)
        if self._keys.shape != self._weights.shape:
            raise ValueError(
                f"keys (length {len(self._keys)}) and weights "
                f"(length {len(self._weights)}) must have the same length."
            )

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def uniform(cls, boxset: BoxSet, value: float = 1.0) -> BoxMeasure:
        """Uniform measure: every active cell gets weight *value*."""
        return cls(
            boxset.partition,
            boxset._keys.copy(),
            np.full(len(boxset), value, dtype=F64),
        )

    @classmethod
    def zeros(cls, boxset: BoxSet) -> BoxMeasure:
        """Zero measure on the support of *boxset*."""
        return cls.uniform(boxset, 0.0)

    @classmethod
    def from_boxset(cls, boxset: BoxSet, weights: NDArray[F64]) -> BoxMeasure:
        """
        Build a BoxMeasure from a BoxSet and a weight array.

        Parameters
        ----------
        boxset : BoxSet
            Defines the support (key ordering).
        weights : array_like, shape (len(boxset),)
            One weight per cell, in the same order as ``boxset._keys``.
        """
        return cls(boxset.partition, boxset._keys.copy(), np.asarray(weights, dtype=F64))

    # ------------------------------------------------------------------
    # Support conversion
    # ------------------------------------------------------------------

    def to_boxset(self) -> BoxSet:
        """Return the BoxSet of cells with non-zero weight."""
        nonzero = self._keys[self._weights != 0.0]
        return BoxSet(self.partition, nonzero)

    def support(self) -> BoxSet:
        """Alias for :meth:`to_boxset`."""
        return self.to_boxset()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def keys(self) -> NDArray[I64]:
        """Sorted unique flat-key array (read-only view)."""
        return self._keys

    @property
    def weights(self) -> NDArray[F64]:
        """Weight array, same order as ``keys`` (read-only view)."""
        return self._weights

    def __len__(self) -> int:
        return int(self._keys.size)

    # ------------------------------------------------------------------
    # Key-indexed access
    # ------------------------------------------------------------------

    def __getitem__(self, key: int) -> float:
        """Return weight at *key*, or 0.0 if key is outside the support."""
        idx = int(np.searchsorted(self._keys, key))
        if idx < len(self._keys) and self._keys[idx] == key:
            return float(self._weights[idx])
        return 0.0

    def __setitem__(self, key: int, value: float) -> None:
        """
        Set weight at *key*.  The key must already be in the support;
        use :meth:`union_support` to extend the support first.
        """
        idx = int(np.searchsorted(self._keys, key))
        if idx < len(self._keys) and self._keys[idx] == key:
            self._weights[idx] = float(value)
        else:
            raise KeyError(
                f"Key {key} is not in the support of this BoxMeasure. "
                "Extend the support first."
            )

    # ------------------------------------------------------------------
    # Vector-space arithmetic
    # ------------------------------------------------------------------

    def _binary_op(self, other: BoxMeasure, op) -> BoxMeasure:
        """Apply element-wise binary op after aligning supports."""
        if self.partition != other.partition:
            raise ValueError("BoxMeasures must share the same partition.")
        # Union of keys
        all_keys = np.union1d(self._keys, other._keys)
        w1 = np.array([self[int(k)] for k in all_keys], dtype=F64)
        w2 = np.array([other[int(k)] for k in all_keys], dtype=F64)
        result_weights = op(w1, w2)
        # Drop exact zeros to keep support minimal
        nonzero = result_weights != 0.0
        return BoxMeasure(self.partition, all_keys[nonzero], result_weights[nonzero])

    def __add__(self, other: BoxMeasure) -> BoxMeasure:
        return self._binary_op(other, np.add)

    def __sub__(self, other: BoxMeasure) -> BoxMeasure:
        return self._binary_op(other, np.subtract)

    def __neg__(self) -> BoxMeasure:
        return BoxMeasure(self.partition, self._keys.copy(), -self._weights)

    def __mul__(self, scalar: float) -> BoxMeasure:
        return BoxMeasure(self.partition, self._keys.copy(), self._weights * float(scalar))

    def __rmul__(self, scalar: float) -> BoxMeasure:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> BoxMeasure:
        return BoxMeasure(self.partition, self._keys.copy(), self._weights / float(scalar))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoxMeasure):
            return NotImplemented
        if self.partition != other.partition:
            return False
        # Both must have same keys and weights (after alignment)
        all_keys = np.union1d(self._keys, other._keys)
        for k in all_keys:
            if not np.isclose(self[int(k)], other[int(k)]):
                return False
        return True

    def __repr__(self) -> str:
        return f"BoxMeasure({len(self)} cells on {self.partition})"

    # ------------------------------------------------------------------
    # Norms and normalization
    # ------------------------------------------------------------------

    def norm(self, p: float = 2) -> float:
        """
        L^p norm of the density with respect to the Lebesgue measure.

        Matches Julia's ``LinearAlgebra.norm(boxmeas, p)``:
        ``||μ||_p = (∑ |w_i / vol^{1/p}|^p)^{1/p}``.
        """
        vol = self.partition.cell_volume
        scaled = self._weights / (vol ** (1.0 / p))
        return float(np.linalg.norm(scaled, ord=p))

    def normalize(self) -> BoxMeasure:
        """Return a new BoxMeasure with L2 norm equal to 1."""
        n = self.norm(2)
        if n == 0.0:
            raise ValueError("Cannot normalize a zero measure.")
        return self / n

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def total(self) -> float:
        """Sum of all weights (integration of the constant function 1)."""
        return float(np.sum(self._weights))

    def integrate(self, f: Callable | None = None) -> float:
        """
        Approximate  ∫ f(x) dμ(x)  by  ∑_i f(center_i) * weight_i.

        If ``f`` is ``None``, returns the total mass ``∑ weight_i``.

        Matches Julia's ``sum(f, boxmeas)``.
        """
        if f is None:
            return self.total()
        centers = np.stack([
            self.partition.key_to_box(int(k)).center for k in self._keys
        ])
        total = 0.0
        for c, w in zip(centers, self._weights):
            total += float(f(c)) * float(w)
        return total

    def __call__(self, region=None) -> float:
        """
        μ(region) — total mass of the measure restricted to *region*.

        Parameters
        ----------
        region : BoxSet or None
            If None, return total mass.  If BoxSet, restrict to cells
            in the intersection of the support with *region*.

        Matches Julia's ``(boxmeas::BoxMeasure)(boxset::Union{Box,BoxSet})``.
        """
        if region is None:
            return self.total()
        if isinstance(region, BoxSet):
            mask = np.isin(self._keys, region._keys)
            return float(np.sum(self._weights[mask]))
        raise TypeError(f"region must be a BoxSet, got {type(region)}")

    # ------------------------------------------------------------------
    # Density function
    # ------------------------------------------------------------------

    def density(self) -> Callable:
        """
        Return a callable  g(x)  such that  dμ/dx = g,  i.e.
        ``∫ f dμ = ∫ f(x) g(x) dx``.

        Evaluates g at a point x by looking up the cell it belongs to
        and returning  weight / cell_volume.
        """
        P = self.partition
        vol = P.cell_volume

        def eval_density(x):
            key = P.point_to_key(np.asarray(x, dtype=F64))
            if key is None:
                return 0.0
            return self[int(key)] / vol

        return eval_density
