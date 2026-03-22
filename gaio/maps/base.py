"""
gaio/maps/base.py
=================
SampledBoxMap — the core BoxMap implementation.

A SampledBoxMap discretises a continuous map ``f: ℝⁿ → ℝⁿ`` onto a
BoxPartition by:

1. For each active cell in the source BoxSet, generate M test points by
   rescaling a fixed set of unit-cube points into the cell's coordinates.
2. Apply ``f`` to each test point.
3. Record which partition cells contain the image points.

The result is an **outer approximation**: every cell that *could* contain a
true image point is included; none are missed.

Phase 3 target
--------------
The inner loop ``for i, p in enumerate(test_pts)`` in
:meth:`SampledBoxMap.map_boxes` is the performance bottleneck.  It will be
replaced by a ``@cuda.jit`` kernel in Phase 3.  The surrounding NumPy code
(test-point generation and key lookup) is already written in
device-friendly shapes to minimise refactoring.

Correspondence with GAIO.jl
----------------------------
``SampledBoxMap`` ↔ ``SampledBoxMap{N,T}`` (boxmap_sampled.jl)

The Julia struct stores ``domain_points`` and ``image_points`` as closures.
Here we store only ``unit_points`` (the raw unit-cube grid) and perform the
rescaling inline in :meth:`map_boxes`, which is simpler and equally
Phase-3-friendly.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gaio.core.box import Box, F64, I64
from gaio.core.boxset import BoxSet


class SampledBoxMap:
    """
    A BoxMap that uses a fixed set of test points to discretise ``f``.

    Parameters
    ----------
    f : callable
        The dynamical system map.  Signature::

            f(x: ndarray[float64, shape (n,)]) -> array_like of shape (n,)

        Must accept and return 1-D float64-compatible arrays.
    domain : Box
        The spatial domain.  Image points outside this box receive key -1
        in :meth:`BoxPartition.point_to_key_batch` and are silently dropped.
    unit_points : ndarray, shape (M, n)
        Test points in the unit cube ``[-1, 1]^n``.  For each active cell
        with centre *c* and radius *r*, the actual test points are::

            test_pt[m] = c + unit_points[m] * r

        This is exactly ``Box.rescale(unit_points[m])`` applied to the cell.

    Examples
    --------
    >>> import numpy as np
    >>> from gaio.core.box import Box
    >>> from gaio.core.partition import BoxPartition
    >>> from gaio.core.boxset import BoxSet
    >>> domain = Box([0.0, 0.0], [1.0, 1.0])
    >>> f = lambda x: x ** 2
    >>> unit_pts = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    >>> g = SampledBoxMap(f, domain, unit_pts)
    >>> partition = BoxPartition(domain, [4, 4])
    >>> source = BoxSet.full(partition)
    >>> image = g(source)
    >>> isinstance(image, BoxSet)
    True
    """

    __slots__ = ("map", "domain", "_unit_points")

    def __init__(
        self,
        f,
        domain: Box,
        unit_points: NDArray[F64],
    ) -> None:
        self.map = f
        self.domain: Box = domain
        self._unit_points: NDArray[F64] = np.ascontiguousarray(unit_points, dtype=F64)
        if self._unit_points.ndim != 2:
            raise ValueError(
                f"unit_points must be a 2-D array of shape (M, n), "
                f"got shape {self._unit_points.shape}."
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_test_points(self) -> int:
        """Number of test points per cell."""
        return int(self._unit_points.shape[0])

    @property
    def ndim(self) -> int:
        """Spatial dimension."""
        return self.domain.ndim

    # ------------------------------------------------------------------
    # Core operation
    # ------------------------------------------------------------------

    def __call__(self, source: BoxSet) -> BoxSet:
        """Apply the BoxMap: return the outer-approximation image of *source*."""
        return self.map_boxes(source)

    def _apply_map(self, test_pts: NDArray[F64]) -> NDArray[F64]:
        """
        Apply ``self.map`` to every row of *test_pts*.

        This is Stage 2 of the three-stage pipeline shared by
        :meth:`map_boxes` and :func:`~gaio.transfer.operator._build_transitions`.
        Subclasses (:class:`~gaio.cuda.AcceleratedBoxMap`) override this
        method to dispatch to GPU or CPU-parallel backends.

        Parameters
        ----------
        test_pts : ndarray, shape (N, n), float64, C-contiguous

        Returns
        -------
        ndarray, shape (N, n), float64
        """
        N, n = test_pts.shape
        mapped = np.empty((N, n), dtype=F64)
        for i, p in enumerate(test_pts):
            mapped[i] = np.asarray(self.map(p), dtype=F64)
        return mapped

    def map_boxes(self, source: BoxSet) -> BoxSet:
        """
        Compute the outer-approximation image of *source*.

        Algorithm
        ---------
        1. **Vectorised** test-point generation: shape ``(K*M, n)``.
        2. Apply ``self.map`` via :meth:`_apply_map` — overridden by
           :class:`~gaio.cuda.AcceleratedBoxMap` to use GPU/CPU backends.
        3. **Vectorised** partition lookup via
           :meth:`BoxPartition.point_to_key_batch`.

        Parameters
        ----------
        source : BoxSet

        Returns
        -------
        BoxSet
            Image cells on the same partition as *source*.
        """
        P = source.partition
        if len(source) == 0:
            return BoxSet.empty(P)

        unit_pts = self._unit_points  # (M, n)
        cell_r = P.cell_radius        # (n,)

        centers = source.centers()   # (K, n)
        K = len(centers)
        M = self.n_test_points
        n = P.ndim

        # ── Stage 1: generate all test points in one vectorised step ─────────
        test_pts = (
            centers[:, np.newaxis, :]
            + unit_pts[np.newaxis, :, :] * cell_r[np.newaxis, np.newaxis, :]
        ).reshape(K * M, n)

        # ── Stage 2: apply map (CPU loop or GPU kernel via _apply_map) ───────
        mapped = self._apply_map(test_pts)

        # ── Stage 3: find which cells were hit ───────────────────────────────
        hit_keys = P.point_to_key_batch(mapped)   # -1 for out-of-domain
        valid = hit_keys[hit_keys >= 0]
        return BoxSet(P, np.unique(valid).astype(I64))

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SampledBoxMap("
            f"n_test_points={self.n_test_points}, "
            f"domain={self.domain})"
        )
