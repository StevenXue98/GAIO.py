"""
gaio/viz/plot2d.py
==================
Matplotlib-based 2-D visualisation for BoxSet and BoxMeasure.

All functions return the matplotlib ``Axes`` object so callers can
compose plots by adding further artists (titles, colorbars, extra patches).

Design
------
* Uses ``matplotlib.collections.PolyCollection`` with a pre-built
  ``(N, 4, 2)`` vertex array instead of constructing N ``Rectangle``
  objects — a single vectorised pass for any N.
* Centers are computed via the same ``np.unravel_index`` formula used by
  ``BoxSet.centers()``, avoiding per-key Python calls to ``key_to_box``.
* 3-D BoxSets are projected onto a chosen pair of dimensions (default: 0,1).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from gaio.core.box import F64, I64
from gaio.core.boxset import BoxSet
from gaio.core.boxmeasure import BoxMeasure


def _require_matplotlib():
    try:
        import matplotlib
        return matplotlib
    except ImportError as exc:
        raise ImportError(
            "2-D plotting requires matplotlib.  Install with:\n"
            "    conda install matplotlib   or   pip install matplotlib"
        ) from exc


def _centers_from_measure(measure: BoxMeasure) -> np.ndarray:
    """Vectorised center lookup for a BoxMeasure — avoids per-key key_to_box calls."""
    P = measure.partition
    multi = np.stack(
        np.unravel_index(measure._keys, P.dims), axis=1
    ).astype(F64)  # (N, ndim)
    return P.domain.lo + 2.0 * P.cell_radius * (multi + 0.5)  # (N, ndim)


def _make_quad_verts(c0: np.ndarray, c1: np.ndarray,
                     r0: float, r1: float) -> np.ndarray:
    """
    Build an (N, 4, 2) float64 array of axis-aligned quad vertices.

    Each quad is the rectangle  [c0-r0, c0+r0] × [c1-r1, c1+r1].
    Vertices are ordered counter-clockwise starting at the bottom-left.
    No Python-level loop — fully vectorised NumPy.
    """
    N = len(c0)
    verts = np.empty((N, 4, 2), dtype=np.float64)
    verts[:, 0, 0] = c0 - r0   # BL x
    verts[:, 0, 1] = c1 - r1   # BL y
    verts[:, 1, 0] = c0 + r0   # BR x
    verts[:, 1, 1] = c1 - r1   # BR y
    verts[:, 2, 0] = c0 + r0   # TR x
    verts[:, 2, 1] = c1 + r1   # TR y
    verts[:, 3, 0] = c0 - r0   # TL x
    verts[:, 3, 1] = c1 + r1   # TL y
    return verts


def plot_boxset(
    boxset: BoxSet,
    ax=None,
    *,
    projection: Sequence[int] | None = None,
    color: str = "steelblue",
    alpha: float = 0.7,
    edge_color: str | None = "white",
    edge_width: float = 0.3,
    title: str | None = None,
    show: bool = False,
):
    """
    Draw a BoxSet as a collection of filled rectangles.

    Parameters
    ----------
    boxset : BoxSet
    ax : matplotlib.axes.Axes, optional
        Target axes.  Created if not provided.
    projection : sequence of 2 ints, optional
        Which two spatial dimensions to plot.  Default: ``[0, 1]``.
    color : str, optional
        Fill colour.  Default: ``'steelblue'``.
    alpha : float, optional
        Opacity.  Default: 0.7.
    edge_color : str or None, optional
        Rectangle edge colour.  Default: ``'white'``.
    edge_width : float, optional
        Edge linewidth.  Default: 0.3.
    title : str, optional
        Axes title.
    show : bool, optional
        Call ``plt.show()`` before returning.

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    dims = projection if projection is not None else [0, 1]
    d0, d1 = dims[0], dims[1]

    if ax is None:
        _, ax = plt.subplots()

    if boxset.is_empty():
        ax.set_aspect("equal")
        if title:
            ax.set_title(title)
        return ax

    centers = boxset.centers()          # (N, ndim) — vectorised
    r       = boxset.cell_radius()      # (ndim,)
    c0, c1  = centers[:, d0], centers[:, d1]
    r0, r1  = float(r[d0]),   float(r[d1])

    verts = _make_quad_verts(c0, c1, r0, r1)
    pc = PolyCollection(
        verts,
        facecolor=color,
        alpha=alpha,
        edgecolor=edge_color if edge_color else "none",
        linewidth=edge_width,
    )
    ax.add_collection(pc)
    ax.set_xlim(c0.min() - r0, c0.max() + r0)
    ax.set_ylim(c1.min() - r1, c1.max() + r1)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    if show:
        plt.show()
    return ax


def plot_boxmeasure(
    measure: BoxMeasure,
    ax=None,
    *,
    projection: Sequence[int] | None = None,
    colormap: str = "viridis",
    alpha: float = 1.0,
    edge_color: str | None = None,
    edge_width: float = 0.0,
    absolute_value: bool = False,
    title: str | None = None,
    colorbar: bool = True,
    show: bool = False,
):
    """
    Draw a BoxMeasure as colormapped rectangles.

    Weight-to-colour mapping uses ``matplotlib.cm`` normalised linearly
    from ``min(weights)`` to ``max(weights)``.

    Parameters
    ----------
    measure : BoxMeasure
    ax : matplotlib.axes.Axes, optional
    projection : sequence of 2 ints, optional
    colormap : str, optional
        Any matplotlib colourmap name.  Default: ``'viridis'``.
    alpha : float, optional
    edge_color : str or None, optional
    edge_width : float, optional
    absolute_value : bool, optional
        Plot ``|weights|`` instead of ``weights``.  Useful for eigenvectors.
    title : str, optional
    colorbar : bool, optional
        Add a colourbar to the figure.  Default: True.
    show : bool, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from matplotlib.collections import PolyCollection

    dims = projection if projection is not None else [0, 1]
    d0, d1 = dims[0], dims[1]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if len(measure) == 0:
        if title:
            ax.set_title(title)
        return ax

    centers = _centers_from_measure(measure)  # (N, ndim) — vectorised
    r   = measure.partition.cell_radius       # (ndim,)
    c0  = centers[:, d0]
    c1  = centers[:, d1]
    r0, r1 = float(r[d0]), float(r[d1])

    weights = np.abs(measure._weights) if absolute_value else measure._weights
    norm    = mcolors.Normalize(vmin=weights.min(), vmax=weights.max())
    cmap    = cm.get_cmap(colormap)
    colors  = cmap(norm(weights))              # (N, 4) RGBA

    verts = _make_quad_verts(c0, c1, r0, r1)
    pc = PolyCollection(
        verts,
        facecolor=colors,
        alpha=alpha,
        edgecolor=edge_color if edge_color else "none",
        linewidth=edge_width,
    )
    ax.add_collection(pc)
    ax.set_xlim(c0.min() - r0, c0.max() + r0)
    ax.set_ylim(c1.min() - r1, c1.max() + r1)
    ax.set_aspect("equal")

    if colorbar:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax)

    if title:
        ax.set_title(title)
    if show:
        plt.show()
    return ax


def plot_morse_tiles(
    tiles: BoxMeasure,
    ax=None,
    *,
    projection: Sequence[int] | None = None,
    colormap: str = "tab10",
    alpha: float = 0.8,
    edge_color: str | None = "white",
    edge_width: float = 0.3,
    title: str | None = None,
    show: bool = False,
):
    """
    Draw Morse tiles with each component in a distinct categorical colour.

    The integer weights from ``morse_tiles()`` are used as category indices
    into the chosen qualitative colourmap.

    Parameters
    ----------
    tiles : BoxMeasure
        Result of ``gaio.morse_tiles(F, B)``.
    ax, projection, alpha, edge_color, edge_width, title, show
        Same as :func:`plot_boxmeasure`.
    colormap : str, optional
        Qualitative colourmap.  Default: ``'tab10'`` (10 distinct colours).

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.collections import PolyCollection

    dims = projection if projection is not None else [0, 1]
    d0, d1 = dims[0], dims[1]

    if ax is None:
        _, ax = plt.subplots()

    if len(tiles) == 0:
        if title:
            ax.set_title(title)
        return ax

    centers = _centers_from_measure(tiles)  # (N, ndim) — vectorised
    r   = tiles.partition.cell_radius
    c0, c1 = centers[:, d0], centers[:, d1]
    r0, r1 = float(r[d0]), float(r[d1])

    labels  = tiles._weights.astype(int)
    n_comp  = labels.max()
    cmap    = cm.get_cmap(colormap, n_comp)
    colors  = cmap((labels - 1) % n_comp)          # 1-indexed → 0-indexed

    verts = _make_quad_verts(c0, c1, r0, r1)
    pc = PolyCollection(
        verts,
        facecolor=colors,
        alpha=alpha,
        edgecolor=edge_color if edge_color else "none",
        linewidth=edge_width,
    )
    ax.add_collection(pc)
    ax.set_xlim(c0.min() - r0, c0.max() + r0)
    ax.set_ylim(c1.min() - r1, c1.max() + r1)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    if show:
        plt.show()
    return ax
