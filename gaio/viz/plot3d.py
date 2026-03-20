"""
gaio/viz/plot3d.py
==================
PyVista-based 3-D visualisation for BoxSet and BoxMeasure.

Architecture
------------
Rather than adding N separate cube ``Actor`` objects (O(N) Python overhead),
we build a **single** ``pyvista.UnstructuredGrid`` with N VTK_HEXAHEDRON
cells.  PyVista renders the entire set in one draw call via VTK's pipeline.

Vertex layout (VTK_HEXAHEDRON, 8 nodes)
----------------------------------------
VTK hexahedron node ordering (canonical)::

         7 ---- 6
        /|      /|
       4 +--- 5  |
       | 3 ---+- 2
       |/     | /
       0 ---- 1

    Node 0: (cx - r0, cy - r1, cz - r2)
    Node 1: (cx + r0, cy - r1, cz - r2)
    Node 2: (cx + r0, cy + r1, cz - r2)
    Node 3: (cx - r0, cy + r1, cz - r2)
    Node 4: (cx - r0, cy - r1, cz + r2)
    Node 5: (cx + r0, cy - r1, cz + r2)
    Node 6: (cx + r0, cy + r1, cz + r2)
    Node 7: (cx - r0, cy + r1, cz + r2)

Memory layout
-------------
For N boxes:
    points : (8*N, 3)  float64   — all vertex coordinates, row-major
    cells  : (N, 9)    int64     — [8, v0..v7] for each hex cell
    offset : each box i uses vertices [8i .. 8i+7]

This layout is fully vectorised (NumPy broadcasting, zero Python loop)
and scales to hundreds of thousands of boxes without performance issues.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from gaio.core.boxset import BoxSet
from gaio.core.boxmeasure import BoxMeasure


def _require_pyvista():
    try:
        import pyvista
        return pyvista
    except ImportError as exc:
        raise ImportError(
            "3-D plotting requires pyvista.  Install with:\n"
            "    conda install -c conda-forge pyvista   or   pip install pyvista"
        ) from exc


# ---------------------------------------------------------------------------
# Core conversion helper
# ---------------------------------------------------------------------------

def boxset_to_pyvista(boxset: BoxSet, projection: Sequence[int] | None = None):
    """
    Convert a BoxSet (or 3-D projection of an n-D BoxSet) to a
    ``pyvista.UnstructuredGrid`` of VTK_HEXAHEDRON cells.

    Parameters
    ----------
    boxset : BoxSet
        Source box set.  Must have at least 3 spatial dimensions, or
        ``projection`` must select exactly 3 dimensions from a higher-D set.
    projection : sequence of 3 ints, optional
        Which three spatial dimensions to use.  Default: ``[0, 1, 2]``.

    Returns
    -------
    pyvista.UnstructuredGrid
        One VTK_HEXAHEDRON cell per box.  The grid has no scalar arrays;
        add them afterwards with ``grid.cell_data['weight'] = ...``.

    Raises
    ------
    ValueError
        If the BoxSet is empty or has fewer than 3 spatial dimensions.
    """
    pv = _require_pyvista()

    dims = list(projection) if projection is not None else [0, 1, 2]
    if len(dims) != 3:
        raise ValueError(f"projection must select exactly 3 dimensions, got {dims}.")
    d0, d1, d2 = dims

    if boxset.is_empty():
        return pv.UnstructuredGrid()

    centers = boxset.centers()          # (N, n)
    r       = boxset.cell_radius()      # (n,)
    N       = centers.shape[0]

    c0, c1, c2 = centers[:, d0], centers[:, d1], centers[:, d2]
    r0, r1, r2 = float(r[d0]), float(r[d1]), float(r[d2])

    # ── Build (8*N, 3) points array via broadcasting (no Python loop) ──────
    # offsets for each of the 8 hex corners relative to cell center
    # shape: (8, 3) — each row is (±r0, ±r1, ±r2)
    _sign = np.array([
        [-1, -1, -1],   # 0
        [+1, -1, -1],   # 1
        [+1, +1, -1],   # 2
        [-1, +1, -1],   # 3
        [-1, -1, +1],   # 4
        [+1, -1, +1],   # 5
        [+1, +1, +1],   # 6
        [-1, +1, +1],   # 7
    ], dtype=np.float64)  # (8, 3)

    radii = np.array([r0, r1, r2], dtype=np.float64)  # (3,)
    offsets = _sign * radii                            # (8, 3) broadcast

    # centers[:, [d0,d1,d2]]: (N, 3); offsets: (8, 3)
    # result[i, j] = centers[i] + offsets[j]  →  (N, 8, 3)
    cx = np.stack([c0, c1, c2], axis=1)              # (N, 3)
    verts = cx[:, np.newaxis, :] + offsets[np.newaxis, :, :]  # (N, 8, 3)
    points = verts.reshape(8 * N, 3)                  # (8*N, 3) C-contiguous

    # ── Build VTK cell connectivity array ──────────────────────────────────
    # Each hex cell: [8, v0, v1, ..., v7]  (9 ints per cell)
    # Vertex i of box k is at global index 8*k + i
    base = (np.arange(N, dtype=np.int64) * 8)         # (N,)
    local = np.arange(8, dtype=np.int64)               # (8,)
    vertex_ids = base[:, np.newaxis] + local[np.newaxis, :]  # (N, 8)

    count_col = np.full((N, 1), 8, dtype=np.int64)
    cells_flat = np.hstack([count_col, vertex_ids]).ravel()  # (9*N,)

    # Cell types: all VTK_HEXAHEDRON = 12
    celltypes = np.full(N, 12, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells_flat, celltypes, points)
    return grid


# ---------------------------------------------------------------------------
# High-level plotting functions
# ---------------------------------------------------------------------------

def plot_boxset_3d(
    boxset: BoxSet,
    plotter=None,
    *,
    projection: Sequence[int] | None = None,
    color: str = "steelblue",
    opacity: float = 0.7,
    show_edges: bool = True,
    edge_color: str = "white",
    title: str | None = None,
    show: bool = False,
    window_size: tuple[int, int] = (800, 600),
    off_screen: bool = False,
):
    """
    Draw a BoxSet as a single pyvista UnstructuredGrid of hexahedral cells.

    Parameters
    ----------
    boxset : BoxSet
    plotter : pyvista.Plotter, optional
        Existing plotter to add geometry to.  Created if not provided.
    projection : sequence of 3 ints, optional
        Which three spatial dimensions to plot.  Default: ``[0, 1, 2]``.
    color : str, optional
        Fill colour.  Default: ``'steelblue'``.
    opacity : float, optional
        Opacity.  Default: 0.7.
    show_edges : bool, optional
        Draw cell edges.  Default: True.
    edge_color : str, optional
        Edge colour.  Default: ``'white'``.
    title : str, optional
        Window title / text annotation.
    show : bool, optional
        Call ``plotter.show()`` before returning.
    window_size : tuple of 2 ints, optional
        Plotter window size in pixels.  Default: ``(800, 600)``.
    off_screen : bool, optional
        Render off-screen (no display window; useful for saving images).

    Returns
    -------
    pyvista.Plotter
    """
    pv = _require_pyvista()

    if plotter is None:
        plotter = pv.Plotter(window_size=window_size, off_screen=off_screen)

    if boxset.is_empty():
        if title:
            plotter.add_title(title)
        return plotter

    grid = boxset_to_pyvista(boxset, projection=projection)
    plotter.add_mesh(
        grid,
        color=color,
        opacity=opacity,
        show_edges=show_edges,
        edge_color=edge_color,
    )
    if title:
        plotter.add_title(title)
    if show:
        plotter.show()
    return plotter


def plot_boxmeasure_3d(
    measure: BoxMeasure,
    plotter=None,
    *,
    projection: Sequence[int] | None = None,
    colormap: str = "viridis",
    opacity: float = 1.0,
    show_edges: bool = False,
    edge_color: str = "white",
    absolute_value: bool = False,
    title: str | None = None,
    scalar_bar: bool = True,
    scalar_bar_title: str = "weight",
    show: bool = False,
    window_size: tuple[int, int] = (800, 600),
    off_screen: bool = False,
):
    """
    Draw a BoxMeasure as colormapped hexahedral cells.

    Parameters
    ----------
    measure : BoxMeasure
    plotter : pyvista.Plotter, optional
    projection : sequence of 3 ints, optional
    colormap : str, optional
        Any matplotlib/pyvista colourmap name.  Default: ``'viridis'``.
    opacity : float, optional
    show_edges : bool, optional
    edge_color : str, optional
    absolute_value : bool, optional
        Plot ``|weights|``.  Useful for eigenvectors.
    title : str, optional
    scalar_bar : bool, optional
        Show a colour scale bar.  Default: True.
    scalar_bar_title : str, optional
        Label on the scalar bar.  Default: ``'weight'``.
    show : bool, optional
    window_size, off_screen
        Passed to new Plotter if created.

    Returns
    -------
    pyvista.Plotter
    """
    pv = _require_pyvista()

    if plotter is None:
        plotter = pv.Plotter(window_size=window_size, off_screen=off_screen)

    if len(measure) == 0:
        if title:
            plotter.add_title(title)
        return plotter

    # Build BoxSet from measure keys
    P = measure.partition
    from gaio.core.boxset import BoxSet
    import numpy as np
    keys_arr = np.asarray(list(measure._keys), dtype=np.int64)
    bs = BoxSet(P, keys_arr)

    grid = boxset_to_pyvista(bs, projection=projection)

    weights = np.abs(measure._weights) if absolute_value else measure._weights
    # Cell scalar — one value per hex cell (already in key order)
    grid.cell_data[scalar_bar_title] = weights.astype(np.float64)

    plotter.add_mesh(
        grid,
        scalars=scalar_bar_title,
        cmap=colormap,
        opacity=opacity,
        show_edges=show_edges,
        edge_color=edge_color,
        show_scalar_bar=scalar_bar,
        scalar_bar_args={"title": scalar_bar_title} if scalar_bar else {},
    )
    if title:
        plotter.add_title(title)
    if show:
        plotter.show()
    return plotter


def plot_morse_tiles_3d(
    tiles: BoxMeasure,
    plotter=None,
    *,
    projection: Sequence[int] | None = None,
    colormap: str = "tab10",
    opacity: float = 0.8,
    show_edges: bool = True,
    edge_color: str = "white",
    title: str | None = None,
    show: bool = False,
    window_size: tuple[int, int] = (800, 600),
    off_screen: bool = False,
):
    """
    Draw 3-D Morse tiles with each component in a distinct categorical colour.

    Parameters
    ----------
    tiles : BoxMeasure
        Result of ``gaio.morse_tiles(F, B)``.  Integer weights = component index.
    plotter, projection, opacity, show_edges, edge_color, title, show,
    window_size, off_screen
        Same as :func:`plot_boxmeasure_3d`.
    colormap : str, optional
        Qualitative colourmap.  Default: ``'tab10'``.

    Returns
    -------
    pyvista.Plotter
    """
    pv = _require_pyvista()

    if plotter is None:
        plotter = pv.Plotter(window_size=window_size, off_screen=off_screen)

    if len(tiles) == 0:
        if title:
            plotter.add_title(title)
        return plotter

    import numpy as np
    from gaio.core.boxset import BoxSet

    P = tiles.partition
    keys_arr = np.asarray(list(tiles._keys), dtype=np.int64)
    bs = BoxSet(P, keys_arr)
    grid = boxset_to_pyvista(bs, projection=projection)

    labels = tiles._weights.astype(np.int64)
    grid.cell_data["component"] = labels

    plotter.add_mesh(
        grid,
        scalars="component",
        cmap=colormap,
        opacity=opacity,
        show_edges=show_edges,
        edge_color=edge_color,
        show_scalar_bar=True,
        scalar_bar_args={"title": "component"},
    )
    if title:
        plotter.add_title(title)
    if show:
        plotter.show()
    return plotter
