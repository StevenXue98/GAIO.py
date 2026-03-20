"""
gaio.viz — Visualisation utilities
====================================
2-D (matplotlib)
    plot_boxset         — BoxSet as coloured rectangles
    plot_boxmeasure     — BoxMeasure as colormapped rectangles
    plot_morse_tiles    — Morse tiles with categorical colours

3-D (pyvista / VTK)
    boxset_to_pyvista   — Convert BoxSet → pyvista.UnstructuredGrid
    plot_boxset_3d      — BoxSet as hexahedral cells
    plot_boxmeasure_3d  — BoxMeasure with scalar colormap
    plot_morse_tiles_3d — Morse tiles with categorical colours

All functions are imported lazily from their sub-modules so that
neither matplotlib nor pyvista is required at import time.
"""
from gaio.viz.plot2d import (
    plot_boxset,
    plot_boxmeasure,
    plot_morse_tiles,
)
from gaio.viz.plot3d import (
    boxset_to_pyvista,
    plot_boxset_3d,
    plot_boxmeasure_3d,
    plot_morse_tiles_3d,
)

__all__ = [
    # 2-D
    "plot_boxset",
    "plot_boxmeasure",
    "plot_morse_tiles",
    # 3-D
    "boxset_to_pyvista",
    "plot_boxset_3d",
    "plot_boxmeasure_3d",
    "plot_morse_tiles_3d",
]
