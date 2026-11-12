# ruff: noqa: INP001
"""Runtime of ``_cell_edge_lengths`` on a 1,000,000-triangle PolyData.

The mesh is built and the function is called once before the timed call.
The timed call is what a repeated edge-length query pays after the mesh exists.
"""

from __future__ import annotations

import gc
import time

import numpy as np

import pyvista as pv
from pyvista.core.utilities._cell_lengths import _cell_edge_lengths

N_TRIANGLES = 1_000_000


def _mesh() -> pv.PolyData:
    generator = np.random.default_rng(0)
    points = generator.random((N_TRIANGLES, 3))
    index = np.arange(N_TRIANGLES)
    faces = np.column_stack(
        (
            np.full(N_TRIANGLES, 3, dtype=np.int64),
            index,
            (index + 1) % N_TRIANGLES,
            (index + 2) % N_TRIANGLES,
        )
    ).ravel()
    return pv.PolyData(points, faces=faces)


def _lengths(mesh: pv.PolyData) -> None:
    lengths = _cell_edge_lengths(mesh)
    if lengths.size != 3 * mesh.n_cells:
        message = f'unexpected edge count {lengths.size} for {mesh.n_cells} cells'
        raise RuntimeError(message)


def main() -> None:
    """Print one warmed call duration in milliseconds."""
    mesh = _mesh()
    _lengths(mesh)
    gc.collect()
    gc.disable()
    try:
        started = time.perf_counter()
        _lengths(mesh)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    finally:
        gc.enable()
    print(f'{elapsed_ms:.6f}')  # noqa: T201


if __name__ == '__main__':
    main()
