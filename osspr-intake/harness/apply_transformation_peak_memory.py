# ruff: noqa: INP001
"""Peak resident memory of a 4x4 transform applied to one million points.

``apply_transformation_to_points`` pads float32 points into a float64
homogeneous ``(N, 4)`` array and multiplies a ``(1, 4, N)`` product. The
matrix is built outside the measurement. The function normalizes its scale
in place, which does not allocate.
"""

from __future__ import annotations

from measure import peak_rss_delta
import numpy as np

from pyvista.core.utilities.transformations import apply_transformation_to_points

N_POINTS = 1_000_000


def main() -> None:
    """Print the peak resident-memory increase, in bytes."""
    points = np.random.default_rng(0).random((N_POINTS, 3), dtype=np.float32)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = (1.0, 2.0, 3.0)

    def call() -> None:
        transformed = apply_transformation_to_points(matrix, points, inplace=False)
        if transformed.shape != points.shape:
            message = f'unexpected transformed shape {transformed.shape}'
            raise RuntimeError(message)

    print(peak_rss_delta(call))  # noqa: T201


if __name__ == '__main__':
    main()
