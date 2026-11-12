# ruff: noqa: INP001
"""Peak resident memory of ``principal_axes`` on one million points.

The centered-point temporary is the same size as the input. Inputs are built
before the measurement, and one fresh process should be used per sample
because the allocator keeps the freed temporary in the resident set.
"""

from __future__ import annotations

from measure import peak_rss_delta
import numpy as np

from pyvista.core.utilities.points import principal_axes

N_POINTS = 1_000_000


def main() -> None:
    """Print the peak resident-memory increase, in bytes."""
    points = np.random.default_rng(0).random((N_POINTS, 3))

    def call() -> None:
        axes = principal_axes(points)
        if axes.shape != (3, 3):
            message = f'unexpected axes shape {axes.shape}'
            raise RuntimeError(message)

    print(peak_rss_delta(call))  # noqa: T201


if __name__ == '__main__':
    main()
