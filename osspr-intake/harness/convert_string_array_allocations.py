# ruff: noqa: INP001
"""Peak live Python blocks while converting 100,000 strings to VTK.

``convert_string_array`` builds ``arr.reshape(-1).tolist()`` and then walks
that list into ``vtkStringArray.SetValue``. The list keeps every Python
string alive at once. The NumPy array is built before the measurement.
"""

from __future__ import annotations

from measure import peak_allocation_delta
import numpy as np

from pyvista.core.utilities.arrays import convert_string_array

N_STRINGS = 100_000


def main() -> None:
    """Print the peak increase in live Python blocks."""
    labels = np.array([f'region-{index:05d}' for index in range(N_STRINGS)])

    def call() -> None:
        vtk_array = convert_string_array(labels)
        if vtk_array.GetNumberOfValues() != N_STRINGS:
            message = f'unexpected string count {vtk_array.GetNumberOfValues()}'
            raise RuntimeError(message)

    print(peak_allocation_delta(call))  # noqa: T201


if __name__ == '__main__':
    main()
