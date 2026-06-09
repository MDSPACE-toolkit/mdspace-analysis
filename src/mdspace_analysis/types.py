from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EulerTransform:
    """Euler angles and image-space shifts for one MDSPACE frame.

    Angles are in degrees.

    Shifts are stored in pixels, following the MDSPACE/XMIPP image-metadata
    convention. Use `pixel_size` to convert shifts to Angstroms before applying
    them to atomic coordinates.
    """

    rot: float
    tilt: float
    psi: float
    shift_x: float
    shift_y: float
    shift_z: float = 0.0

    def shifts_pixels(self) -> tuple[float, float, float]:
        return self.shift_x, self.shift_y, self.shift_z

    def shifts_angstrom(self, pixel_size: float) -> tuple[float, float, float]:
        return (
            self.shift_x * pixel_size,
            self.shift_y * pixel_size,
            self.shift_z * pixel_size,
        )


@dataclass(frozen=True)
class PairedAtomSelection:
    """Paired atom selection.

    left[i] and right[i] refer to the same biological atom.
    """

    left: NDArray[np.int_]
    right: NDArray[np.int_]

    @property
    def size(self) -> int:
        return int(self.left.size)

    def __len__(self) -> int:
        return self.size
