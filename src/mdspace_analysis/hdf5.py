from __future__ import annotations

from pathlib import Path
from typing import Iterator, Literal

import h5py
import numpy as np

from .pdb import PdbAtom, parse_pdb_atoms
from .types import EulerTransform

Selection = Literal["all", "ca", "calpha", "c-alpha"] | None


class MdspaceHdf5:
    """Reader for MDSPACE HDF5 coordinate archives.

    Unit convention
    ---------------
    Coordinate datasets are in Angstroms:

    - /frames/raw
    - /frames/registered
    - /frames/rotated

    Image-pose metadata uses the MDSPACE/XMIPP convention:

    - Euler angles are in degrees.
    - shifts are stored in pixels.
    - /metadata/pixel_size stores Angstroms/pixel.

    Therefore, before applying an Euler or composed transform to coordinates,
    translations must be converted:

        shift_angstrom = shift_pixel * pixel_size

    The reference PDB stored in /metadata/reference_pdb is used to recover atom
    metadata and to apply selections such as C-alpha-only.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(f"HDF5 archive not found: {self.path}")

        self._h5 = h5py.File(self.path, "r")
        self._atoms_cache: list[PdbAtom] | None = None

    def close(self) -> None:
        self._h5.close()

    def __enter__(self) -> "MdspaceHdf5":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def pixel_size(self) -> float:
        """Pixel size in Angstroms/pixel."""

        return float(np.asarray(self._h5["/metadata/pixel_size"]))

    @property
    def reference_pdb_text(self) -> str:
        """Reference PDB content stored in the HDF5 archive."""

        dataset = "/metadata/reference_pdb"

        if dataset not in self._h5:
            raise KeyError(f"Dataset not found: {dataset}")

        data = self._h5[dataset][()]

        if isinstance(data, bytes):
            return data.decode("utf-8")

        return str(data)

    @property
    def atoms(self) -> list[PdbAtom]:
        """Atom metadata parsed from /metadata/reference_pdb."""

        if self._atoms_cache is None:
            self._atoms_cache = parse_pdb_atoms(self.reference_pdb_text)

        return self._atoms_cache

    @property
    def ca_indices(self) -> np.ndarray:
        """Indices of C-alpha atoms in the reference PDB."""

        return np.array(
            [atom.index for atom in self.atoms if atom.name == "CA"],
            dtype=int,
        )

    @property
    def n_atoms(self) -> int:
        """Number of atoms described by the reference PDB."""

        return len(self.atoms)

    @property
    def n_ca_atoms(self) -> int:
        """Number of C-alpha atoms described by the reference PDB."""

        return int(self.ca_indices.size)

    @property
    def n_frames(self) -> int:
        """Number of stored coordinate frames."""

        if "/frames/raw" in self._h5:
            return int(self._h5["/frames/raw"].shape[0])

        if "/frames/registered" in self._h5:
            return int(self._h5["/frames/registered"].shape[0])

        if "/frames/rotated" in self._h5:
            return int(self._h5["/frames/rotated"].shape[0])

        raise KeyError("No frame dataset found under /frames")

    def raw(self, frame: int, selection: Selection = None) -> np.ndarray:
        """Raw coordinates for one frame, in Angstroms."""

        return self._select(self._read_frame("/frames/raw", frame), selection)

    def registered(self, frame: int,
                   selection: Selection = None) -> np.ndarray:
        """Registered coordinates for one frame, in Angstroms."""

        return self._select(self._read_frame(
            "/frames/registered", frame), selection)

    def rotated(self, frame: int, selection: Selection = None) -> np.ndarray:
        """Rotated/image-pose coordinates for one frame, in Angstroms."""

        return self._select(self._read_frame(
            "/frames/rotated", frame), selection)

    def euler(self, frame: int) -> EulerTransform:
        """Euler angles and image-space shifts for one frame.

        Shifts are returned in pixels.
        """

        data = np.asarray(self._h5["/transforms/euler"]
                          [frame], dtype=float).reshape(-1)

        if data.shape[0] < 5:
            raise ValueError(
                f"Invalid Euler transform at frame {frame}: expected at least 5 values, "
                f"got {data.shape[0]}"
            )

        shift_z = float(data[5]) if data.shape[0] > 5 else 0.0

        return EulerTransform(
            rot=float(data[0]),
            tilt=float(data[1]),
            psi=float(data[2]),
            shift_x=float(data[3]),
            shift_y=float(data[4]),
            shift_z=shift_z,
        )

    def composed_transform_pixels(self, frame: int) -> np.ndarray:
        """Composed image-space transform for one frame.

        Returns a 4x4 matrix. The rotation part is unitless, but the translation
        column is stored in pixels.
        """

        return self._read_transform("/transforms/composed", frame)

    def composed_transform_angstrom(self, frame: int) -> np.ndarray:
        """Composed transform converted for direct use on atomic coordinates.

        Returns a 4x4 matrix whose translation column is in Angstroms.
        """

        transform = self.composed_transform_pixels(frame).copy()
        transform[:3, 3] *= self.pixel_size
        return transform

    def iter_frames(
        self,
        selection: Selection = None,
    ) -> Iterator[tuple[int, np.ndarray, np.ndarray | None, np.ndarray]]:
        """Iterate over available coordinate frames.

        Returns tuples:

            frame_index, raw, registered_or_none, rotated

        `registered` may be None for synthetic data-generation archives that do
        not store /frames/registered.
        """

        has_registered = "/frames/registered" in self._h5

        for frame in range(self.n_frames):
            raw = self.raw(frame, selection=selection)
            registered = self.registered(
                frame, selection=selection) if has_registered else None
            rotated = self.rotated(frame, selection=selection)

            yield frame, raw, registered, rotated

    def _read_frame(self, dataset: str, frame: int) -> np.ndarray:
        if dataset not in self._h5:
            raise KeyError(f"Dataset not found: {dataset}")

        data = np.asarray(self._h5[dataset][frame], dtype=float)

        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(
                f"Invalid coordinate frame in {dataset}[{frame}]: "
                f"expected shape (n_atoms, 3), got {data.shape}"
            )

        return data

    def _read_transform(self, dataset: str, frame: int) -> np.ndarray:
        if dataset not in self._h5:
            raise KeyError(f"Dataset not found: {dataset}")

        data = np.asarray(self._h5[dataset][frame], dtype=float)

        if data.shape == (3, 4):
            out = np.eye(4, dtype=float)
            out[:3, :4] = data
            return out

        if data.shape == (4, 4):
            return data

        raise ValueError(
            f"Invalid transform in {dataset}[{frame}]: expected 3x4 or 4x4, got {data.shape}"
        )

    def _select(self, coords: np.ndarray, selection: Selection) -> np.ndarray:
        if selection is None or selection == "all":
            return coords

        if selection in {"ca", "calpha", "c-alpha"}:
            if coords.shape[0] != self.n_atoms:
                raise ValueError(
                    "Cannot apply C-alpha selection: coordinate atom count does not match "
                    "the number of atoms in /metadata/reference_pdb. If the coordinate "
                    "frames are already C-alpha-only, the stored reference PDB should also "
                    "be C-alpha-only or an explicit atom mapping must be stored."
                )

            return coords[self.ca_indices]

        raise ValueError(f"Unknown selection: {selection!r}")
