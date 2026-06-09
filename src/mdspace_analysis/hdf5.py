from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Iterator, Literal
from difflib import SequenceMatcher

import h5py
import numpy as np
from numpy.typing import NDArray
import re

from .pdb import PdbAtom, parse_pdb_atoms
from .types import EulerTransform, PairedAtomSelection

Selection = (
    Literal["all", "ca", "calpha", "c-alpha"]
    | Sequence[int]
    | NDArray[np.integer]
    | tuple[object, ...]
    | list[object]
    | None
)


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

    @property
    def has_image_paths(self) -> bool:
        """Whether the archive stores image paths in /metadata/image_path."""

        return "/metadata/image_path" in self._h5

    def image_path(self, frame: int) -> str:
        """EM image path associated with one frame.

        The path is read from /metadata/image_path.
        """

        dataset = "/metadata/image_path"

        if dataset not in self._h5:
            raise KeyError(f"Dataset not found: {dataset}")

        if frame < 0 or frame >= self.n_frames:
            raise IndexError(
                f"Frame index out of range: {frame}; archive has {
                    self.n_frames} frames"
            )

        data = self._h5[dataset][frame]

        if isinstance(data, bytes):
            return data.decode("utf-8")

        return str(data)

    def image_paths(self) -> list[str]:
        """All EM image paths stored in /metadata/image_path."""

        dataset = "/metadata/image_path"

        if dataset not in self._h5:
            raise KeyError(f"Dataset not found: {dataset}")

        values = self._h5[dataset][()]

        paths = []
        for value in values:
            if isinstance(value, bytes):
                paths.append(value.decode("utf-8"))
            else:
                paths.append(str(value))

        return paths

    def image_name(self, frame: int) -> str:
        """Basename of the EM image path associated with one frame."""

        return Path(self.image_path(frame)).name

    def image_stem(self, frame: int) -> str:
        """Filename stem of the EM image path associated with one frame."""

        return Path(self.image_path(frame)).stem

    def image_index(self, frame: int) -> int:
        """Integer image index extracted from the image filename stem.

        This accepts filenames such as:

        - 00042.mrc
        - image_00042.mrc
        - inputEm_000001.spi

        The last group of digits in the filename stem is used.
        """

        stem = self.image_stem(frame)
        matches = re.findall(r"\d+", stem)

        if not matches:
            raise ValueError(
                f"Cannot extract numeric image index from image path: "
                f"{self.image_path(frame)!r}"
            )

        return int(matches[-1])

    def frame_by_image_name(self) -> dict[str, int]:
        """Map image filename basename to archive frame index."""

        mapping: dict[str, int] = {}

        for frame in range(self.n_frames):
            name = self.image_name(frame)

            if name in mapping:
                raise ValueError(
                    f"Duplicate image filename in archive: {
                        name!r}")

            mapping[name] = frame

        return mapping

    def frame_by_image_index(self) -> dict[int, int]:
        """Map numeric image index to archive frame index."""

        mapping: dict[int, int] = {}

        for frame in range(self.n_frames):
            index = self.image_index(frame)

            if index in mapping:
                raise ValueError(f"Duplicate image index in archive: {index}")

            mapping[index] = frame

        return mapping

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

    def _selection_to_indices(self, selection: Selection,
                              n_atoms: int) -> np.ndarray:
        if selection is None:
            return np.arange(n_atoms, dtype=int)

        if isinstance(selection, str):
            normalized = selection.lower()

            if normalized == "all":
                return np.arange(n_atoms, dtype=int)

            if normalized in {"ca", "calpha", "c-alpha"}:
                return np.asarray(self.ca_indices, dtype=int)

            raise ValueError(f"Unknown atom selection: {selection}")

        if isinstance(selection, (tuple, list)) and selection and not all(
            isinstance(x, (int, np.integer)) for x in selection
        ):
            selected: set[int] | None = None

            for part in selection:
                part_indices = set(
                    self._selection_to_indices(
                        part, n_atoms).tolist())

                if selected is None:
                    selected = part_indices
                else:
                    selected &= part_indices

            if selected is None:
                raise ValueError("Empty combined atom selection")

            indices = np.asarray(sorted(selected), dtype=int)
        else:
            indices = np.asarray(selection, dtype=int)

        if indices.ndim != 1:
            raise ValueError("Atom selection indices must be one-dimensional")

        if indices.size == 0:
            raise ValueError("Atom selection is empty")

        if np.any(indices < 0) or np.any(indices >= n_atoms):
            raise IndexError(
                "Atom selection contains indices outside the coordinate array: "
                f"valid range is [0, {n_atoms - 1}]"
            )

        return indices

    def _select(self, coords: np.ndarray, selection: Selection) -> np.ndarray:
        indices = self._selection_to_indices(selection, coords.shape[0])
        return coords[indices]

    def _atom_key(atom) -> tuple[str, str, str, str, str]:
        """Return a stable atom identity key.

        The key intentionally ignores atom serial numbers because those are often
        regenerated during preprocessing.
        """

        return (
            str(atom.chain),
            str(atom.resid),
            str(getattr(atom, "altloc", "")),
            str(atom.resname),
            str(atom.name),
        )

    def selection_from_pdb_text(
        self,
        pdb_text: str,
        *,
        selection: ExternalPdbSelection = "ca",
    ) -> PairedAtomSelection:
        """Return paired atom selections between this archive and an external PDB.

        left contains indices in this HDF5 archive.
        right contains indices in the external parsed PDB atom list.

        For C-alpha selection, matching is done by per-chain residue-sequence
        alignment, so residue renumbering does not break the selection.
        """

        external_atoms = parse_pdb_atoms(pdb_text)
        archive_atoms = self.atoms

        if selection is None:
            normalized = "all"
        elif isinstance(selection, str):
            normalized = selection.lower()
        else:
            raise TypeError(
                f"Unsupported external PDB selection: {
                    selection!r}")

        if normalized not in {"ca", "calpha", "c-alpha"}:
            raise ValueError(
                "Paired selection from PDB text is currently implemented for C-alpha atoms only"
            )

        left_indices: list[int] = []
        right_indices: list[int] = []

        chains: list[str] = []

        for atom in archive_atoms:
            if atom.name == "CA" and atom.chain not in chains:
                chains.append(atom.chain)

        for chain in chains:
            archive_chain = [
                (i, atom)
                for i, atom in enumerate(archive_atoms)
                if atom.chain == chain and atom.name == "CA"
            ]

            external_chain = [
                (i, atom)
                for i, atom in enumerate(external_atoms)
                if atom.chain == chain and atom.name == "CA"
            ]

            if not archive_chain or not external_chain:
                continue

            archive_seq = [atom.resname for _, atom in archive_chain]
            external_seq = [atom.resname for _, atom in external_chain]

            matcher = SequenceMatcher(
                a=archive_seq,
                b=external_seq,
                autojunk=False,
            )

            for block in matcher.get_matching_blocks():
                for offset in range(block.size):
                    archive_index, _ = archive_chain[block.a + offset]
                    external_index, _ = external_chain[block.b + offset]

                    left_indices.append(archive_index)
                    right_indices.append(external_index)

        if not left_indices:
            raise ValueError(
                "No common C-alpha atoms found between archive reference PDB "
                "and external PDB by sequence matching"
            )

        return PairedAtomSelection(
            left=np.asarray(left_indices, dtype=int),
            right=np.asarray(right_indices, dtype=int),
        )

    def selection_from_pdb(
        self,
        pdb_path: str | Path,
        *,
        selection: ExternalPdbSelection = "ca",
    ) -> PairedAtomSelection:
        with open(pdb_path, "r") as f:
            pdb_text = f.read()

        return self.selection_from_pdb_text(
            pdb_text,
            selection=selection,
        )

    def selection_from_archive(
        self,
        archive: "MdspaceHdf5",
        *,
        selection: ExternalPdbSelection = "ca",
    ) -> PairedAtomSelection:
        return self.selection_from_pdb_text(
            archive.reference_pdb_text,
            selection=selection,
        )
