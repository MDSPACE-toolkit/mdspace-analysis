from pathlib import Path

import numpy as np
import h5py

from mdspace_analysis import MdspaceHdf5
from mdspace_analysis.geometry import align_coordinates, rmsd


REAL_ARCHIVE = Path(__file__).parent / "data" / "coords.h5"
REAL_GENERATOR = Path(__file__).parent / "data" / "generated_data.h5"


def assert_coordinates_close(a, b, *, rmsd_tol, max_tol, label):
    value_rmsd = rmsd(a, b)
    assert value_rmsd < rmsd_tol, f"{label}: RMSD = {value_rmsd}"

    max_error = np.abs(a - b).max()
    assert max_error < max_tol, f"{label}: max error = {max_error}"


def test_real_mdspace_archive_can_be_opened():
    assert REAL_ARCHIVE.exists(), f"Missing test archive: {REAL_ARCHIVE}"

    with MdspaceHdf5(REAL_ARCHIVE) as archive:
        assert archive.n_frames > 0
        assert archive.pixel_size > 0
        assert archive.n_atoms > 0

        raw = archive.raw(0)
        registered = archive.registered(0)
        rotated = archive.rotated(0)

        assert raw.shape == registered.shape == rotated.shape
        assert raw.shape[1] == 3


def test_real_mdspace_archive_composed_transform_recovers_rotated():
    with MdspaceHdf5(REAL_ARCHIVE) as archive:
        pixel_size = archive.pixel_size

        for frame in range(archive.n_frames):
            registered = archive.registered(frame)
            rotated = archive.rotated(frame)

            transform_px = archive.composed_transform_pixels(frame)

            rotation = transform_px[:3, :3]
            translation_angstrom = transform_px[:3, 3] * pixel_size

            recovered = (rotation @ registered.T).T + translation_angstrom

            assert_coordinates_close(
                recovered,
                rotated,
                rmsd_tol=1e-6,
                max_tol=1e-6,
                label=f"frame {frame}",
            )


def test_real_mdspace_archive_euler_shifts_match_composed_translation():
    with MdspaceHdf5(REAL_ARCHIVE) as archive:
        for frame in range(archive.n_frames):
            euler = archive.euler(frame)
            transform_px = archive.composed_transform_pixels(frame)

            expected = np.array([euler.shift_x, euler.shift_y, euler.shift_z])
            actual = transform_px[:3, 3]

            max_error = np.abs(actual - expected).max()
            assert max_error < 1e-12, f"frame {frame}: max shift error = {max_error}"


def test_real_mdspace_archive_calpha_selection():
    with MdspaceHdf5(REAL_ARCHIVE) as archive:
        ca_raw = archive.raw(0, selection="ca")
        ca_rotated = archive.rotated(0, selection="ca")

        assert ca_raw.shape == ca_rotated.shape
        assert ca_raw.shape[1] == 3
        assert ca_raw.shape[0] == archive.n_ca_atoms


def test_real_generated_archive_raw_to_rotated_from_composed_matrix():
    with MdspaceHdf5(REAL_GENERATOR) as archive:
        pixel_size = archive.pixel_size

        for frame in range(archive.n_frames):
            raw = archive.raw(frame)
            rotated = archive.rotated(frame)

            transform_px = archive.composed_transform_pixels(frame)

            rotation = transform_px[:3, :3]
            translation_angstrom = transform_px[:3, 3] * pixel_size

            recovered = (rotation @ raw.T).T + translation_angstrom

            assert_coordinates_close(
                recovered,
                rotated,
                rmsd_tol=1e-6,
                max_tol=1e-6,
                label=f"frame {frame}",
            )


def test_real_generated_archive_without_deformation_aligns_rotated_to_raw():
    with MdspaceHdf5(REAL_GENERATOR) as archive:
        for frame in range(archive.n_frames):
            raw = archive.raw(frame)
            rotated = archive.rotated(frame)

            aligned = align_coordinates(rotated, raw)

            assert_coordinates_close(
                aligned,
                raw,
                rmsd_tol=1e-6,
                max_tol=1e-6,
                label=f"frame {frame}",
            )


def test_real_generated_archive_without_deformation_raw_frames_are_identical():
    with MdspaceHdf5(REAL_GENERATOR) as archive:
        reference_raw = archive.raw(0)

        for frame in range(1, archive.n_frames):
            raw = archive.raw(frame)

            assert_coordinates_close(
                raw,
                reference_raw,
                rmsd_tol=1e-12,
                max_tol=1e-12,
                label=f"frame {frame}",
            )


def test_real_generated_archive_euler_and_matrix_units_are_consistent():
    with MdspaceHdf5(REAL_GENERATOR) as archive:
        pixel_size = archive.pixel_size

        for frame in range(archive.n_frames):
            euler = archive.euler(frame)
            transform_px = archive.composed_transform_pixels(frame)

            euler_shift_px = np.array(
                [euler.shift_x, euler.shift_y, euler.shift_z],
                dtype=float,
            )

            matrix_translation_px = transform_px[:3, 3]

            np.testing.assert_allclose(
                matrix_translation_px,
                euler_shift_px,
                atol=1e-12,
            )

            np.testing.assert_allclose(
                matrix_translation_px * pixel_size,
                euler_shift_px * pixel_size,
                atol=1e-12,
            )


def test_real_mdspace_archive_euler_and_matrix_units_are_consistent():
    with MdspaceHdf5(REAL_ARCHIVE) as archive:
        pixel_size = archive.pixel_size

        for frame in range(archive.n_frames):
            euler = archive.euler(frame)
            transform_px = archive.composed_transform_pixels(frame)

            euler_shift_px = np.array(
                [euler.shift_x, euler.shift_y, euler.shift_z],
                dtype=float,
            )

            matrix_translation_px = transform_px[:3, 3]

            np.testing.assert_allclose(
                matrix_translation_px,
                euler_shift_px,
                atol=1e-12,
            )

            euler_shift_angstrom = euler_shift_px * pixel_size
            matrix_translation_angstrom = matrix_translation_px * pixel_size

            np.testing.assert_allclose(
                matrix_translation_angstrom,
                euler_shift_angstrom,
                atol=1e-12,
            )


def test_real_mdspace_archive_composed_transform_angstrom_recovers_rotated():
    with MdspaceHdf5(REAL_ARCHIVE) as archive:
        for frame in range(archive.n_frames):
            registered = archive.registered(frame)
            rotated = archive.rotated(frame)

            transform = archive.composed_transform_angstrom(frame)

            rotation = transform[:3, :3]
            translation = transform[:3, 3]

            recovered = (rotation @ registered.T).T + translation

            assert_coordinates_close(
                recovered,
                rotated,
                rmsd_tol=1e-6,
                max_tol=1e-6,
                label=f"frame {frame}",
            )


def test_real_mdspace_archive_registered_frames_are_close_in_rmsd():
    with MdspaceHdf5(REAL_ARCHIVE) as archive:
        reference = archive.registered(0, selection="ca")

        for frame in range(1, archive.n_frames):
            registered = archive.registered(frame, selection="ca")

            error = rmsd(registered, reference)
            assert error < 2.0, f"frame {frame}: registered C-alpha RMSD = {error}"


def test_align_coordinates_recovers_known_isometry_on_real_hdf5_structure():
    with MdspaceHdf5(REAL_ARCHIVE) as archive:
        reference = archive.registered(0, selection="ca")

    angle = np.deg2rad(37.0)

    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    translation = np.array([120.0, -80.0, 35.0], dtype=float)

    mobile = (rotation @ reference.T).T + translation

    aligned = align_coordinates(mobile, reference)

    assert_coordinates_close(
        aligned,
        reference,
        rmsd_tol=1e-10,
        max_tol=1e-10,
        label="known isometry alignment",
    )


def test_selection_from_pdb_ca(tmp_path: Path) -> None:
    archive_pdb_text = "\n".join(
        [
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N",
            "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C",
            "ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C",
            "ATOM      4  N   GLY A   2       3.000   0.000   0.000  1.00  0.00           N",
            "ATOM      5  CA  GLY A   2       4.000   0.000   0.000  1.00  0.00           C",
            "ATOM      6  C   GLY A   2       5.000   0.000   0.000  1.00  0.00           C",
            "ATOM      7  N   SER A   3       6.000   0.000   0.000  1.00  0.00           N",
            "ATOM      8  CA  SER A   3       7.000   0.000   0.000  1.00  0.00           C",
            "ATOM      9  C   SER A   3       8.000   0.000   0.000  1.00  0.00           C",
            "END",
            "",
        ]
    )

    external_pdb_text = "\n".join(
        [
            "ATOM    101  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N",
            "ATOM    102  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C",
            "ATOM    103  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C",
            "ATOM    107  N   SER A   3       6.000   0.000   0.000  1.00  0.00           N",
            "ATOM    108  CA  SER A   3       7.000   0.000   0.000  1.00  0.00           C",
            "ATOM    109  C   SER A   3       8.000   0.000   0.000  1.00  0.00           C",
            "END",
            "",
        ]
    )

    archive_path = tmp_path / "coords.h5"
    external_pdb = tmp_path / "external.pdb"

    external_pdb.write_text(external_pdb_text)

    coords = np.zeros((1, 9, 3), dtype=np.float64)

    with h5py.File(archive_path, "w") as h5:
        metadata = h5.create_group("metadata")
        metadata.create_dataset(
            "reference_pdb",
            data=archive_pdb_text.encode("utf-8"),
        )
        metadata.create_dataset("pixel_size", data=1.0)

        frames = h5.create_group("frames")
        frames.create_dataset("raw", data=coords)
        frames.create_dataset("registered", data=coords)
        frames.create_dataset("rotated", data=coords)

    with MdspaceHdf5(archive_path) as archive:
        selection = archive.selection_from_pdb(external_pdb, selection="ca")

    # Archive C-alpha atoms are absolute indices:
    # ALA A1 CA -> 1
    # GLY A2 CA -> 4
    # SER A3 CA -> 7
    #
    # External parsed atoms are also absolute indices in the external PDB atom list:
    # ALA A1 CA -> 1
    # SER A3 CA -> 4
    #
    # External PDB contains ALA A1 and SER A3, but not GLY A2.
    np.testing.assert_array_equal(
        selection.left, np.asarray([1, 7], dtype=int))
    np.testing.assert_array_equal(
        selection.right, np.asarray([1, 4], dtype=int))
    assert selection.size == 2


def test_selection_from_pdb_indices_can_select_coordinates(
        tmp_path: Path) -> None:
    archive_pdb_text = "\n".join(
        [
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N",
            "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C",
            "ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C",
            "ATOM      4  N   GLY A   2       3.000   0.000   0.000  1.00  0.00           N",
            "ATOM      5  CA  GLY A   2       4.000   0.000   0.000  1.00  0.00           C",
            "ATOM      6  C   GLY A   2       5.000   0.000   0.000  1.00  0.00           C",
            "ATOM      7  N   SER A   3       6.000   0.000   0.000  1.00  0.00           N",
            "ATOM      8  CA  SER A   3       7.000   0.000   0.000  1.00  0.00           C",
            "ATOM      9  C   SER A   3       8.000   0.000   0.000  1.00  0.00           C",
            "END",
            "",
        ]
    )

    external_pdb_text = "\n".join(
        [
            "ATOM    102  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C",
            "ATOM    108  CA  SER A   3       7.000   0.000   0.000  1.00  0.00           C",
            "END",
            "",
        ]
    )

    archive_path = tmp_path / "coords.h5"
    external_pdb = tmp_path / "external.pdb"

    external_pdb.write_text(external_pdb_text)

    coords = np.asarray(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [7.0, 0.0, 0.0],
                [8.0, 0.0, 0.0],
            ]
        ],
        dtype=np.float64,
    )

    with h5py.File(archive_path, "w") as h5:
        metadata = h5.create_group("metadata")
        metadata.create_dataset(
            "reference_pdb",
            data=archive_pdb_text.encode("utf-8"),
        )
        metadata.create_dataset("pixel_size", data=1.0)

        frames = h5.create_group("frames")
        frames.create_dataset("raw", data=coords)
        frames.create_dataset("registered", data=coords)
        frames.create_dataset("rotated", data=coords)

    with MdspaceHdf5(archive_path) as archive:
        selection = archive.selection_from_pdb(external_pdb, selection="ca")
        selected = archive.raw(0, selection=selection.left)

    expected = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(
        selection.left, np.asarray([1, 7], dtype=int))
    np.testing.assert_array_equal(
        selection.right, np.asarray([0, 1], dtype=int))
    np.testing.assert_array_equal(selected, expected)


def test_reference_coordinates_support_calpha_selection(
        tmp_path: Path) -> None:
    reference_pdb_text = "\n".join(
        [
            "ATOM      1  N   ALA A   1      10.000  20.000  30.000  1.00  0.00           N",
            "ATOM      2  CA  ALA A   1      11.000  21.000  31.000  1.00  0.00           C",
            "ATOM      3  C   ALA A   1      12.000  22.000  32.000  1.00  0.00           C",
            "ATOM      4  N   GLY A   2      13.000  23.000  33.000  1.00  0.00           N",
            "ATOM      5  CA  GLY A   2      14.000  24.000  34.000  1.00  0.00           C",
            "ATOM      6  C   GLY A   2      15.000  25.000  35.000  1.00  0.00           C",
            "END",
            "",
        ]
    )

    archive_path = tmp_path / "coords.h5"
    coords = np.zeros((1, 6, 3), dtype=np.float64)

    with h5py.File(archive_path, "w") as h5:
        metadata = h5.create_group("metadata")
        metadata.create_dataset(
            "reference_pdb",
            data=reference_pdb_text.encode("utf-8"),
        )
        metadata.create_dataset("pixel_size", data=1.0)

        frames = h5.create_group("frames")
        frames.create_dataset("raw", data=coords)
        frames.create_dataset("registered", data=coords)
        frames.create_dataset("rotated", data=coords)

    with MdspaceHdf5(archive_path) as archive:
        ca_reference = archive.reference_coordinates(selection="ca")

    expected = np.asarray(
        [
            [11.0, 21.0, 31.0],
            [14.0, 24.0, 34.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(ca_reference, expected)


def test_reference_coordinates_are_read_from_reference_pdb(
        tmp_path: Path) -> None:
    reference_pdb_text = "\n".join(
        [
            "ATOM      1  N   ALA A   1      10.000  20.000  30.000  1.00  0.00           N",
            "ATOM      2  CA  ALA A   1      11.000  21.000  31.000  1.00  0.00           C",
            "ATOM      3  C   ALA A   1      12.000  22.000  32.000  1.00  0.00           C",
            "ATOM      4  N   GLY A   2      13.000  23.000  33.000  1.00  0.00           N",
            "ATOM      5  CA  GLY A   2      14.000  24.000  34.000  1.00  0.00           C",
            "ATOM      6  C   GLY A   2      15.000  25.000  35.000  1.00  0.00           C",
            "END",
            "",
        ]
    )

    archive_path = tmp_path / "coords.h5"

    # Deliberately different from the coordinates in reference_pdb_text.
    # This ensures reference_coordinates() is not silently reading /frames/raw.
    raw_coords = np.asarray(
        [
            [
                [100.0, 200.0, 300.0],
                [101.0, 201.0, 301.0],
                [102.0, 202.0, 302.0],
                [103.0, 203.0, 303.0],
                [104.0, 204.0, 304.0],
                [105.0, 205.0, 305.0],
            ]
        ],
        dtype=np.float64,
    )

    with h5py.File(archive_path, "w") as h5:
        metadata = h5.create_group("metadata")
        metadata.create_dataset(
            "reference_pdb",
            data=reference_pdb_text.encode("utf-8"),
        )
        metadata.create_dataset("pixel_size", data=1.0)

        frames = h5.create_group("frames")
        frames.create_dataset("raw", data=raw_coords)
        frames.create_dataset("registered", data=raw_coords)
        frames.create_dataset("rotated", data=raw_coords)

    with MdspaceHdf5(archive_path) as archive:
        reference = archive.reference_coordinates()

    expected = np.asarray(
        [
            [10.0, 20.0, 30.0],
            [11.0, 21.0, 31.0],
            [12.0, 22.0, 32.0],
            [13.0, 23.0, 33.0],
            [14.0, 24.0, 34.0],
            [15.0, 25.0, 35.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(reference, expected)
