from pathlib import Path

import numpy as np

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
