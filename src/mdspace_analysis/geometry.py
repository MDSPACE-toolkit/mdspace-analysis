from __future__ import annotations

import numpy as np


def align_coordinates(
    mobile: np.ndarray,
    reference: np.ndarray,
    *,
    return_transform: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigidly align one coordinate array onto another.

    Parameters
    ----------
    mobile
        Coordinates to align, with shape ``(n_points, 3)``.
    reference
        Reference coordinates, with shape ``(n_points, 3)``.
    return_transform
        If ``True``, also return the rotation matrix and translation vector.

    Returns
    -------
    aligned
        Aligned mobile coordinates, with shape ``(n_points, 3)``.

    rotation, translation
        Returned only if ``return_transform=True``. The transform satisfies:

        ``aligned = mobile @ rotation.T + translation``

    Notes
    -----
    The two arrays must contain matching points in the same order.
    """

    mobile = np.asarray(mobile, dtype=float)
    reference = np.asarray(reference, dtype=float)

    if mobile.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch: mobile has shape {mobile.shape}, "
            f"reference has shape {reference.shape}"
        )

    if mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(f"Expected arrays with shape (n_points, 3), got {mobile.shape}")

    mobile_centroid = mobile.mean(axis=0)
    reference_centroid = reference.mean(axis=0)

    mobile_centered = mobile - mobile_centroid
    reference_centered = reference - reference_centroid

    covariance = mobile_centered.T @ reference_centered
    u, _, vt = np.linalg.svd(covariance)

    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))

    rotation = vt.T @ correction @ u.T
    translation = reference_centroid - mobile_centroid @ rotation.T

    aligned = mobile @ rotation.T + translation

    if return_transform:
        return aligned, rotation, translation

    return aligned


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Compute RMSD between two coordinate arrays."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: a has shape {a.shape}, b has shape {b.shape}")

    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"Expected arrays with shape (n_points, 3), got {a.shape}")

    diff = a - b
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
