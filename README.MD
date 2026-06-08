# mdspace-analysis

Small Python library for reading and analyzing MDSPACE output files.

The package provides helpers to inspect MDSPACE coordinate archives, access molecular coordinate frames, parse the reference PDB stored in the archive, select C-alpha atoms, and perform simple coordinate comparisons such as RMSD and rigid alignment.

## Features

- Read MDSPACE HDF5 archives produced by the MDSPACE workflow and data generator.
- Access raw, registered, and rotated coordinate frames.
- Read the PDB reference from the HDF5 metadata.
- Parse atom metadata from the reference PDB.
- Select C-alpha atoms directly from coordinate frames.
- Read Euler metadata and composed transform matrices.
- Convert image-space shifts from pixels to Å using the stored pixel size.
- Align coordinate arrays using the Kabsch algorithm.
- Compute RMSD between coordinate arrays.

## Installation

From the repository root:

```bash
pip install -e ".[dev]"
```

This installs the package in editable mode with the development dependencies needed to run the tests.

For end users:

```bash
pip install mdspace-analysis
```

## HDF5 archive convention

MDSPACE HDF5 archives store both molecular coordinates and image-pose metadata.

Coordinate datasets are stored in Å:

- `/frames/raw`
- `/frames/registered`
- `/frames/rotated`

Image-pose metadata follows the MDSPACE/XMIPP metadata convention:

- Euler angles are stored in degrees.
- Image shifts are stored in pixels.
- `/metadata/pixel_size` stores the pixel size in $Å/pixel$.

Therefore, when applying stored image shifts to atomic coordinates, the shifts must first be converted to Å:

`shift_angstrom = shift_pixel * pixel_size`

For example, when using` /transforms/composed`:

```python
rotation = transform_px[:3, :3]

translation_angstrom = transform_px[:3, 3] * archive.pixel_size

recovered = (rotation @ registered.T).T + translation_angstrom

```

## Basic usage

from mdspace_analysis import MdspaceHdf5

```python
with MdspaceHdf5("coords.h5") as archive:

    print("Number of frames:", archive.n_frames)

    print("Pixel size:", archive.pixel_size)

    print("Number of atoms:", archive.n_atoms)

    print("Number of C-alpha atoms:", archive.n_ca_atoms)

    raw = archive.raw(0)

    registered = archive.registered(0)

    rotated = archive.rotated(0)

    print(raw.shape)

    print(registered.shape)

    print(rotated.shape)

```

## C-alpha selection

If the HDF5 archive contains the reference PDB in /metadata/reference_pdb, the package can parse atom names and automatically select C-alpha atoms.

```python
from mdspace_analysis import MdspaceHdf5

with MdspaceHdf5("coords.h5") as archive:

    raw_ca = archive.raw(0, selection="ca")

    registered_ca = archive.registered(0, selection="ca")

    rotated_ca = archive.rotated(0, selection="ca")

    print(raw_ca.shape)

```

The atom order in /metadata/reference_pdb must match the atom order in the coordinate frames.

## Reading transforms

MDSPACE stores composed transform matrices in the HDF5 archive. For most analysis scripts, use composed_transform_angstrom(). This returns a 4×4 transform whose translation is already converted to Å, so it can be applied directly to coordinate arrays.

```python
from mdspace_analysis import MdspaceHdf5

with MdspaceHdf5("coords.h5") as archive:

    transform = archive.composed_transform_angstrom(0)

    rotation = transform[:3, :3]

    translation = transform[:3, 3]

    registered = archive.registered(0)

    recovered_rotated = (rotation @ registered.T).T + translation
```

The lower-level method composed_transform_pixels() returns the raw transform stored in the HDF5 file, expressed in pixels. Use it only when you specifically need to inspect the archive metadata convention.

## Geometry utilities

The package includes simple geometry helpers.

```python
from mdspace_analysis.geometry import align_coordinates, rmsd

aligned = align_coordinates(mobile_coords, reference_coords)

error = rmsd(aligned, reference_coords)

print("RMSD:", error)
```

align_coordinates uses the Kabsch algorithm and performs only rigid-body alignment. It applies rotation and translation, with no scaling or deformation.

## Example: PCA of registered conformations

The registered frames can be used to inspect the conformational space explored by MDSPACE. Since the structures are already aligned in a common reference frame, they are suitable for PCA-based visualization.

This example loads all registered C-alpha frames, flattens each structure into one vector, computes a two-dimensional PCA projection, and plots the resulting conformational landscape.

```python
import matplotlib.pyplot as plt

import numpy as np

from sklearn.decomposition import PCA

from mdspace_analysis import MdspaceHdf5

with MdspaceHdf5("coords.h5") as archive:

    frames = []

    for frame in range(archive.n_frames):

        coords = archive.registered(frame, selection="ca")

        frames.append(coords.reshape(-1))

    X = np.asarray(frames)

pca = PCA(n_components=2)

projection = pca.fit_transform(X)

plt.figure()

plt.scatter(projection[:, 0], projection[:, 1])

plt.xlabel("PC1")

plt.ylabel("PC2")

plt.title("MDSPACE conformational landscape")

plt.show()

```

The same idea applies to `/frames/raw` or` /frames/rotated`, but `/frames/registered` is usually the best choice for conformational analysis because global rotations and translations have already been removed.

## Notes

This package is intended as a lightweight analysis helper for MDSPACE practicals and post-processing scripts. It does not run MDSPACE itself. It only reads and analyzes existing MDSPACE HDF5 output files.
