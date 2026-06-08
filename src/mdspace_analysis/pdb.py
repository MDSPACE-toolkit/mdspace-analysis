from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PdbAtom:
    """Minimal atom metadata parsed from an ATOM/HETATM PDB record."""

    index: int
    record: str
    name: str
    resname: str
    chain: str
    resid: int
    altloc: str
    element: str


def parse_pdb_atoms(pdb_text: str) -> list[PdbAtom]:
    """Parse atom metadata from PDB text.

    Only ATOM and HETATM records are parsed.

    The returned atom order is the same as the order in the PDB file. This order
    must match the order of coordinates stored in the HDF5 frames if atom-based
    selections are applied.
    """

    atoms: list[PdbAtom] = []

    for line in pdb_text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue

        if len(line) < 54:
            raise ValueError(f"Malformed PDB atom line: {line!r}")

        record = line[0:6].strip()
        name = line[12:16].strip()
        altloc = line[16:17].strip()
        resname = line[17:20].strip()
        chain = line[21:22].strip()

        try:
            resid = int(line[22:26])
        except ValueError as exc:
            raise ValueError(
                f"Invalid residue number in PDB line: {
                    line!r}") from exc

        element = line[76:78].strip() if len(line) >= 78 else ""

        atoms.append(
            PdbAtom(
                index=len(atoms),
                record=record,
                name=name,
                resname=resname,
                chain=chain,
                resid=resid,
                altloc=altloc,
                element=element,
            )
        )

    if not atoms:
        raise ValueError("No ATOM/HETATM records found in reference PDB")

    return atoms
