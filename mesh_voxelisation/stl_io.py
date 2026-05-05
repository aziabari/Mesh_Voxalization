"""
STL file I/O.

Reads triangular-polygon meshes from ASCII or binary STL files, returning the
vertex coordinates as an (N, 3, 3) array (N facets, 3 spatial coords, 3 vertices),
matching the layout used by the MATLAB code this package was ported from.

Originally:
    READ_stl.m by Adam H. Aitkenhead, The Christie NHS Foundation Trust.
"""

from __future__ import annotations

import os
import struct
from typing import Tuple

import numpy as np


def _identify_format(filename: str) -> str:
    """Detect whether an STL file is ASCII or binary.

    Binary STLs MUST have a size of 84 + 50*N bytes. If the size doesn't match
    that pattern, the file is ASCII. Otherwise we look at the first/last 80
    bytes for the literal 'solid'/'endsolid' markers, since binary headers
    shouldn't begin with the word 'solid' (and many ASCII writers set both).
    """
    file_size = os.path.getsize(filename)

    # Files that don't fit the binary record pattern must be ASCII.
    if (file_size - 84) % 50 != 0:
        return "ascii"

    with open(filename, "rb") as f:
        first_eighty = f.read(80).decode("ascii", errors="ignore").strip()
        first_five = first_eighty[:5].lower()

        if first_five == "solid":
            # Probably ASCII, but double-check: ASCII files end with 'endsolid'.
            f.seek(-80, os.SEEK_END)
            last_eighty = f.read(80).decode("ascii", errors="ignore")
            if "endsolid" in last_eighty:
                return "ascii"
            return "binary"

    return "binary"


def _read_ascii(filename: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """Read an ASCII STL file."""
    with open(filename, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # Object name from the first line: "solid <name>"
    first = lines[0]
    name = first[6:] if first.lower().startswith("solid ") else "unnamed_object"

    normals = []
    vertices = []
    for ln in lines:
        low = ln.lower()
        if low.startswith("facet normal"):
            normals.append([float(x) for x in ln.split()[2:5]])
        elif low.startswith("vertex"):
            vertices.append([float(x) for x in ln.split()[1:4]])

    coord_normals = np.asarray(normals, dtype=np.float64)
    verts = np.asarray(vertices, dtype=np.float64)
    n_facets = verts.shape[0] // 3
    # MATLAB convention: (N_facets, 3 coords, 3 vertices)
    coord_vertices = verts.reshape(n_facets, 3, 3).transpose(0, 2, 1)
    return coord_vertices, coord_normals, name


def _read_binary(filename: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """Read a binary STL file using a single bulk numpy.fromfile call."""
    with open(filename, "rb") as f:
        f.seek(80)  # skip 80-byte header
        n_facets = struct.unpack("<I", f.read(4))[0]
        # Each facet: 12 floats (normal + 3 vertices) + 2 bytes attribute.
        facet_dtype = np.dtype(
            [("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")]
        )
        data = np.fromfile(f, dtype=facet_dtype, count=n_facets)

    coord_normals = data["normal"].astype(np.float64)
    # data["v"] is (N, 3 vertices, 3 coords); rearrange to (N, 3 coords, 3 vertices).
    coord_vertices = data["v"].astype(np.float64).transpose(0, 2, 1)
    return coord_vertices, coord_normals, "unnamed_object"


def read_stl(
    filename: str, stl_format: str = "auto"
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Read mesh data from an STL file.

    Parameters
    ----------
    filename : str
        Path to the STL file.
    stl_format : {'auto', 'ascii', 'binary'}, optional
        Force a particular format. ``'auto'`` (the default) detects it.

    Returns
    -------
    coord_vertices : ndarray, shape (N, 3, 3)
        Vertex coordinates for each facet. Axis 0 = facet, axis 1 = (x, y, z),
        axis 2 = vertex index (0, 1, 2). Matches the MATLAB layout exactly.
    coord_normals : ndarray, shape (N, 3)
        Facet normal vectors (as stored in the file; not necessarily unit).
    stl_name : str
        Object name from the ASCII header, or ``'unnamed_object'`` for binary.
    """
    fmt = stl_format.lower()
    if fmt == "auto":
        fmt = _identify_format(filename)

    if fmt == "ascii":
        return _read_ascii(filename)
    if fmt == "binary":
        return _read_binary(filename)
    raise ValueError(f"Unknown STL format: {stl_format!r}")
