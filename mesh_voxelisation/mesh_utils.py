"""
Mesh format utilities.

Provides:

* :func:`convert_meshformat` - convert between the (faces, vertices) format
  produced by ``isosurface``/``trimesh`` and the (N, 3, 3) facet-vertex array
  used internally by the voxeliser.
* :func:`compute_mesh_normals` - compute per-facet unit normal vectors.

Originally:
    CONVERT_meshformat.m and COMPUTE_mesh_normals.m
    by Adam H. Aitkenhead, The Christie NHS Foundation Trust.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np


def faces_vertices_to_meshxyz(
    faces: np.ndarray, vertices: np.ndarray
) -> np.ndarray:
    """Convert (faces, vertices) -> meshXYZ array of shape (N, 3, 3).

    Parameters
    ----------
    faces : ndarray, shape (N, 3)
        Indices into ``vertices`` for each facet's three corners.
    vertices : ndarray, shape (M, 3)
        Unique vertex coordinates.

    Returns
    -------
    mesh_xyz : ndarray, shape (N, 3, 3)
        Axis 0 = facet, axis 1 = (x, y, z), axis 2 = vertex index.
    """
    faces = np.asarray(faces, dtype=np.int64)
    vertices = np.asarray(vertices, dtype=np.float64)

    # vertices[faces] -> (N, 3 verts, 3 coords); transpose last two axes
    # to get (N, 3 coords, 3 verts) which matches the MATLAB layout.
    return vertices[faces].transpose(0, 2, 1)


def meshxyz_to_faces_vertices(
    mesh_xyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert an (N, 3, 3) facet-vertex array into (faces, vertices).

    Duplicate vertices are merged. The output ``vertices`` array has
    lexicographically sorted unique rows.

    Parameters
    ----------
    mesh_xyz : ndarray, shape (N, 3, 3)
        Per-facet vertex coordinates.

    Returns
    -------
    faces : ndarray, shape (N, 3)
        Vertex indices into ``vertices`` for each facet.
    vertices : ndarray, shape (M, 3)
        Unique vertex coordinates.
    """
    mesh_xyz = np.asarray(mesh_xyz, dtype=np.float64)
    n = mesh_xyz.shape[0]

    # Stack all 3*N corner coordinates as rows of an (3N, 3) array.
    all_verts = mesh_xyz.transpose(0, 2, 1).reshape(-1, 3)

    # np.unique with axis=0 deduplicates rows and gives us the inverse mapping
    # back to the original index list — exactly what we need to build `faces`.
    vertices, inverse = np.unique(all_verts, axis=0, return_inverse=True)
    faces = inverse.reshape(n, 3)
    return faces, vertices


def convert_meshformat(*args):
    """Polymorphic convert (mirrors the MATLAB CONVERT_meshformat signature).

    * ``convert_meshformat(faces, vertices)`` -> ``mesh_xyz``
    * ``convert_meshformat(mesh_xyz)`` -> ``(faces, vertices)``
    """
    if len(args) == 2:
        return faces_vertices_to_meshxyz(args[0], args[1])
    if len(args) == 1:
        return meshxyz_to_faces_vertices(args[0])
    raise TypeError("convert_meshformat takes 1 or 2 positional arguments")


def compute_mesh_normals(
    mesh_data: Union[np.ndarray, dict, Tuple[np.ndarray, np.ndarray]],
    invert: bool = False,
) -> np.ndarray:
    """Compute per-facet unit normals.

    Parameters
    ----------
    mesh_data
        Either an ``(N, 3, 3)`` facet-vertex array, a ``dict`` with
        ``'faces'`` and ``'vertices'`` keys, or a ``(faces, vertices)`` tuple.
    invert : bool, optional
        If True, swap vertices 2 and 3 before computing, which flips every
        normal. Defaults to False.

    Returns
    -------
    normals : ndarray, shape (N, 3)
        Unit normal vector for each facet.

    Notes
    -----
    The MATLAB version optionally also returned a vertex-ordering-checked
    copy of the mesh; that traversal-based check has been omitted here. It was
    slow, fragile on non-manifold meshes, and rarely the right tool for the
    job — if you need consistent winding, libraries like ``trimesh`` do this
    far more robustly than a naive edge walk.
    """
    if isinstance(mesh_data, dict):
        mesh_xyz = faces_vertices_to_meshxyz(
            mesh_data["faces"], mesh_data["vertices"]
        )
    elif isinstance(mesh_data, tuple) and len(mesh_data) == 2:
        mesh_xyz = faces_vertices_to_meshxyz(mesh_data[0], mesh_data[1])
    else:
        mesh_xyz = np.asarray(mesh_data, dtype=np.float64)

    if invert:
        mesh_xyz = mesh_xyz.copy()
        mesh_xyz[:, :, [1, 2]] = mesh_xyz[:, :, [2, 1]]

    a = mesh_xyz[:, :, 0]
    b = mesh_xyz[:, :, 1]
    c = mesh_xyz[:, :, 2]
    ab = b - a
    ac = c - a
    n = np.cross(ab, ac)

    # Normalise; guard against zero-area degenerate facets.
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return n / norms
