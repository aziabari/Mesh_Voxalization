"""
mesh_voxelisation
=================

Voxelise closed (watertight) triangular-polygon meshes into 3-D boolean grids,
using axis-aligned ray casting.

Python port of Adam H. Aitkenhead's MATLAB code (The Christie NHS Foundation
Trust). The numerics follow the original closely; the Python implementation is
vectorised with NumPy where possible.

Quick start
-----------
>>> from mesh_voxelisation import voxelise
>>> grid = voxelise(100, 100, 100, "model.stl")           # 100^3 voxels
>>> grid.shape
(100, 100, 100)
>>> grid.dtype
dtype('bool')
"""

from .mesh_utils import (
    compute_mesh_normals,
    convert_meshformat,
    faces_vertices_to_meshxyz,
    meshxyz_to_faces_vertices,
)
from .stl_io import read_stl
from .voxelise import voxelise

__all__ = [
    "voxelise",
    "read_stl",
    "compute_mesh_normals",
    "convert_meshformat",
    "faces_vertices_to_meshxyz",
    "meshxyz_to_faces_vertices",
]

__version__ = "0.1.0"
