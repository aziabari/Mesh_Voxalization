"""
Direct port of VOXELISE_example.m.

Loads ``sample.stl``, plots the original triangle mesh, voxelises onto a
100x100x100 grid, and shows three orthogonal sum-projections of the result.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mesh_voxelisation import read_stl, voxelise


def main(stl_filename: str = "sample.stl") -> None:
    # Plot the original STL mesh.
    coord_vertices, _, _ = read_stl(stl_filename)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # coord_vertices: (N, 3 coords, 3 verts) -> (N, 3 verts, 3 coords)
    triangles = coord_vertices.transpose(0, 2, 1)
    ax.add_collection3d(
        Poly3DCollection(triangles, facecolor="b", edgecolor="k", linewidth=0.1)
    )
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_xlim(triangles[..., 0].min(), triangles[..., 0].max())
    ax.set_ylim(triangles[..., 1].min(), triangles[..., 1].max())
    ax.set_zlim(triangles[..., 2].min(), triangles[..., 2].max())
    ax.set_box_aspect((1, 1, 1))

    # Voxelise.
    grid = voxelise(100, 100, 100, stl_filename, raydirection="xyz")

    # Show three orthogonal sum projections (matches MATLAB subplot layout).
    fig2, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(grid.sum(axis=0), cmap="gray")
    axes[0].set_xlabel("Z-direction"); axes[0].set_ylabel("Y-direction")
    axes[0].set_aspect("equal"); axes[0].set_title("sum over X")

    axes[1].imshow(grid.sum(axis=1), cmap="gray")
    axes[1].set_xlabel("Z-direction"); axes[1].set_ylabel("X-direction")
    axes[1].set_aspect("equal"); axes[1].set_title("sum over Y")

    axes[2].imshow(grid.sum(axis=2), cmap="gray")
    axes[2].set_xlabel("Y-direction"); axes[2].set_ylabel("X-direction")
    axes[2].set_aspect("equal"); axes[2].set_title("sum over Z")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else "sample.stl"
    main(fname)
