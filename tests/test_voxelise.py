"""Sanity-check the voxeliser against shapes with analytically known volumes."""

import os
import struct
import sys
import tempfile

import numpy as np

# Allow running as a plain script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mesh_voxelisation import read_stl, voxelise


def write_binary_stl(path, triangles, normals=None):
    """Write a binary STL given an (N, 3, 3) MATLAB-layout array
    (axis 1 = xyz, axis 2 = vertex index)."""
    triangles = np.asarray(triangles, dtype=np.float32)
    # Convert from MATLAB layout (N, coord, vertex) to STL on-disk layout
    # (N, vertex, coord) for sequential writing.
    triangles = triangles.transpose(0, 2, 1)
    n = triangles.shape[0]
    if normals is None:
        normals = np.zeros((n, 3), dtype=np.float32)
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", n))
        for i in range(n):
            f.write(normals[i].astype("<f4").tobytes())
            f.write(triangles[i].astype("<f4").tobytes())
            f.write(struct.pack("<H", 0))


def cube_triangles(size=10.0):
    """Twelve triangles for an axis-aligned cube spanning [0, size]^3."""
    s = size
    v = np.array(
        [
            [0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],
            [0, 0, s], [s, 0, s], [s, s, s], [0, s, s],
        ], dtype=np.float64
    )
    # Triangles given as (N, 3 vertices, 3 coords) for easy authoring,
    # then transposed to MATLAB layout.
    tris = np.array([
        [v[0], v[2], v[1]], [v[0], v[3], v[2]],  # z = 0  (bottom)
        [v[4], v[5], v[6]], [v[4], v[6], v[7]],  # z = s  (top)
        [v[0], v[1], v[5]], [v[0], v[5], v[4]],  # y = 0
        [v[2], v[3], v[7]], [v[2], v[7], v[6]],  # y = s
        [v[1], v[2], v[6]], [v[1], v[6], v[5]],  # x = s
        [v[0], v[4], v[7]], [v[0], v[7], v[3]],  # x = 0
    ])
    return tris.transpose(0, 2, 1)  # -> (N, 3 coords, 3 verts)


def sphere_triangles(radius=5.0, lat=24, lon=24):
    """Triangulated UV-sphere centred at the origin."""
    pts = []
    for i in range(lat + 1):
        theta = np.pi * i / lat
        for j in range(lon):
            phi = 2 * np.pi * j / lon
            pts.append([
                radius * np.sin(theta) * np.cos(phi),
                radius * np.sin(theta) * np.sin(phi),
                radius * np.cos(theta),
            ])
    pts = np.asarray(pts)

    def idx(i, j):
        return i * lon + (j % lon)

    tris = []
    for i in range(lat):
        for j in range(lon):
            a, b = idx(i, j), idx(i, j + 1)
            c, d = idx(i + 1, j), idx(i + 1, j + 1)
            tris.append([pts[a], pts[c], pts[b]])
            tris.append([pts[b], pts[c], pts[d]])
    return np.asarray(tris).transpose(0, 2, 1)


def test_cube():
    """Cube of side 10 voxelised on a 50^3 grid -> ~half the voxels filled."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cube.stl")
        write_binary_stl(path, cube_triangles(size=10.0))

        # Re-read and check the round trip.
        verts, _, _ = read_stl(path)
        assert verts.shape == (12, 3, 3), f"unexpected STL shape {verts.shape}"

        # 50^3 grid spanning [-5, 15] on each axis -> cube [0, 10] occupies
        # half of each axis -> ~1/8 of the volume = 12.5%.
        coords = np.linspace(-5, 15, 50)
        grid = voxelise(coords, coords, coords, path, raydirection="xyz")
        filled_frac = grid.mean()
        expected = 0.125
        print(f"  Cube fill fraction: {filled_frac:.4f}  (expected ~{expected})")
        assert abs(filled_frac - expected) < 0.02, (
            f"cube fill fraction {filled_frac} too far from {expected}"
        )


def test_sphere():
    """Sphere of radius 5 -> volume 4/3 pi r^3 ≈ 523.6, in a 20^3 box -> 6.545%."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sphere.stl")
        write_binary_stl(path, sphere_triangles(radius=5.0, lat=20, lon=20))

        coords = np.linspace(-10, 10, 60)
        grid = voxelise(coords, coords, coords, path, raydirection="xyz")
        filled_frac = grid.mean()
        # Box volume = 20^3 = 8000;  sphere volume = 4/3 pi 5^3 ≈ 523.6
        expected = (4.0 / 3.0 * np.pi * 5**3) / (20**3)
        print(f"  Sphere fill fraction: {filled_frac:.4f}  (expected ~{expected:.4f})")
        assert abs(filled_frac - expected) < 0.01, (
            f"sphere fill fraction {filled_frac} too far from {expected}"
        )


def test_ray_directions_agree():
    """All three single-axis ray directions should give similar results
    on a well-behaved closed mesh."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sphere.stl")
        write_binary_stl(path, sphere_triangles(radius=4.0, lat=16, lon=16))

        coords = np.linspace(-6, 6, 40)
        gx = voxelise(coords, coords, coords, path, raydirection="x")
        gy = voxelise(coords, coords, coords, path, raydirection="y")
        gz = voxelise(coords, coords, coords, path, raydirection="z")

        agreement_xy = (gx == gy).mean()
        agreement_xz = (gx == gz).mean()
        print(f"  Agreement x-vs-y: {agreement_xy:.4f}, x-vs-z: {agreement_xz:.4f}")
        assert agreement_xy > 0.98
        assert agreement_xz > 0.98


def test_in_memory_mesh():
    """The voxeliser should accept an (N, 3, 3) array directly, no STL on disk."""
    mesh = cube_triangles(size=4.0)
    coords = np.linspace(-2, 6, 20)
    grid = voxelise(coords, coords, coords, mesh, raydirection="xyz")
    expected = (4.0 / 8.0) ** 3  # cube occupies half of each axis
    assert abs(grid.mean() - expected) < 0.02
    print(f"  In-memory mesh fill fraction: {grid.mean():.4f}  (expected ~{expected})")


def test_return_coords():
    mesh = cube_triangles(size=2.0)
    grid, cx, cy, cz = voxelise(20, 20, 20, mesh, return_coords=True)
    assert grid.shape == (20, 20, 20)
    assert cx.size == 20 and cy.size == 20 and cz.size == 20
    print(f"  Auto grid coords range: [{cx.min():.3f}, {cx.max():.3f}]")


if __name__ == "__main__":
    print("test_cube")
    test_cube()
    print("test_sphere")
    test_sphere()
    print("test_ray_directions_agree")
    test_ray_directions_agree()
    print("test_in_memory_mesh")
    test_in_memory_mesh()
    print("test_return_coords")
    test_return_coords()
    print("\nAll tests passed!")
