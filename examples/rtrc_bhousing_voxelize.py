"""
Port of RTRC_BHOUSING_VOXELIZE.m.

Same workflow as ``blade13_5_voxelize.py`` but driven by a target voxel
*resolution* rather than a target X-count.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mesh_voxelisation import read_stl, voxelise


def main(
    filename: str,
    resol: float = 4 * 7.873892570952e-02,
    padding: int = 25,
    out_folder: str | None = None,
) -> None:
    coord_vertices, _, _ = read_stl(filename)
    triangles = coord_vertices.transpose(0, 2, 1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(
        Poly3DCollection(triangles, facecolor="b", edgecolor="k", linewidth=0.05)
    )
    for setlim, axis in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), range(3)):
        setlim(triangles[..., axis].min(), triangles[..., axis].max())
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))

    # Target resolution -> grid sizes.
    mmx = triangles[..., 0].max() - triangles[..., 0].min()
    mmy = triangles[..., 1].max() - triangles[..., 1].min()
    mmz = triangles[..., 2].max() - triangles[..., 2].min()
    xslices = round(mmx / resol)
    yslices = round(xslices * mmy / mmx)
    zslices = round(xslices * mmz / mmx)
    print(f"Grid: {xslices} x {yslices} x {zslices} at resol={resol:g}")

    output_grid = voxelise(xslices, yslices, zslices, filename, raydirection="xyz")

    fig2, ax2 = plt.subplots()
    ax2.imshow(output_grid[xslices // 2, :, :], cmap="gray")
    ax2.set_xlabel("Y"); ax2.set_ylabel("X")
    ax2.set_aspect("equal")
    ax2.set_title("mid-X slice")

    xyl = max(xslices, yslices)
    xyl = int(np.ceil(xyl / 2)) * 2 + 2 * padding
    zl = int(np.ceil(zslices / 2)) * 2 + 2 * padding

    blade_out = np.zeros((xyl, xyl, zl), dtype=np.float64)

    def _centred_slice(outer: int, inner: int) -> slice:
        start = outer // 2 - -(-inner // 2)  # ceil-div
        return slice(start, start + inner)

    blade_out[
        _centred_slice(xyl, xslices),
        _centred_slice(xyl, yslices),
        _centred_slice(zl, zslices),
    ] = output_grid.astype(np.float64)

    temp = blade_out.astype(np.int8)

    base, _ = os.path.splitext(filename)
    out_filename = (
        f"{base}_resol={resol:g}_dims={xyl}x{xyl}x{zl}_withPadding10slice.npz"
    )
    np.savez_compressed(out_filename, temp=temp)
    print(f"Saved -> {out_filename}")

    zid = padding + zslices // 2
    xid = padding + xyl // 2
    fig3, axes3 = plt.subplots(1, 2, figsize=(8, 4))
    axes3[0].imshow(temp[:, :, zid], cmap="gray")
    axes3[0].set_title(f"Z slice {zid}"); axes3[0].set_aspect("equal")
    axes3[1].imshow(temp[xid, :, :], cmap="gray")
    axes3[1].set_title(f"X slice {xid}"); axes3[1].set_aspect("equal")
    plt.tight_layout()

    if out_folder is not None:
        try:
            from PIL import Image
        except ImportError:
            print("Pillow not installed — skipping TIFF export.")
            return
        os.makedirs(out_folder, exist_ok=True)
        for i in range(zl):
            img = Image.fromarray((temp[:, :, i] > 0).astype(np.uint8) * 255, "L")
            img.save(os.path.join(out_folder, f"_{i}.tiff"), compression=None)
        print(f"Wrote {zl} TIFF slices to {out_folder}")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python rtrc_bhousing_voxelize.py <stl_file> [resol] [padding] [out_folder]"
        )
        sys.exit(1)
    args = sys.argv[1:]
    main(
        args[0],
        resol=float(args[1]) if len(args) > 1 else 4 * 7.873892570952e-02,
        padding=int(args[2]) if len(args) > 2 else 25,
        out_folder=args[3] if len(args) > 3 else None,
    )
