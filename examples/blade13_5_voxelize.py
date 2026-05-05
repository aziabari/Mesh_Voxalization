"""
Port of Blade13_5_Voxelize.m.

Voxelises a turbine-blade STL at a chosen X-resolution (so Y and Z resolutions
follow from the bounding box aspect ratio), then embeds the result in a larger
zero-padded volume and saves both:

* a ``.npz`` file with the padded ``int8`` voxel volume, and
* one TIFF per Z-slice for downstream slice-based workflows.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mesh_voxelisation import read_stl, voxelise


def main(
    filename: str,
    xslices: int = 214,
    padding: int = 25,
    out_folder: str | None = None,
) -> None:
    # -------------------------------------------------------------
    # 1. Plot the input STL.
    # -------------------------------------------------------------
    coord_vertices, _, _ = read_stl(filename)
    triangles = coord_vertices.transpose(0, 2, 1)  # (N, vertex, coord) for plotting

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(
        Poly3DCollection(triangles, facecolor="b", edgecolor="k", linewidth=0.05)
    )
    for setlim, axis in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), range(3)):
        setlim(triangles[..., axis].min(), triangles[..., axis].max())
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))

    # -------------------------------------------------------------
    # 2. Pick voxel grid sizes from bounding-box aspect ratio.
    # -------------------------------------------------------------
    mmx = triangles[..., 0].max() - triangles[..., 0].min()
    mmy = triangles[..., 1].max() - triangles[..., 1].min()
    mmz = triangles[..., 2].max() - triangles[..., 2].min()

    yslices = round(xslices * mmy / mmx)
    zslices = round(xslices * mmz / mmx)
    resol = mmx / xslices
    print(f"Resolution: {resol:.6g}   grid: {xslices} x {yslices} x {zslices}")

    # -------------------------------------------------------------
    # 3. Voxelise.
    # -------------------------------------------------------------
    output_grid = voxelise(xslices, yslices, zslices, filename, raydirection="xyz")

    # Mid-slice preview, same convention as the MATLAB script.
    fig2, ax2 = plt.subplots()
    ax2.imshow(output_grid[xslices // 2, :, :], cmap="gray")
    ax2.set_xlabel("Y-direction"); ax2.set_ylabel("X-direction")
    ax2.set_aspect("equal")
    ax2.set_title("mid-X slice (raw voxelisation)")

    # -------------------------------------------------------------
    # 4. Embed in a square, padded volume.
    # -------------------------------------------------------------
    xyl = max(xslices, yslices)
    xyl = int(np.ceil(xyl / 2)) * 2 + 2 * padding
    zl = int(np.ceil(zslices / 2)) * 2 + 2 * padding

    blade_out = np.zeros((xyl, xyl, zl), dtype=np.float64)

    # Centre the voxelised region inside the padded volume. The MATLAB
    # `xyl/2-ceil(xslices/2)+1 : xyl/2+floor(xslices/2)` formula simplifies
    # in 0-indexed Python to a nicely centred slice of length ``xslices``.
    def _centred_slice(outer: int, inner: int) -> slice:
        start = outer // 2 - (inner + 1) // 2 + 1 - 1  # MATLAB->Python: -1 then +1 net 0
        # Equivalently: start = outer//2 - ceil(inner/2)
        start = outer // 2 - -(-inner // 2)
        return slice(start, start + inner)

    blade_out[
        _centred_slice(xyl, xslices),
        _centred_slice(xyl, yslices),
        _centred_slice(zl, zslices),
    ] = output_grid.astype(np.float64)

    temp = blade_out.astype(np.int8)

    # -------------------------------------------------------------
    # 5. Save the padded volume.
    # -------------------------------------------------------------
    base, _ = os.path.splitext(filename)
    out_filename = (
        f"{base}_resol={resol:g}_dims={xyl}x{xyl}x{zl}_withPadding10slice.npz"
    )
    np.savez_compressed(out_filename, temp=temp)
    print(f"Saved padded volume -> {out_filename}")

    # Slice previews of the padded volume.
    zid = padding + zslices // 2
    xid = padding + xyl // 2
    fig3, axes3 = plt.subplots(1, 2, figsize=(8, 4))
    axes3[0].imshow(temp[:, :, zid], cmap="gray")
    axes3[0].set_xlabel("Y"); axes3[0].set_ylabel("X")
    axes3[0].set_title(f"Z slice {zid}"); axes3[0].set_aspect("equal")
    axes3[1].imshow(temp[xid, :, :], cmap="gray")
    axes3[1].set_xlabel("Z"); axes3[1].set_ylabel("Y")
    axes3[1].set_title(f"X slice {xid}"); axes3[1].set_aspect("equal")
    plt.tight_layout()

    # -------------------------------------------------------------
    # 6. Optional TIFF stack.
    # -------------------------------------------------------------
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
        print("Usage: python blade13_5_voxelize.py <stl_file> [xslices] [padding] [out_folder]")
        sys.exit(1)
    args = sys.argv[1:]
    main(
        args[0],
        xslices=int(args[1]) if len(args) > 1 else 214,
        padding=int(args[2]) if len(args) > 2 else 25,
        out_folder=args[3] if len(args) > 3 else None,
    )
