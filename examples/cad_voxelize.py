"""
Port of Blade13_5_Voxelize.m (resolution-driven version).

Voxelises an STL at a chosen voxel *resolution* (so the slice counts on each
axis follow from the bounding box), embeds the result in a larger zero-padded
volume, and saves it as a single ``.npy`` file. The output filename is
``<input_basename_with_spaces_replaced>_voxelized_{resol}mm.npy``, written
either next to the input STL or into ``out_folder`` if one is supplied.

Equivalent MATLAB:
    resol   = 0.5;
    xslices = round(mmx/resol);
    yslices = round(xslices * mmy/mmx);
    zslices = round(xslices * mmz/mmx);
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mesh_voxelisation import read_stl, voxelise


def main(
    filename: str,
    resol: float = 0.5,
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
    # 2. Pick voxel grid sizes from the requested resolution.
    #    Slice counts on Y and Z follow from the bounding-box aspect ratio
    #    so that voxels stay (approximately) cubic.
    # -------------------------------------------------------------
    mmx = triangles[..., 0].max() - triangles[..., 0].min()
    mmy = triangles[..., 1].max() - triangles[..., 1].min()
    mmz = triangles[..., 2].max() - triangles[..., 2].min()

    xslices = round(mmx / resol)
    yslices = round(xslices * mmy / mmx)
    zslices = round(xslices * mmz / mmx)
    print(f"Resolution: {resol:g}   grid: {xslices} x {yslices} x {zslices}")

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

    # Centre the voxelised region inside the padded volume.
    def _centred_slice(outer: int, inner: int) -> slice:
        start = outer // 2 - -(-inner // 2)  # ceil-div
        return slice(start, start + inner)

    blade_out[
        _centred_slice(xyl, xslices),
        _centred_slice(xyl, yslices),
        _centred_slice(zl, zslices),
    ] = output_grid.astype(np.float64)

    temp = blade_out.astype(np.int8)

    # -------------------------------------------------------------
    # 5. Save the padded volume as a single .npy file.
    #    Output name: <input_basename_with_spaces_replaced>_voxelized_{resol}mm.npy
    #    Written to out_folder if given, else next to the input STL.
    # -------------------------------------------------------------
    if out_folder is not None:
        os.makedirs(out_folder, exist_ok=True)
        out_dir = out_folder
    else:
        out_dir = os.path.dirname(os.path.abspath(filename)) or "."

    in_base, _ = os.path.splitext(os.path.basename(filename))
    safe_base = in_base.replace(" ", "_")
    out_filename = os.path.join(out_dir, f"{safe_base}_voxelized_{resol:g}mm.npy")
    np.save(out_filename, temp)
    print(f"Saved padded volume -> {out_filename}")

    # Slice previews of the padded volume.
    zid = zl // 2
    xid = xyl // 2
    fig3, axes3 = plt.subplots(1, 2, figsize=(8, 4))
    axes3[0].imshow(temp[:, :, zid], cmap="gray")
    axes3[0].set_xlabel("Y"); axes3[0].set_ylabel("X")
    axes3[0].set_title(f"Z slice {zid}"); axes3[0].set_aspect("equal")
    axes3[1].imshow(temp[xid, :, :], cmap="gray")
    axes3[1].set_xlabel("Z"); axes3[1].set_ylabel("Y")
    axes3[1].set_title(f"X slice {xid}"); axes3[1].set_aspect("equal")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Defaults for IDE use: edit these four lines and hit Run.
    # For command-line use: pass them as positional args.
    # ----------------------------------------------------------------
    DEFAULT_FILENAME = "P3_Polymer_Solid_AH_can.STL"
    DEFAULT_RESOL    = 0.5
    DEFAULT_PADDING  = 25
    DEFAULT_OUT_DIR  = None  # None => save next to input; else a folder path

    if len(sys.argv) >= 2:
        args = sys.argv[1:]
        filename   = args[0]
        resol      = float(args[1]) if len(args) > 1 else DEFAULT_RESOL
        padding    = int(args[2])   if len(args) > 2 else DEFAULT_PADDING
        out_folder = args[3]        if len(args) > 3 else DEFAULT_OUT_DIR
    else:
        filename, resol, padding, out_folder = (
            DEFAULT_FILENAME, DEFAULT_RESOL, DEFAULT_PADDING, DEFAULT_OUT_DIR,
        )

    main(filename, resol=resol, padding=padding, out_folder=out_folder)