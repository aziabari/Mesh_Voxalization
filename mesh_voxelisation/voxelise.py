"""
Mesh voxelisation by ray casting.

A vectorised Python port of ``VOXELISE.m`` by Adam H. Aitkenhead
(The Christie NHS Foundation Trust). The mesh is voxelised by casting axis-aligned
rays through the centre of each grid pixel and counting how many times each ray
crosses the surface — an even number of crossings means the entry/exit pairs
mark the spans of voxels lying inside the closed mesh.

The MATLAB original processes one ray at a time. Here, for each Z-slab, we find
all candidate facets, compute crossings per pixel using vectorised numpy
operations, then assemble the full grid. This is typically 1-2 orders of
magnitude faster for moderately sized meshes/grids.

Reference
---------
Patil S and Ravi B. Voxel-based representation, display and thickness analysis
of intricate shapes. Ninth International Conference on Computer Aided Design
and Computer Graphics (CAD/CG 2005)
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from .mesh_utils import faces_vertices_to_meshxyz
from .stl_io import read_stl

MeshInput = Union[str, np.ndarray, dict, Tuple[np.ndarray, np.ndarray]]
GridSpec = Union[int, np.ndarray]

_VALID_RAYDIRS = {
    "x", "y", "z",
    "xy", "xz", "yx", "yz", "zx", "zy",
    "xyz", "xzy", "yxz", "yzx", "zxy", "zyx",
}


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def _load_mesh(mesh: MeshInput) -> np.ndarray:
    """Coerce any supported input form into an (N, 3, 3) facet-vertex array."""
    if isinstance(mesh, str):
        coord_vertices, _, _ = read_stl(mesh)
        return coord_vertices

    if isinstance(mesh, dict):
        return faces_vertices_to_meshxyz(mesh["faces"], mesh["vertices"])

    if isinstance(mesh, tuple) and len(mesh) == 2:
        return faces_vertices_to_meshxyz(mesh[0], mesh[1])

    arr = np.asarray(mesh, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"Expected (N, 3, 3) mesh array, got shape {arr.shape}"
        )
    return arr


def _resolve_axis_grid(
    spec: GridSpec, mesh_min: float, mesh_max: float
) -> np.ndarray:
    """Convert a grid spec into an explicit 1-D coordinate vector.

    A scalar integer means "auto-generate this many evenly spaced voxel
    centres covering [mesh_min, mesh_max]". The unusual ``+1/2`` factor
    matches the MATLAB original — it ensures the first/last voxel centres
    sit half a voxel inside the mesh bounding box.
    """
    arr = np.atleast_1d(np.asarray(spec))

    if arr.size > 1:
        return arr.astype(np.float64).ravel()

    val = arr.item()
    if val == 1:
        return np.array([(mesh_min + mesh_max) / 2.0], dtype=np.float64)

    if isinstance(val, (int, np.integer)) or float(val).is_integer():
        n = int(val)
        vox_width = (mesh_max - mesh_min) / (n + 0.5)
        return np.arange(
            mesh_min + vox_width / 2.0,
            mesh_max - vox_width / 2.0 + vox_width / 2.0,  # +eps for inclusive end
            vox_width,
            dtype=np.float64,
        )[:n]

    raise ValueError(f"Cannot interpret grid spec: {spec!r}")


# ---------------------------------------------------------------------------
# Core ray-casting
# ---------------------------------------------------------------------------

def _voxelise_internal(
    grid_co_x: np.ndarray,
    grid_co_y: np.ndarray,
    grid_co_z: np.ndarray,
    mesh_xyz: np.ndarray,
) -> np.ndarray:
    """Cast Z-direction rays through every (x, y) pixel.

    Returns a (Nx, Ny, Nz) boolean array marking voxels inside the mesh.
    The caller handles axis permutation for X- and Y-direction passes.
    """
    voxcount_x = grid_co_x.size
    voxcount_y = grid_co_y.size
    voxcount_z = grid_co_z.size

    grid_output = np.zeros((voxcount_x, voxcount_y, voxcount_z), dtype=bool)

    # Per-facet x/y bounding boxes — used to cheaply reject facets that can't
    # possibly be hit by a ray at a given (x, y).
    facet_xmin = mesh_xyz[:, 0, :].min(axis=1)
    facet_xmax = mesh_xyz[:, 0, :].max(axis=1)
    facet_ymin = mesh_xyz[:, 1, :].min(axis=1)
    facet_ymax = mesh_xyz[:, 1, :].max(axis=1)

    mesh_z_min = mesh_xyz[:, 2, :].min()
    mesh_z_max = mesh_xyz[:, 2, :].max()

    # Restrict the (x, y) loop to the mesh's bounding box in pixel coords.
    def _bbox_pixel_range(coords: np.ndarray, lo: float, hi: float):
        i_lo = int(np.argmin(np.abs(coords - lo)))
        i_hi = int(np.argmin(np.abs(coords - hi)))
        if i_lo > i_hi:
            i_lo, i_hi = i_hi, i_lo
        return i_lo, i_hi

    yi_lo, yi_hi = _bbox_pixel_range(
        grid_co_y, mesh_xyz[:, 1, :].min(), mesh_xyz[:, 1, :].max()
    )
    xi_lo, xi_hi = _bbox_pixel_range(
        grid_co_x, mesh_xyz[:, 0, :].min(), mesh_xyz[:, 0, :].max()
    )

    # Collect (x_idx, y_idx) pairs that need correction (rays hitting an edge
    # exactly, etc.) so we can interpolate them from neighbours afterwards.
    correction_list: list = []

    # Per-facet vertex coordinates, broken out for readability.
    x1 = mesh_xyz[:, 0, 0]; y1 = mesh_xyz[:, 1, 0]
    x2 = mesh_xyz[:, 0, 1]; y2 = mesh_xyz[:, 1, 1]
    x3 = mesh_xyz[:, 0, 2]; y3 = mesh_xyz[:, 1, 2]

    # Plane equation coefficients for each facet (used for ray/plane intersection).
    plane_a = (mesh_xyz[:, 1, 0] * (mesh_xyz[:, 2, 1] - mesh_xyz[:, 2, 2])
               + mesh_xyz[:, 1, 1] * (mesh_xyz[:, 2, 2] - mesh_xyz[:, 2, 0])
               + mesh_xyz[:, 1, 2] * (mesh_xyz[:, 2, 0] - mesh_xyz[:, 2, 1]))
    plane_b = (mesh_xyz[:, 2, 0] * (mesh_xyz[:, 0, 1] - mesh_xyz[:, 0, 2])
               + mesh_xyz[:, 2, 1] * (mesh_xyz[:, 0, 2] - mesh_xyz[:, 0, 0])
               + mesh_xyz[:, 2, 2] * (mesh_xyz[:, 0, 0] - mesh_xyz[:, 0, 1]))
    plane_c = (mesh_xyz[:, 0, 0] * (mesh_xyz[:, 1, 1] - mesh_xyz[:, 1, 2])
               + mesh_xyz[:, 0, 1] * (mesh_xyz[:, 1, 2] - mesh_xyz[:, 1, 0])
               + mesh_xyz[:, 0, 2] * (mesh_xyz[:, 1, 0] - mesh_xyz[:, 1, 1]))
    plane_d = (- mesh_xyz[:, 0, 0] * (mesh_xyz[:, 1, 1] * mesh_xyz[:, 2, 2]
                                       - mesh_xyz[:, 1, 2] * mesh_xyz[:, 2, 1])
               - mesh_xyz[:, 0, 1] * (mesh_xyz[:, 1, 2] * mesh_xyz[:, 2, 0]
                                       - mesh_xyz[:, 1, 0] * mesh_xyz[:, 2, 2])
               - mesh_xyz[:, 0, 2] * (mesh_xyz[:, 1, 0] * mesh_xyz[:, 2, 1]
                                       - mesh_xyz[:, 1, 1] * mesh_xyz[:, 2, 0]))
    # Treat near-zero C as exactly zero — the ray is parallel to the plane.
    plane_c_safe = np.where(np.abs(plane_c) < 1e-14, np.nan, plane_c)

    # Per-facet normal Z component (vectorised cross product of edge vectors,
    # same direction as ``plane_c`` up to sign).
    nz = ((mesh_xyz[:, 0, 1] - mesh_xyz[:, 0, 0]) *
          (mesh_xyz[:, 1, 2] - mesh_xyz[:, 1, 0])
          - (mesh_xyz[:, 1, 1] - mesh_xyz[:, 1, 0]) *
            (mesh_xyz[:, 0, 2] - mesh_xyz[:, 0, 0]))

    grid_co_z_arr = np.asarray(grid_co_z)

    # Pre-compute, per Y row, which facets straddle that y coordinate. This
    # mirrors the MATLAB optimisation but vectorised across X within each row.
    for loop_y in range(yi_lo, yi_hi + 1):
        cy = grid_co_y[loop_y]
        possible_y = np.where((facet_ymin <= cy) & (facet_ymax >= cy))[0]
        if possible_y.size == 0:
            continue

        # Pull out per-facet quantities for this row's candidate facets only.
        py_xmin = facet_xmin[possible_y]
        py_xmax = facet_xmax[possible_y]
        py_x1 = x1[possible_y]; py_y1 = y1[possible_y]
        py_x2 = x2[possible_y]; py_y2 = y2[possible_y]
        py_x3 = x3[possible_y]; py_y3 = y3[possible_y]
        py_pa = plane_a[possible_y]
        py_pb = plane_b[possible_y]
        py_pc = plane_c_safe[possible_y]
        py_pd = plane_d[possible_y]
        py_nz = nz[possible_y]

        for loop_x in range(xi_lo, xi_hi + 1):
            cx = grid_co_x[loop_x]
            in_x = (py_xmin <= cx) & (py_xmax >= cx)
            if not np.any(in_x):
                continue

            # Candidate facets for *this* (x, y) ray.
            cx1 = py_x1[in_x]; cy1 = py_y1[in_x]
            cx2 = py_x2[in_x]; cy2 = py_y2[in_x]
            cx3 = py_x3[in_x]; cy3 = py_y3[in_x]
            pa = py_pa[in_x]; pb = py_pb[in_x]; pc = py_pc[in_x]; pd = py_pd[in_x]
            nz_c = py_nz[in_x]

            # --- Vertex-on-ray special case (exact ray-vertex hit) ---
            on_v1 = (cx1 == cx) & (cy1 == cy)
            on_v2 = (cx2 == cx) & (cy2 == cy)
            on_v3 = (cx3 == cx) & (cy3 == cy)
            vertex_hit = on_v1 | on_v2 | on_v3

            crossing_mask = np.zeros_like(vertex_hit)
            needs_correction = False

            if np.any(vertex_hit):
                # If every facet sharing this vertex has its Z-normal pointing
                # the same way, the crossing is unambiguous — count each such
                # facet once. Otherwise, defer to neighbour-based correction.
                normals_z = nz_c[vertex_hit]
                if np.all(normals_z >= 0) or np.all(normals_z <= 0):
                    crossing_mask = crossing_mask | vertex_hit
                else:
                    needs_correction = True

            if not needs_correction:
                # --- Standard edge-side test (same maths as MATLAB original) ---
                # For each of the three edges, compute the predicted Y at the
                # far vertex's X using the opposite-edge slope, then do the
                # same prediction at the ray's X. If the signs match, the ray
                # is on the same side of that edge as the third vertex.
                # Repeating for all three edges = point-in-triangle.
                non_vertex = ~vertex_hit
                if np.any(non_vertex):
                    ax1 = cx1[non_vertex]; ay1 = cy1[non_vertex]
                    ax2 = cx2[non_vertex]; ay2 = cy2[non_vertex]
                    ax3 = cx3[non_vertex]; ay3 = cy3[non_vertex]

                    inside = _point_in_triangle_2d(
                        ax1, ay1, ax2, ay2, ax3, ay3, cx, cy
                    )
                    full_inside = np.zeros(len(cx1), dtype=bool)
                    full_inside[np.where(non_vertex)[0]] = inside
                    crossing_mask = crossing_mask | full_inside

            if needs_correction or not np.any(crossing_mask):
                if needs_correction:
                    correction_list.append((loop_x, loop_y))
                # No clean crossings — nothing to mark in this column.
                continue

            # --- Ray/plane intersection: solve for Z at the (x, y) ray. ---
            pa_c = pa[crossing_mask]
            pb_c = pb[crossing_mask]
            pc_c = pc[crossing_mask]
            pd_c = pd[crossing_mask]

            # NaN in pc means the facet is parallel to the ray; drop it.
            valid = ~np.isnan(pc_c)
            if not np.any(valid):
                continue

            z_cross = (-pd_c[valid] - pa_c[valid] * cx
                       - pb_c[valid] * cy) / pc_c[valid]

            # Drop intersections outside the mesh bbox (guarding rounding).
            z_cross = z_cross[
                (z_cross >= mesh_z_min - 1e-12)
                & (z_cross <= mesh_z_max + 1e-12)
            ]
            if z_cross.size == 0:
                continue

            # Round and dedupe to handle near-coincident crossings cleanly.
            z_cross = np.unique(np.round(z_cross * 1e12) / 1e12)

            if z_cross.size % 2 == 0:
                # Each consecutive pair (entry, exit) defines an interior span.
                column = np.zeros(voxcount_z, dtype=bool)
                for k in range(0, z_cross.size, 2):
                    column |= (
                        (grid_co_z_arr > z_cross[k])
                        & (grid_co_z_arr < z_cross[k + 1])
                    )
                grid_output[loop_x, loop_y, :] = column
            else:
                # Odd number of crossings = ambiguous; flag for repair below.
                correction_list.append((loop_x, loop_y))

    # ------------------------------------------------------------------
    # Patch ambiguous rays by majority vote of 8-connected neighbours.
    # ------------------------------------------------------------------
    if correction_list:
        corr = np.asarray(correction_list, dtype=np.int64)

        # If any correction sits on the array boundary, pad by one pixel so we
        # can sample 3x3 neighbourhoods without an index-out-of-bounds.
        need_pad = (
            corr[:, 0].min() == 0
            or corr[:, 0].max() == voxcount_x - 1
            or corr[:, 1].min() == 0
            or corr[:, 1].max() == voxcount_y - 1
        )
        if need_pad:
            grid_output = np.pad(
                grid_output, ((1, 1), (1, 1), (0, 0)), mode="constant"
            )
            corr = corr + 1

        for ix, iy in corr:
            neighbours = (
                grid_output[ix - 1, iy - 1, :].astype(np.int8)
                + grid_output[ix - 1, iy, :]
                + grid_output[ix - 1, iy + 1, :]
                + grid_output[ix,     iy - 1, :]
                + grid_output[ix,     iy + 1, :]
                + grid_output[ix + 1, iy - 1, :]
                + grid_output[ix + 1, iy, :]
                + grid_output[ix + 1, iy + 1, :]
            )
            grid_output[ix, iy, :] = neighbours >= 4

        if need_pad:
            grid_output = grid_output[1:-1, 1:-1, :]

    return grid_output


def _point_in_triangle_2d(x1, y1, x2, y2, x3, y3, px, py):
    """Vectorised 2-D point-in-triangle, using the sign-of-cross-product test.

    Equivalent in result to the three Y-prediction comparisons in the MATLAB
    original, but expressed more transparently. All inputs of the same length
    N (one per candidate facet); ``px``, ``py`` are scalars.
    """
    d1 = (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2)
    d2 = (px - x3) * (y2 - y3) - (x2 - x3) * (py - y3)
    d3 = (px - x1) * (y3 - y1) - (x3 - x1) * (py - y1)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def voxelise(
    grid_x: GridSpec,
    grid_y: GridSpec,
    grid_z: GridSpec,
    mesh: MeshInput,
    raydirection: str = "xyz",
    return_coords: bool = False,
):
    """Voxelise a closed triangular mesh.

    Parameters
    ----------
    grid_x, grid_y, grid_z
        Either a 1-D coordinate array (voxel centres) for that axis, or a
        scalar integer specifying how many voxels to auto-generate covering
        the mesh's bounding box. ``1`` collapses that axis to a single slice
        through the mesh centre.
    mesh
        One of: STL filename, ``(N, 3, 3)`` facet-vertex array, ``dict`` with
        ``'faces'`` and ``'vertices'``, or a ``(faces, vertices)`` tuple.
    raydirection : str, optional
        Which directions to ray-cast in. Any combination of ``'x'``, ``'y'``,
        ``'z'``. With multiple directions, results are combined by majority
        vote, which is the MATLAB default and the most robust option. Single
        directions are faster but vulnerable to artefacts where a ray exactly
        grazes a facet edge.
    return_coords : bool, optional
        If True, also return the resolved ``(grid_co_x, grid_co_y, grid_co_z)``
        coordinate vectors.

    Returns
    -------
    grid : ndarray of bool, shape (Nx, Ny, Nz)
        Voxel grid; True = inside the mesh.
    grid_co_x, grid_co_y, grid_co_z : ndarray
        (Only if ``return_coords=True``) Voxel centre coordinates per axis.

    Notes
    -----
    The mesh must be watertight. Open meshes will produce nonsense — every ray
    needs an even number of surface crossings for the inside/outside test to
    be well-defined.
    """
    raydirection = raydirection.lower()
    if raydirection not in _VALID_RAYDIRS:
        raise ValueError(f"Invalid raydirection {raydirection!r}")

    mesh_xyz = _load_mesh(mesh)

    mesh_x_min = mesh_xyz[:, 0, :].min()
    mesh_x_max = mesh_xyz[:, 0, :].max()
    mesh_y_min = mesh_xyz[:, 1, :].min()
    mesh_y_max = mesh_xyz[:, 1, :].max()
    mesh_z_min = mesh_xyz[:, 2, :].min()
    mesh_z_max = mesh_xyz[:, 2, :].max()

    grid_co_x = _resolve_axis_grid(grid_x, mesh_x_min, mesh_x_max)
    grid_co_y = _resolve_axis_grid(grid_y, mesh_y_min, mesh_y_max)
    grid_co_z = _resolve_axis_grid(grid_z, mesh_z_min, mesh_z_max)

    # If the user-supplied grid doesn't cover the full mesh, temporarily extend
    # it for the relevant axis so internal logic can find min/max pixel
    # indices, then trim back to the requested size at the end. Mirrors the
    # MATLAB original's gridcheckX/Y/Z bookkeeping.
    grid_check = {"x": 0, "y": 0, "z": 0}

    def _extend(coords, lo, hi, axis):
        """If the user's grid doesn't cover the mesh on this axis, prepend/append
        the missing endpoints so the internal pixel-index logic has somewhere
        to anchor. We trim back to the requested size at the very end."""
        flag = 0
        if coords.min() > lo:
            coords = np.concatenate(([lo], coords))
            flag += 1
        if coords.max() < hi:
            coords = np.concatenate((coords, [hi]))
            flag += 2
        grid_check[axis] = flag
        return coords

    if "x" in raydirection and (
        grid_co_x.min() > mesh_x_min or grid_co_x.max() < mesh_x_max
    ):
        grid_co_x = _extend(grid_co_x, mesh_x_min, mesh_x_max, "x")
    elif "y" in raydirection and (
        grid_co_y.min() > mesh_y_min or grid_co_y.max() < mesh_y_max
    ):
        grid_co_y = _extend(grid_co_y, mesh_y_min, mesh_y_max, "y")
    elif "z" in raydirection and (
        grid_co_z.min() > mesh_z_min or grid_co_z.max() < mesh_z_max
    ):
        grid_co_z = _extend(grid_co_z, mesh_z_min, mesh_z_max, "z")

    # Run one ray-cast pass per direction. For X and Y, we permute the mesh
    # axes so that the "ray direction" is always the third (Z) axis from the
    # internal function's point of view — then permute the result back.
    results = []

    if "x" in raydirection:
        # Ray along X: rename axes (x, y, z) -> (y, z, x).
        permuted_mesh = mesh_xyz[:, [1, 2, 0], :]
        out = _voxelise_internal(grid_co_y, grid_co_z, grid_co_x, permuted_mesh)
        # out has shape (Ny, Nz, Nx) in the permuted frame; reorder to (Nx, Ny, Nz).
        results.append(np.transpose(out, (2, 0, 1)))

    if "y" in raydirection:
        # Ray along Y: rename axes (x, y, z) -> (z, x, y).
        permuted_mesh = mesh_xyz[:, [2, 0, 1], :]
        out = _voxelise_internal(grid_co_z, grid_co_x, grid_co_y, permuted_mesh)
        results.append(np.transpose(out, (1, 2, 0)))

    if "z" in raydirection:
        out = _voxelise_internal(grid_co_x, grid_co_y, grid_co_z, mesh_xyz)
        results.append(out)

    if len(results) == 1:
        grid_output = results[0]
    else:
        # Majority vote across directions.
        stacked = np.stack(results, axis=-1).astype(np.int8)
        grid_output = stacked.sum(axis=-1) >= (len(results) / 2.0)

    # Trim back to the user's requested grid size where we previously extended.
    def _trim(arr, axis_idx, flag):
        if flag == 1:
            sl = [slice(None)] * 3
            sl[axis_idx] = slice(1, None)
            return arr[tuple(sl)]
        if flag == 2:
            sl = [slice(None)] * 3
            sl[axis_idx] = slice(None, -1)
            return arr[tuple(sl)]
        if flag == 3:
            sl = [slice(None)] * 3
            sl[axis_idx] = slice(1, -1)
            return arr[tuple(sl)]
        return arr

    if grid_check["x"]:
        grid_output = _trim(grid_output, 0, grid_check["x"])
        if grid_check["x"] == 1:
            grid_co_x = grid_co_x[1:]
        elif grid_check["x"] == 2:
            grid_co_x = grid_co_x[:-1]
        elif grid_check["x"] == 3:
            grid_co_x = grid_co_x[1:-1]
    if grid_check["y"]:
        grid_output = _trim(grid_output, 1, grid_check["y"])
        if grid_check["y"] == 1:
            grid_co_y = grid_co_y[1:]
        elif grid_check["y"] == 2:
            grid_co_y = grid_co_y[:-1]
        elif grid_check["y"] == 3:
            grid_co_y = grid_co_y[1:-1]
    if grid_check["z"]:
        grid_output = _trim(grid_output, 2, grid_check["z"])
        if grid_check["z"] == 1:
            grid_co_z = grid_co_z[1:]
        elif grid_check["z"] == 2:
            grid_co_z = grid_co_z[:-1]
        elif grid_check["z"] == 3:
            grid_co_z = grid_co_z[1:-1]

    if return_coords:
        return grid_output, grid_co_x, grid_co_y, grid_co_z
    return grid_output
