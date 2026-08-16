"""Synthetic watertight meshes used by the demo and tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def cube_with_hole(
    *,
    size: tuple[float, float, float] = (10.0, 8.0, 6.0),
    radius: float = 2.0,
    sections_per_side: int = 12,
) -> trimesh.Trimesh:
    """Create an axis-aligned rectangular block with a Z-axis through-hole."""
    if sections_per_side < 2:
        raise ValueError("sections_per_side must be at least 2")
    width, depth, height = (float(value) for value in size)
    if min(width, depth, height) <= 0.0:
        raise ValueError("all dimensions must be positive")
    if radius <= 0.0 or radius >= min(width, depth) / 2.0:
        raise ValueError("radius must fit strictly inside the block")

    half_width = width / 2.0
    half_depth = depth / 2.0
    half_height = height / 2.0
    corner_angle = float(np.arctan2(half_depth, half_width))
    corners = np.array(
        [
            corner_angle,
            np.pi - corner_angle,
            np.pi + corner_angle,
            2.0 * np.pi - corner_angle,
            2.0 * np.pi + corner_angle,
        ]
    )
    angle_parts = [
        np.linspace(corners[index], corners[index + 1], sections_per_side, endpoint=False)
        for index in range(4)
    ]
    angles = np.concatenate(angle_parts)
    cosines = np.cos(angles)
    sines = np.sin(angles)
    ray_scale = np.minimum(
        half_width / np.maximum(np.abs(cosines), 1e-15),
        half_depth / np.maximum(np.abs(sines), 1e-15),
    )
    inner_xy = np.column_stack((radius * cosines, radius * sines))
    outer_xy = np.column_stack((ray_scale * cosines, ray_scale * sines))

    def ring(xy: np.ndarray, z: float) -> np.ndarray:
        return np.column_stack((xy, np.full(len(xy), z)))

    top_inner = ring(inner_xy, half_height)
    top_outer = ring(outer_xy, half_height)
    bottom_inner = ring(inner_xy, -half_height)
    bottom_outer = ring(outer_xy, -half_height)
    vertices = np.vstack((top_inner, top_outer, bottom_inner, bottom_outer))
    count = len(angles)
    ti, to, bi, bo = 0, count, 2 * count, 3 * count

    faces: list[list[int]] = []
    for index in range(count):
        following = (index + 1) % count
        # Top (+Z) and bottom (-Z) annular faces.
        faces.extend(
            [
                [ti + index, to + index, to + following],
                [ti + index, to + following, ti + following],
                [bi + index, bo + following, bo + index],
                [bi + index, bi + following, bo + following],
                # Outer walls and the inward-facing wall of the hole.
                [bo + index, bo + following, to + following],
                [bo + index, to + following, to + index],
                [bi + index, ti + index, ti + following],
                [bi + index, ti + following, bi + following],
            ]
        )

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=True)
    mesh.fix_normals(multibody=True)
    return mesh


def write_cube_with_hole(path: str | Path, **kwargs: object) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cube_with_hole(**kwargs).export(destination, file_type="stl")
    return destination
