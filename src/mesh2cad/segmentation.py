"""Deterministic region growing for planar and cylindrical mesh patches."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from mesh2cad.fitting import FitError, fit_cylinder, fit_plane
from mesh2cad.mesh import MeshData
from mesh2cad.types import IntArray, SegmentationResult, SurfaceRegion


@dataclass(frozen=True)
class SegmentationConfig:
    plane_normal_tolerance_degrees: float = 1.0
    plane_distance_tolerance_fraction: float = 1e-5
    min_plane_faces: int = 2
    min_plane_area_fraction: float = 1e-3
    cylinder_neighbour_angle_degrees: float = 35.0
    cylinder_fit_tolerance_fraction: float = 1e-3
    min_cylinder_faces: int = 12


def _grow_planar_candidate(
    data: MeshData,
    seed: int,
    available: np.ndarray,
    visited: np.ndarray,
    config: SegmentationConfig,
) -> IntArray:
    normals = np.asarray(data.mesh.face_normals)
    centers = np.asarray(data.mesh.triangles_center)
    seed_normal = normals[seed]
    seed_center = centers[seed]
    min_dot = np.cos(np.deg2rad(config.plane_normal_tolerance_degrees))
    max_distance = config.plane_distance_tolerance_fraction * data.scale

    queue: deque[int] = deque([seed])
    visited[seed] = True
    members: list[int] = []
    while queue:
        current = queue.popleft()
        members.append(current)
        for neighbour_value in data.neighbours[current]:
            neighbour = int(neighbour_value)
            if visited[neighbour] or not available[neighbour]:
                continue
            normal_match = float(np.dot(normals[neighbour], seed_normal)) >= min_dot
            plane_distance = abs(float(np.dot(centers[neighbour] - seed_center, seed_normal)))
            if normal_match and plane_distance <= max_distance:
                visited[neighbour] = True
                queue.append(neighbour)
    return np.asarray(members, dtype=np.int64)


def _grow_smooth_candidate(
    data: MeshData,
    seed: int,
    available: np.ndarray,
    visited: np.ndarray,
    max_angle_degrees: float,
) -> IntArray:
    normals = np.asarray(data.mesh.face_normals)
    min_dot = np.cos(np.deg2rad(max_angle_degrees))
    queue: deque[int] = deque([seed])
    visited[seed] = True
    members: list[int] = []
    while queue:
        current = queue.popleft()
        members.append(current)
        for neighbour_value in data.neighbours[current]:
            neighbour = int(neighbour_value)
            if visited[neighbour] or not available[neighbour]:
                continue
            if float(np.dot(normals[current], normals[neighbour])) >= min_dot:
                visited[neighbour] = True
                queue.append(neighbour)
    return np.asarray(members, dtype=np.int64)


def _touches_smooth_nonplanar_neighbour(
    data: MeshData,
    candidate: IntArray,
    *,
    plane_tolerance_degrees: float,
    smooth_tolerance_degrees: float,
) -> bool:
    """Distinguish a two-triangle box face from one flat chord of a curved surface."""
    normals = np.asarray(data.mesh.face_normals)
    members = set(int(face) for face in candidate)
    plane_dot = float(np.cos(np.deg2rad(plane_tolerance_degrees)))
    smooth_dot = float(np.cos(np.deg2rad(smooth_tolerance_degrees)))
    for face in members:
        for neighbour_value in data.neighbours[face]:
            neighbour = int(neighbour_value)
            if neighbour in members:
                continue
            normal_dot = float(np.dot(normals[face], normals[neighbour]))
            if smooth_dot <= normal_dot < plane_dot:
                return True
    return False


def segment_surfaces(
    data: MeshData, config: SegmentationConfig | None = None
) -> SegmentationResult:
    """Segment planes first, then fit cylinders to remaining smooth components."""
    config = config or SegmentationConfig()
    face_count = len(data.mesh.faces)
    total_area = float(data.mesh.area)
    face_region_ids = np.full(face_count, -1, dtype=np.int64)
    regions: list[SurfaceRegion] = []

    available = np.ones(face_count, dtype=bool)
    visited = np.zeros(face_count, dtype=bool)
    for seed in range(face_count):
        if visited[seed]:
            continue
        candidate = _grow_planar_candidate(data, seed, available, visited, config)
        area = float(data.mesh.area_faces[candidate].sum())
        if (
            len(candidate) < config.min_plane_faces
            or area < config.min_plane_area_fraction * total_area
            or (
                len(candidate) == 2
                and _touches_smooth_nonplanar_neighbour(
                    data,
                    candidate,
                    plane_tolerance_degrees=config.plane_normal_tolerance_degrees,
                    smooth_tolerance_degrees=config.cylinder_neighbour_angle_degrees,
                )
            )
        ):
            continue
        fit = fit_plane(data.mesh, candidate)
        region_id = len(regions)
        face_region_ids[candidate] = region_id
        regions.append(
            SurfaceRegion(
                id=region_id,
                kind="plane",
                face_indices=candidate,
                area=area,
                fit=fit,
            )
        )

    available = face_region_ids < 0
    visited = ~available.copy()
    for seed in range(face_count):
        if visited[seed] or not available[seed]:
            continue
        candidate = _grow_smooth_candidate(
            data,
            seed,
            available,
            visited,
            config.cylinder_neighbour_angle_degrees,
        )
        if len(candidate) < config.min_cylinder_faces:
            continue
        try:
            fit = fit_cylinder(data.mesh, candidate)
        except FitError:
            continue
        if fit.max_error > config.cylinder_fit_tolerance_fraction * data.scale:
            continue
        area = float(data.mesh.area_faces[candidate].sum())
        region_id = len(regions)
        face_region_ids[candidate] = region_id
        regions.append(
            SurfaceRegion(
                id=region_id,
                kind="cylinder",
                face_indices=candidate,
                area=area,
                fit=fit,
            )
        )

    return SegmentationResult(
        regions=tuple(regions),
        face_region_ids=face_region_ids,
        unassigned_faces=np.flatnonzero(face_region_ids < 0).astype(np.int64),
    )
