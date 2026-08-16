"""Reconstruction quality metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from mesh2cad.mesh import MeshData
from mesh2cad.preview import ShapePreview, tessellate_shape
from mesh2cad.types import ReconstructionMetrics, SegmentationResult


@dataclass(frozen=True)
class FidelityMeasurement:
    """Symmetric sampled surface comparison between the source mesh and rebuilt BRep."""

    tolerance: float
    geometric_fidelity: float
    mean_error: float
    rms_error: float
    p95_error: float
    p99_error: float
    max_error: float
    source_to_step_coverage: float
    step_to_source_coverage: float
    surface_area_relative_error: float
    volume_relative_error: float
    normal_median_error_degrees: float
    normal_p95_error_degrees: float
    sample_count: int


def _triangle_mesh(positions: np.ndarray) -> trimesh.Trimesh:
    triangles = np.asarray(positions, dtype=np.float64).reshape(-1, 3, 3)
    vertices = triangles.reshape(-1, 3)
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True, validate=False)


def _closest_surface(
    target: trimesh.Trimesh, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact point-to-triangle distances with a low-memory dependency fallback."""
    try:
        _, distances, triangle_ids = trimesh.proximity.closest_point(target, points)
        return np.asarray(distances), np.asarray(triangle_ids, dtype=np.int64)
    except (ImportError, ModuleNotFoundError):  # pragma: no cover - rtree is a dependency
        # Keep minimal installations functional. The naive implementation allocates an
        # O(points * triangles) working set, so bound each batch by triangle count.
        batch_size = max(1, min(32, 1_000_000 // max(1, len(target.faces))))
        distance_batches = []
        triangle_batches = []
        for start in range(0, len(points), batch_size):
            _, distances, triangle_ids = trimesh.proximity.closest_point_naive(
                target, points[start : start + batch_size]
            )
            distance_batches.append(distances)
            triangle_batches.append(triangle_ids)
        return np.concatenate(distance_batches), np.concatenate(triangle_batches).astype(np.int64)


def _normal_errors(
    source_normals: np.ndarray, target_normals: np.ndarray
) -> np.ndarray:
    dot = np.einsum("ij,ij->i", source_normals, target_normals)
    # Compare tangent-plane orientation. STL winding is frequently inconsistent at import and
    # should not turn a geometrically identical surface into a 180-degree error.
    return np.rad2deg(np.arccos(np.clip(np.abs(dot), 0.0, 1.0)))


def geometric_fidelity(
    source: trimesh.Trimesh,
    reconstructed_positions: np.ndarray,
    *,
    tolerance: float | None = None,
    sample_count: int = 20_000,
) -> FidelityMeasurement:
    """Measure symmetric, area-weighted source/reconstruction surface fidelity."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    reconstructed = _triangle_mesh(reconstructed_positions)
    if len(reconstructed.faces) == 0:
        raise ValueError("reconstructed surface contains no triangles")
    diagonal = float(np.linalg.norm(source.extents))
    resolved_tolerance = float(tolerance or max(diagonal * 1e-4, 1e-9))
    if resolved_tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    source_points, source_faces = trimesh.sample.sample_surface(
        source, sample_count, seed=17
    )
    step_points, step_faces = trimesh.sample.sample_surface(
        reconstructed, sample_count, seed=23
    )
    source_to_step, closest_step_faces = _closest_surface(reconstructed, source_points)
    step_to_source, closest_source_faces = _closest_surface(source, step_points)
    distances = np.concatenate([source_to_step, step_to_source])

    source_normal_error = _normal_errors(
        np.asarray(source.face_normals[source_faces]),
        np.asarray(reconstructed.face_normals[closest_step_faces]),
    )
    step_normal_error = _normal_errors(
        np.asarray(reconstructed.face_normals[step_faces]),
        np.asarray(source.face_normals[closest_source_faces]),
    )
    normal_errors = np.concatenate([source_normal_error, step_normal_error])
    source_coverage = float(np.mean(source_to_step <= resolved_tolerance))
    step_coverage = float(np.mean(step_to_source <= resolved_tolerance))
    source_area = float(source.area)
    source_volume = abs(float(source.volume))
    reconstructed_volume = abs(float(reconstructed.volume))

    return FidelityMeasurement(
        tolerance=resolved_tolerance,
        geometric_fidelity=0.5 * (source_coverage + step_coverage),
        mean_error=float(np.mean(distances)),
        rms_error=float(np.sqrt(np.mean(np.square(distances)))),
        p95_error=float(np.percentile(distances, 95.0)),
        p99_error=float(np.percentile(distances, 99.0)),
        max_error=float(np.max(distances)),
        source_to_step_coverage=source_coverage,
        step_to_source_coverage=step_coverage,
        surface_area_relative_error=abs(float(reconstructed.area) - source_area)
        / max(source_area, np.finfo(float).eps),
        volume_relative_error=abs(reconstructed_volume - source_volume)
        / max(source_volume, np.finfo(float).eps),
        normal_median_error_degrees=float(np.median(normal_errors)),
        normal_p95_error_degrees=float(np.percentile(normal_errors, 95.0)),
        sample_count=2 * sample_count,
    )


def shape_fidelity(
    data: MeshData,
    shape: object,
    *,
    tolerance: float | None = None,
    sample_count: int = 20_000,
    max_tessellation_triangles: int = 200_000,
) -> tuple[FidelityMeasurement, ShapePreview]:
    """Tessellate a rebuilt BRep and compare it with the source STL surface."""
    preview = tessellate_shape(
        shape,
        relative_deflection=5e-4,
        angular_deflection=0.1,
        max_triangles=max_tessellation_triangles,
    )
    return (
        geometric_fidelity(
            data.mesh,
            preview.positions,
            tolerance=tolerance,
            sample_count=sample_count,
        ),
        preview,
    )


def preview_deviations(source: trimesh.Trimesh, positions: np.ndarray) -> np.ndarray:
    """Return source-surface distance for each reconstructed preview triangle."""
    triangles = np.asarray(positions, dtype=np.float64).reshape(-1, 3, 3)
    centroids = triangles.mean(axis=1)
    distances, _ = _closest_surface(source, centroids)
    return np.asarray(distances, dtype=np.float32)


def analytic_surface_metrics(
    data: MeshData,
    segmentation: SegmentationResult,
    *,
    brep_distances: np.ndarray | None = None,
    fidelity: FidelityMeasurement | None = None,
    fidelity_tessellation_sampled: bool | None = None,
) -> ReconstructionMetrics:
    """Measure mesh vertices against their assigned infinite analytic surfaces."""
    squared_error_sum = 0.0
    weight_sum = 0.0
    max_error = 0.0
    assigned_area = 0.0
    assigned_faces = 0

    for region in segmentation.regions:
        triangles = np.asarray(data.mesh.triangles[region.face_indices], dtype=np.float64)
        points = triangles.reshape(-1, 3)
        errors = region.fit.distances(points)
        face_weights = np.asarray(data.mesh.area_faces[region.face_indices], dtype=np.float64)
        weights = np.repeat(face_weights / 3.0, 3)
        squared_error_sum += float(np.dot(weights, np.square(errors)))
        weight_sum += float(weights.sum())
        max_error = max(max_error, float(errors.max(initial=0.0)))
        assigned_area += region.area
        assigned_faces += len(region.face_indices)

    rmse = float(np.sqrt(squared_error_sum / weight_sum)) if weight_sum else float("inf")
    brep_rmse: float | None = None
    brep_max: float | None = None
    if brep_distances is not None and len(brep_distances):
        brep_rmse = float(np.sqrt(np.mean(np.square(brep_distances))))
        brep_max = float(np.max(brep_distances))

    return ReconstructionMetrics(
        analytic_surface_rmse=rmse,
        analytic_surface_max_error=max_error,
        assigned_area_fraction=assigned_area / float(data.mesh.area),
        assigned_face_fraction=assigned_faces / len(data.mesh.faces),
        brep_vertex_rmse=brep_rmse,
        brep_vertex_max_error=brep_max,
        fidelity_tolerance=None if fidelity is None else fidelity.tolerance,
        geometric_fidelity=None if fidelity is None else fidelity.geometric_fidelity,
        bidirectional_mean_error=None if fidelity is None else fidelity.mean_error,
        bidirectional_rms_error=None if fidelity is None else fidelity.rms_error,
        bidirectional_p95_error=None if fidelity is None else fidelity.p95_error,
        bidirectional_p99_error=None if fidelity is None else fidelity.p99_error,
        bidirectional_max_error=None if fidelity is None else fidelity.max_error,
        source_to_step_coverage=(
            None if fidelity is None else fidelity.source_to_step_coverage
        ),
        step_to_source_coverage=(
            None if fidelity is None else fidelity.step_to_source_coverage
        ),
        surface_area_relative_error=(
            None if fidelity is None else fidelity.surface_area_relative_error
        ),
        volume_relative_error=None if fidelity is None else fidelity.volume_relative_error,
        normal_median_error_degrees=(
            None if fidelity is None else fidelity.normal_median_error_degrees
        ),
        normal_p95_error_degrees=(
            None if fidelity is None else fidelity.normal_p95_error_degrees
        ),
        fidelity_sample_count=None if fidelity is None else fidelity.sample_count,
        fidelity_tessellation_sampled=fidelity_tessellation_sampled,
    )
