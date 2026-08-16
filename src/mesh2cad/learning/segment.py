"""Turn neural triangle predictions into geometrically verified analytic regions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mesh2cad.fitting import FitError, fit_cylinder, fit_plane
from mesh2cad.learning.features import SURFACE_CLASSES
from mesh2cad.learning.infer import predict_mesh
from mesh2cad.mesh import MeshData
from mesh2cad.segmentation import SegmentationConfig
from mesh2cad.types import SegmentationResult, SurfaceRegion


def _cylinder_normals_match(
    data: MeshData,
    faces: np.ndarray,
    fit: Any,
    *,
    radial_tolerance_degrees: float,
) -> bool:
    normals = np.asarray(data.mesh.face_normals[faces], dtype=np.float64)
    areas = np.asarray(data.mesh.area_faces[faces], dtype=np.float64)
    normal_coherence = float(np.linalg.norm(np.average(normals, axis=0, weights=areas)))
    if normal_coherence > np.cos(np.deg2rad(10.0)):
        return False
    axial_alignment = np.abs(normals @ fit.axis)
    if float(np.quantile(axial_alignment, 0.95)) > np.sin(np.deg2rad(10.0)):
        return False
    centers = np.asarray(data.mesh.triangles_center[faces], dtype=np.float64)
    relative = centers - fit.axis_point
    radial = relative - np.outer(relative @ fit.axis, fit.axis)
    lengths = np.linalg.norm(radial, axis=1)
    if np.any(lengths <= np.finfo(float).eps):
        return False
    radial /= lengths[:, None]
    radial_alignment = np.abs(np.einsum("ij,ij->i", normals, radial))
    return float(np.quantile(radial_alignment, 0.10)) >= np.cos(
        np.deg2rad(radial_tolerance_degrees)
    )


def segmentation_from_prediction(
    data: MeshData,
    prediction: dict[str, Any],
    *,
    config: SegmentationConfig | None = None,
) -> SegmentationResult:
    """Fit supported analytic surfaces to predicted instances and reject bad fits."""
    config = config or SegmentationConfig()
    surface_labels = np.asarray(prediction["surface_labels"], dtype=np.int64)
    instance_labels = np.asarray(prediction["instance_labels"], dtype=np.int64)
    face_count = len(data.mesh.faces)
    if len(surface_labels) != face_count or len(instance_labels) != face_count:
        raise ValueError("Neural prediction does not match the cleaned mesh")

    total_area = float(data.mesh.area)
    face_region_ids = np.full(face_count, -1, dtype=np.int64)
    regions: list[SurfaceRegion] = []
    for instance_id in np.unique(instance_labels):
        faces = np.flatnonzero(instance_labels == instance_id).astype(np.int64)
        if not len(faces):
            continue
        semantic_ids, semantic_counts = np.unique(surface_labels[faces], return_counts=True)
        semantic_id = int(semantic_ids[int(np.argmax(semantic_counts))])
        if semantic_id < 0 or semantic_id >= len(SURFACE_CLASSES):
            continue
        predicted_kind = SURFACE_CLASSES[semantic_id]
        area = float(data.mesh.area_faces[faces].sum())
        candidates: list[tuple[str, Any, float]] = []
        if (
            len(faces) >= config.min_plane_faces
            and area >= config.min_plane_area_fraction * total_area
        ):
            try:
                fit = fit_plane(data.mesh, faces)
                tolerance = (
                    max(config.plane_distance_tolerance_fraction * 10.0, 1e-4)
                    * data.scale
                )
                if fit.max_error <= tolerance:
                    candidates.append(
                        ("plane", fit, fit.rms_error / max(tolerance, np.finfo(float).eps))
                    )
            except FitError:
                pass
        if len(faces) >= config.min_cylinder_faces:
            try:
                fit = fit_cylinder(data.mesh, faces)
                tolerance = config.cylinder_fit_tolerance_fraction * data.scale
                if fit.max_error <= tolerance and _cylinder_normals_match(
                    data,
                    faces,
                    fit,
                    radial_tolerance_degrees=config.cylinder_neighbour_angle_degrees,
                ):
                    candidates.append(
                        ("cylinder", fit, fit.rms_error / max(tolerance, np.finfo(float).eps))
                    )
            except FitError:
                pass
        if not candidates:
            continue

        # The network owns the instance proposal. Prefer its semantic class when the geometry
        # verifies it; otherwise recover a mislabeled analytic patch with the best normalized fit.
        predicted_candidates = [item for item in candidates if item[0] == predicted_kind]
        kind, fit, _score = min(predicted_candidates or candidates, key=lambda item: item[2])

        region_id = len(regions)
        face_region_ids[faces] = region_id
        regions.append(
            SurfaceRegion(
                id=region_id,
                kind=kind,
                face_indices=faces,
                area=area,
                fit=fit,
            )
        )

    return SegmentationResult(
        regions=tuple(regions),
        face_region_ids=face_region_ids,
        unassigned_faces=np.flatnonzero(face_region_ids < 0).astype(np.int64),
    )


def segment_with_model(
    data: MeshData,
    checkpoint_path: str | Path,
    *,
    config: SegmentationConfig | None = None,
    device_name: str = "auto",
    boundary_threshold: float = 0.5,
) -> SegmentationResult:
    """Run neural inference, then analytically fit supported predicted regions."""
    prediction = predict_mesh(
        data,
        checkpoint_path,
        device_name=device_name,
        boundary_threshold=boundary_threshold,
    )
    return segmentation_from_prediction(data, prediction, config=config)
