"""Shared data structures used by the reconstruction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
SurfaceKind = Literal["plane", "cylinder"]


def _vector(value: FloatArray) -> list[float]:
    return [float(item) for item in value]


@dataclass(frozen=True)
class PlaneFit:
    normal: FloatArray
    offset: float
    origin: FloatArray
    rms_error: float
    max_error: float

    def distances(self, points: FloatArray) -> FloatArray:
        return np.abs(points @ self.normal - self.offset)

    def to_dict(self) -> dict[str, Any]:
        return {
            "normal": _vector(self.normal),
            "offset": float(self.offset),
            "origin": _vector(self.origin),
            "rms_error": float(self.rms_error),
            "max_error": float(self.max_error),
        }


@dataclass(frozen=True)
class CylinderFit:
    axis: FloatArray
    axis_point: FloatArray
    radius: float
    axial_min: float
    axial_max: float
    rms_error: float
    max_error: float

    @property
    def height(self) -> float:
        return self.axial_max - self.axial_min

    def distances(self, points: FloatArray) -> FloatArray:
        relative = points - self.axis_point
        axial = np.outer(relative @ self.axis, self.axis)
        radial = np.linalg.norm(relative - axial, axis=1)
        return np.abs(radial - self.radius)

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": _vector(self.axis),
            "axis_point": _vector(self.axis_point),
            "radius": float(self.radius),
            "axial_min": float(self.axial_min),
            "axial_max": float(self.axial_max),
            "height": float(self.height),
            "rms_error": float(self.rms_error),
            "max_error": float(self.max_error),
        }


@dataclass(frozen=True)
class SurfaceRegion:
    id: int
    kind: SurfaceKind
    face_indices: IntArray
    area: float
    fit: PlaneFit | CylinderFit

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "face_count": int(len(self.face_indices)),
            "area": float(self.area),
            "fit": self.fit.to_dict(),
        }


@dataclass(frozen=True)
class SegmentationResult:
    regions: tuple[SurfaceRegion, ...]
    face_region_ids: IntArray
    unassigned_faces: IntArray

    @property
    def plane_regions(self) -> tuple[SurfaceRegion, ...]:
        return tuple(region for region in self.regions if region.kind == "plane")

    @property
    def cylinder_regions(self) -> tuple[SurfaceRegion, ...]:
        return tuple(region for region in self.regions if region.kind == "cylinder")


@dataclass(frozen=True)
class ReconstructionMetrics:
    analytic_surface_rmse: float
    analytic_surface_max_error: float
    assigned_area_fraction: float
    assigned_face_fraction: float
    brep_vertex_rmse: float | None = None
    brep_vertex_max_error: float | None = None
    fidelity_tolerance: float | None = None
    geometric_fidelity: float | None = None
    bidirectional_mean_error: float | None = None
    bidirectional_rms_error: float | None = None
    bidirectional_p95_error: float | None = None
    bidirectional_p99_error: float | None = None
    bidirectional_max_error: float | None = None
    source_to_step_coverage: float | None = None
    step_to_source_coverage: float | None = None
    surface_area_relative_error: float | None = None
    volume_relative_error: float | None = None
    normal_median_error_degrees: float | None = None
    normal_p95_error_degrees: float | None = None
    fidelity_sample_count: int | None = None
    fidelity_tessellation_sampled: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "analytic_surface_rmse": float(self.analytic_surface_rmse),
            "analytic_surface_max_error": float(self.analytic_surface_max_error),
            "assigned_area_fraction": float(self.assigned_area_fraction),
            "assigned_face_fraction": float(self.assigned_face_fraction),
            "brep_vertex_rmse": (
                None if self.brep_vertex_rmse is None else float(self.brep_vertex_rmse)
            ),
            "brep_vertex_max_error": (
                None if self.brep_vertex_max_error is None else float(self.brep_vertex_max_error)
            ),
            "fidelity_tolerance": self.fidelity_tolerance,
            "geometric_fidelity": self.geometric_fidelity,
            "bidirectional_mean_error": self.bidirectional_mean_error,
            "bidirectional_rms_error": self.bidirectional_rms_error,
            "bidirectional_p95_error": self.bidirectional_p95_error,
            "bidirectional_p99_error": self.bidirectional_p99_error,
            "bidirectional_max_error": self.bidirectional_max_error,
            "source_to_step_coverage": self.source_to_step_coverage,
            "step_to_source_coverage": self.step_to_source_coverage,
            "surface_area_relative_error": self.surface_area_relative_error,
            "volume_relative_error": self.volume_relative_error,
            "normal_median_error_degrees": self.normal_median_error_degrees,
            "normal_p95_error_degrees": self.normal_p95_error_degrees,
            "fidelity_sample_count": self.fidelity_sample_count,
            "fidelity_tessellation_sampled": self.fidelity_tessellation_sampled,
        }


@dataclass(frozen=True)
class TopologySummary:
    recognizer: str
    valid: bool
    face_count: int
    solid_count: int
    cylinders_cut: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "recognizer": self.recognizer,
            "valid": self.valid,
            "face_count": self.face_count,
            "solid_count": self.solid_count,
            "cylinders_cut": self.cylinders_cut,
        }


@dataclass
class ConversionResult:
    input_path: str
    output_path: str | None
    report_path: str | None
    mesh_vertices: int
    mesh_faces: int
    mesh_bounds: FloatArray
    segmentation: SegmentationResult
    metrics: ReconstructionMetrics
    mesh_quality: Any | None = None
    segmentation_engine: str = "deterministic"
    topology: TopologySummary | None = None
    _shape: Any = field(default=None, repr=False)
    _comparison_preview: Any = field(default=None, repr=False)

    def cad_quality(self) -> dict[str, Any] | None:
        if self.topology is None:
            return None
        analytic = self.topology.recognizer != "faithful_unified_mesh_brep"
        return {
            "representation": "analytic-csg" if analytic else "mesh-derived-brep",
            "analytic_reconstruction": analytic,
            "analytic_area_fraction": float(self.metrics.assigned_area_fraction),
            "source_triangle_count": self.mesh_faces,
            "brep_face_count": self.topology.face_count,
            "face_reduction_fraction": float(
                1.0 - self.topology.face_count / max(1, self.mesh_faces)
            ),
            "valid_solid": self.topology.valid and self.topology.solid_count > 0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input_path,
            "output": self.output_path,
            "mesh": {
                "vertices": self.mesh_vertices,
                "faces": self.mesh_faces,
                "bounds": [[float(item) for item in row] for row in self.mesh_bounds],
                "quality": (
                    None if self.mesh_quality is None else self.mesh_quality.to_dict()
                ),
            },
            "segmentation": {
                "engine": self.segmentation_engine,
                "region_count": len(self.segmentation.regions),
                "plane_count": len(self.segmentation.plane_regions),
                "cylinder_count": len(self.segmentation.cylinder_regions),
                "unassigned_face_count": int(len(self.segmentation.unassigned_faces)),
                "regions": [region.to_dict() for region in self.segmentation.regions],
            },
            "metrics": self.metrics.to_dict(),
            "cad_quality": self.cad_quality(),
            "topology": None if self.topology is None else self.topology.to_dict(),
        }
