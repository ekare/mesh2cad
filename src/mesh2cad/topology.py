"""Milestone-one topology recognition and OpenCascade BRep construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh

from mesh2cad.csg import Boolean, Box, CsgEvaluationError, CsgNode, Cylinder, evaluate
from mesh2cad.mesh import MeshData
from mesh2cad.preview import tessellate_shape
from mesh2cad.types import CylinderFit, SegmentationResult, TopologySummary


class TopologyError(RuntimeError):
    """Raised when the current milestone cannot reconstruct the detected surfaces."""


class BackendUnavailable(TopologyError):
    """Raised when the OpenCascade Python bindings are unavailable."""


@dataclass(frozen=True)
class TopologyResult:
    shape: Any
    summary: TopologySummary
    vertex_distances: np.ndarray


ANALYTIC_ACCEPT_RMSE_RELATIVE = 2e-4
ANALYTIC_ACCEPT_MAX_RELATIVE = 1e-3


def _load_ocp() -> dict[str, Any]:
    try:
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
        from OCP.BRepBuilderAPI import (
            BRepBuilderAPI_MakeFace,
            BRepBuilderAPI_MakePolygon,
            BRepBuilderAPI_MakeSolid,
            BRepBuilderAPI_MakeVertex,
            BRepBuilderAPI_Sewing,
        )
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.BRepExtrema import BRepExtrema_DistShapeShape
        from OCP.BRepLib import BRepLib
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
        from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
        from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
        from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
    except ImportError as exc:
        raise BackendUnavailable(
            "OpenCascade bindings are unavailable; install the project dependencies "
            "with `pip install -e .` using Python 3.10-3.13"
        ) from exc
    return {
        "BRepAlgoAPI_Cut": BRepAlgoAPI_Cut,
        "BRepAlgoAPI_Fuse": BRepAlgoAPI_Fuse,
        "BRepAlgoAPI_Common": BRepAlgoAPI_Common,
        "BRepBuilderAPI_MakeFace": BRepBuilderAPI_MakeFace,
        "BRepBuilderAPI_MakePolygon": BRepBuilderAPI_MakePolygon,
        "BRepBuilderAPI_MakeSolid": BRepBuilderAPI_MakeSolid,
        "BRepBuilderAPI_MakeVertex": BRepBuilderAPI_MakeVertex,
        "BRepBuilderAPI_Sewing": BRepBuilderAPI_Sewing,
        "BRepCheck_Analyzer": BRepCheck_Analyzer,
        "BRepExtrema_DistShapeShape": BRepExtrema_DistShapeShape,
        "BRepLib": BRepLib,
        "BRepPrimAPI_MakeBox": BRepPrimAPI_MakeBox,
        "BRepPrimAPI_MakeCylinder": BRepPrimAPI_MakeCylinder,
        "ShapeUpgrade_UnifySameDomain": ShapeUpgrade_UnifySameDomain,
        "TopAbs_FACE": TopAbs_FACE,
        "TopAbs_SHELL": TopAbs_SHELL,
        "TopAbs_SOLID": TopAbs_SOLID,
        "TopExp_Explorer": TopExp_Explorer,
        "TopoDS": TopoDS,
        "gp_Ax2": gp_Ax2,
        "gp_Dir": gp_Dir,
        "gp_Pnt": gp_Pnt,
    }


def _count_subshapes(shape: Any, kind: Any, explorer_class: Any) -> int:
    explorer = explorer_class(shape, kind)
    count = 0
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def _recognize_box_frame(
    data: MeshData,
    segmentation: SegmentationResult,
    tolerance_degrees: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover a right-handed box frame from three orthogonal plane-normal families."""
    plane_regions = segmentation.plane_regions
    if len(plane_regions) < 6:
        raise TopologyError("Milestone one needs six detected outer box planes")

    normals = np.asarray([region.fit.normal for region in plane_regions], dtype=np.float64)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    areas = np.asarray([region.area for region in plane_regions], dtype=np.float64)
    parallel_threshold = float(np.cos(np.deg2rad(tolerance_degrees)))
    orthogonal_threshold = float(np.sin(np.deg2rad(tolerance_degrees)))

    best_axes: np.ndarray | None = None
    best_score = -np.inf
    order = np.argsort(-areas)
    for first_index in order:
        first = normals[first_index]
        for second_index in order:
            second_candidate = normals[second_index]
            if abs(float(np.dot(first, second_candidate))) > orthogonal_threshold:
                continue
            second = second_candidate - np.dot(second_candidate, first) * first
            second_norm = float(np.linalg.norm(second))
            if second_norm <= np.finfo(float).eps:
                continue
            second /= second_norm
            third = np.cross(first, second)
            third /= np.linalg.norm(third)
            axes = np.column_stack((first, second, third))

            family_areas = []
            valid = True
            for axis in axes.T:
                aligned = np.abs(normals @ axis) >= parallel_threshold
                if int(np.count_nonzero(aligned)) < 2:
                    valid = False
                    break
                family_areas.append(float(areas[aligned].sum()))
            if valid and (score := float(sum(family_areas))) > best_score:
                best_axes = axes
                best_score = score

    if best_axes is None:
        raise TopologyError("Detected planes do not describe an orthogonal box")

    local_vertices = np.asarray(data.vertices, dtype=np.float64) @ best_axes
    local_minimum = local_vertices.min(axis=0)
    local_maximum = local_vertices.max(axis=0)
    dimensions = local_maximum - local_minimum
    if np.any(dimensions <= data.scale * 1e-9):
        raise TopologyError("Detected box frame has a degenerate dimension")
    world_origin = best_axes @ local_minimum
    return world_origin, dimensions, best_axes


def recognize_csg(data: MeshData, segmentation: SegmentationResult) -> CsgNode:
    """Recognize milestone-one primitives as a kernel-independent boolean tree."""
    origin, dimensions, axes = _recognize_box_frame(data, segmentation)
    graph: CsgNode = Box(minimum=origin, dimensions=dimensions, axes=axes)

    diagonal = float(np.linalg.norm(dimensions))
    axis_threshold = np.cos(np.deg2rad(5.0))
    for region in segmentation.cylinder_regions:
        fit = region.fit
        if not isinstance(fit, CylinderFit):
            continue
        alignments = axes.T @ fit.axis
        axis_index = int(np.argmax(np.abs(alignments)))
        if abs(float(alignments[axis_index])) < axis_threshold:
            raise TopologyError("Milestone one needs cylinders aligned with a box axis")
        axis = axes[:, axis_index] * np.sign(alignments[axis_index])
        start = fit.axis_point - axis * diagonal
        graph = Boolean(
            "cut",
            graph,
            Cylinder(
                axis_point=start,
                axis=axis,
                radius=float(fit.radius),
                height=2.0 * diagonal,
            ),
        )
    return graph


def _validate_shape(shape: Any, ocp: dict[str, Any]) -> None:
    if shape.IsNull() or not ocp["BRepCheck_Analyzer"](shape, True).IsValid():
        raise TopologyError("Reconstructed OpenCascade shape is invalid")


def simplify_shape(shape: Any, ocp: dict[str, Any], scale: float) -> Any:
    """Merge redundant same-domain BRep faces without changing the solid geometry."""
    unify = ocp["ShapeUpgrade_UnifySameDomain"](shape, True, True, False)
    tolerance = max(float(scale) * 1e-7, 1e-9)
    unify.SetLinearTolerance(tolerance)
    unify.SetAngularTolerance(1e-7)
    unify.Build()
    simplified = unify.Shape()
    _validate_shape(simplified, ocp)
    return simplified


def _faithful_mesh_brep(data: MeshData, ocp: dict[str, Any]) -> Any:
    """Build a watertight BRep from the mesh and merge every coplanar domain."""
    if not data.mesh.is_watertight:
        raise TopologyError("Faithful BRep reconstruction requires a watertight mesh")
    if not data.mesh.is_winding_consistent:
        raise TopologyError("Faithful BRep reconstruction requires consistent face winding")

    tolerance = max(data.scale * 1e-7, 1e-9)
    point_type = ocp["gp_Pnt"]
    points = [point_type(*coordinates) for coordinates in data.vertices]
    sewing = ocp["BRepBuilderAPI_Sewing"](tolerance, True, True, True, False)
    for vertex_ids in data.faces:
        polygon = ocp["BRepBuilderAPI_MakePolygon"](
            points[int(vertex_ids[0])],
            points[int(vertex_ids[1])],
            points[int(vertex_ids[2])],
            True,
        )
        if not polygon.IsDone():
            raise TopologyError("OpenCascade could not build a mesh triangle wire")
        face = ocp["BRepBuilderAPI_MakeFace"](polygon.Wire(), True)
        if not face.IsDone():
            raise TopologyError("OpenCascade could not build a mesh triangle face")
        sewing.Add(face.Face())
    sewing.Perform()
    sewed = sewing.SewedShape()
    if sewed.IsNull():
        raise TopologyError("OpenCascade could not sew the mesh faces")

    shell_explorer = ocp["TopExp_Explorer"](sewed, ocp["TopAbs_SHELL"])
    shells = []
    while shell_explorer.More():
        shells.append(ocp["TopoDS"].Shell_s(shell_explorer.Current()))
        shell_explorer.Next()
    if len(shells) != 1:
        raise TopologyError(
            f"Faithful BRep reconstruction currently needs one closed shell; found {len(shells)}"
        )

    solid_builder = ocp["BRepBuilderAPI_MakeSolid"](shells[0])
    if not solid_builder.IsDone():
        raise TopologyError("OpenCascade could not turn the sewn shell into a solid")
    solid = solid_builder.Solid()
    if not ocp["BRepLib"].OrientClosedSolid_s(solid):
        raise TopologyError("OpenCascade could not orient the reconstructed solid")

    return simplify_shape(solid, ocp, data.scale)


def _topology_summary(
    shape: Any,
    ocp: dict[str, Any],
    *,
    recognizer: str,
    cylinders_cut: int,
) -> TopologySummary:
    return TopologySummary(
        recognizer=recognizer,
        valid=True,
        face_count=_count_subshapes(shape, ocp["TopAbs_FACE"], ocp["TopExp_Explorer"]),
        solid_count=_count_subshapes(shape, ocp["TopAbs_SOLID"], ocp["TopExp_Explorer"]),
        cylinders_cut=cylinders_cut,
    )


def _candidate_to_mesh_distances(
    data: MeshData,
    shape: Any,
    *,
    sample_count: int = 384,
    chunk_size: int = 8,
) -> np.ndarray:
    """Measure candidate-surface samples back to the source mesh without spatial extras."""
    preview = tessellate_shape(shape, max_triangles=20_000)
    vertices = np.asarray(preview.positions, dtype=np.float64).reshape(-1, 3)
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    candidate = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    points, _ = trimesh.sample.sample_surface(candidate, sample_count, seed=17)
    distances = []
    for start in range(0, len(points), chunk_size):
        _, batch_distances, _ = trimesh.proximity.closest_point_naive(
            data.mesh, points[start : start + chunk_size]
        )
        distances.append(batch_distances)
    return np.concatenate(distances)


def reconstruct_brep(data: MeshData, segmentation: SegmentationResult) -> TopologyResult:
    """Build competing BRep hypotheses and return the verified highest-fidelity result."""
    ocp = _load_ocp()
    try:
        graph = recognize_csg(data, segmentation)
        analytic_shape = simplify_shape(evaluate(graph, ocp), ocp, data.scale)
        _validate_shape(analytic_shape, ocp)
        analytic_distances = vertex_distances_to_shape(data.vertices, analytic_shape)
        reverse_distances = _candidate_to_mesh_distances(data, analytic_shape)
        relative_rmse = float(np.sqrt(np.mean(np.square(analytic_distances)))) / data.scale
        relative_max = float(np.max(analytic_distances)) / data.scale
        reverse_relative_rmse = (
            float(np.sqrt(np.mean(np.square(reverse_distances)))) / data.scale
        )
        reverse_relative_max = float(np.max(reverse_distances)) / data.scale
        if (
            relative_rmse <= ANALYTIC_ACCEPT_RMSE_RELATIVE
            and relative_max <= ANALYTIC_ACCEPT_MAX_RELATIVE
            and reverse_relative_rmse <= ANALYTIC_ACCEPT_RMSE_RELATIVE
            and reverse_relative_max <= ANALYTIC_ACCEPT_MAX_RELATIVE
        ):
            return TopologyResult(
                shape=analytic_shape,
                summary=_topology_summary(
                    analytic_shape,
                    ocp,
                    recognizer="verified_oriented_box_with_through_cylinders",
                    cylinders_cut=len(segmentation.cylinder_regions),
                ),
                vertex_distances=analytic_distances,
            )
    except (CsgEvaluationError, TopologyError):
        pass

    faithful_shape = _faithful_mesh_brep(data, ocp)
    return TopologyResult(
        shape=faithful_shape,
        summary=_topology_summary(
            faithful_shape,
            ocp,
            recognizer="faithful_unified_mesh_brep",
            cylinders_cut=0,
        ),
        vertex_distances=np.zeros(len(data.vertices), dtype=np.float64),
    )


def vertex_distances_to_shape(vertices: np.ndarray, shape: Any) -> np.ndarray:
    """Compute exact OpenCascade point-to-BRep distances for metric reporting."""
    ocp = _load_ocp()
    distances: list[float] = []
    for coordinates in np.asarray(vertices, dtype=np.float64):
        vertex = ocp["BRepBuilderAPI_MakeVertex"](ocp["gp_Pnt"](*coordinates)).Vertex()
        query = ocp["BRepExtrema_DistShapeShape"](vertex, shape)
        query.Perform()
        if not query.IsDone():
            raise TopologyError("OpenCascade failed to calculate a reconstruction distance")
        distances.append(float(query.Value()))
    return np.asarray(distances, dtype=np.float64)
