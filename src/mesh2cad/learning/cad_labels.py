"""Tessellate STEP BReps while retaining source-face ground truth labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from mesh2cad.learning.features import SURFACE_TO_ID, TriangleGraphSample, build_sample
from mesh2cad.topology import BackendUnavailable


class CadLabelError(RuntimeError):
    """Raised when a CAD model cannot be read or tessellated."""


def _ocp() -> dict[str, Any]:
    try:
        from OCP.Bnd import Bnd_Box
        from OCP.BRep import BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.BRepBndLib import BRepBndLib
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.GeomAbs import (
            GeomAbs_BSplineSurface,
            GeomAbs_Cone,
            GeomAbs_Cylinder,
            GeomAbs_Plane,
            GeomAbs_Sphere,
            GeomAbs_Torus,
        )
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Reader
        from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopLoc import TopLoc_Location
        from OCP.TopoDS import TopoDS
    except ImportError as exc:
        raise BackendUnavailable("CAD labeling requires cadquery-ocp") from exc
    return {
        "Bnd_Box": Bnd_Box,
        "BRep_Tool": BRep_Tool,
        "BRepAdaptor_Surface": BRepAdaptor_Surface,
        "BRepBndLib": BRepBndLib,
        "BRepCheck_Analyzer": BRepCheck_Analyzer,
        "BRepMesh_IncrementalMesh": BRepMesh_IncrementalMesh,
        "GeomAbs_BSplineSurface": GeomAbs_BSplineSurface,
        "GeomAbs_Cone": GeomAbs_Cone,
        "GeomAbs_Cylinder": GeomAbs_Cylinder,
        "GeomAbs_Plane": GeomAbs_Plane,
        "GeomAbs_Sphere": GeomAbs_Sphere,
        "GeomAbs_Torus": GeomAbs_Torus,
        "IFSelect_RetDone": IFSelect_RetDone,
        "STEPControl_Reader": STEPControl_Reader,
        "TopAbs_FACE": TopAbs_FACE,
        "TopAbs_REVERSED": TopAbs_REVERSED,
        "TopAbs_SOLID": TopAbs_SOLID,
        "TopExp_Explorer": TopExp_Explorer,
        "TopLoc_Location": TopLoc_Location,
        "TopoDS": TopoDS,
    }


def read_step(path: str | Path) -> Any:
    ocp = _ocp()
    reader = ocp["STEPControl_Reader"]()
    if reader.ReadFile(str(path)) != ocp["IFSelect_RetDone"]:
        raise CadLabelError(f"OpenCascade could not read STEP file: {path}")
    if reader.TransferRoots() == 0:
        raise CadLabelError(f"STEP file has no transferable roots: {path}")
    shape = reader.OneShape()
    if shape.IsNull():
        raise CadLabelError(f"STEP transfer produced a null shape: {path}")
    if not ocp["BRepCheck_Analyzer"](shape, True).IsValid():
        raise CadLabelError(f"STEP contains an invalid BRep: {path}")
    solids = ocp["TopExp_Explorer"](shape, ocp["TopAbs_SOLID"])
    if not solids.More():
        raise CadLabelError(f"STEP contains no solid: {path}")
    return shape


def _surface_class(surface_type: Any, ocp: dict[str, Any]) -> int:
    mapping = {
        ocp["GeomAbs_Plane"]: "plane",
        ocp["GeomAbs_Cylinder"]: "cylinder",
        ocp["GeomAbs_Cone"]: "cone",
        ocp["GeomAbs_Sphere"]: "sphere",
        ocp["GeomAbs_Torus"]: "torus",
        ocp["GeomAbs_BSplineSurface"]: "bspline",
    }
    return SURFACE_TO_ID[mapping.get(surface_type, "other")]


def _triangulation_node(triangulation: Any, index: int) -> Any:
    if hasattr(triangulation, "Node"):
        return triangulation.Node(index)
    return triangulation.InternalNodes().Value(index)


def _triangulation_triangle(triangulation: Any, index: int) -> tuple[int, int, int]:
    if hasattr(triangulation, "Triangle"):
        triangle = triangulation.Triangle(index)
    else:
        triangle = triangulation.InternalTriangles().Value(index)
    return tuple(int(value) for value in triangle.Get())


def shape_diagonal(shape: Any) -> float:
    """Return the OpenCascade bounding-box diagonal in model units."""
    ocp = _ocp()
    bounds = ocp["Bnd_Box"]()
    ocp["BRepBndLib"].Add_s(shape, bounds, True)
    minimum_x, minimum_y, minimum_z, maximum_x, maximum_y, maximum_z = bounds.Get()
    diagonal = float(
        np.linalg.norm(
            np.asarray(
                [
                    maximum_x - minimum_x,
                    maximum_y - minimum_y,
                    maximum_z - minimum_z,
                ]
            )
        )
    )
    if not np.isfinite(diagonal) or diagonal <= np.finfo(float).eps:
        raise CadLabelError("CAD shape has a degenerate bounding box")
    return diagonal


def tessellate_labeled_shape(
    shape: Any,
    *,
    linear_deflection: float | None = None,
    relative_deflection: float = 2e-3,
    angular_deflection: float = 0.2,
    merge_tolerance: float | None = None,
) -> TriangleGraphSample:
    """Return a triangle graph whose labels come from source BRep faces."""
    ocp = _ocp()
    diagonal = shape_diagonal(shape)
    if linear_deflection is None:
        if relative_deflection <= 0.0:
            raise ValueError("relative_deflection must be positive")
        linear_deflection = diagonal * relative_deflection
    if linear_deflection <= 0.0:
        raise ValueError("linear_deflection must be positive")
    if merge_tolerance is None:
        merge_tolerance = max(diagonal * 1e-9, 1e-10)
    mesher = ocp["BRepMesh_IncrementalMesh"](
        shape, float(linear_deflection), False, float(angular_deflection), True
    )
    mesher.Perform()
    if not mesher.IsDone():
        raise CadLabelError("OpenCascade tessellation failed")

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    surface_labels: list[int] = []
    instance_labels: list[int] = []
    vertex_map: dict[tuple[int, int, int], int] = {}
    missing_triangulations = 0

    explorer = ocp["TopExp_Explorer"](shape, ocp["TopAbs_FACE"])
    face_id = 0
    while explorer.More():
        face = ocp["TopoDS"].Face_s(explorer.Current())
        location = ocp["TopLoc_Location"]()
        triangulation = ocp["BRep_Tool"].Triangulation_s(face, location)
        if triangulation is None:
            explorer.Next()
            face_id += 1
            missing_triangulations += 1
            continue
        transform = location.Transformation()
        local_to_global: dict[int, int] = {}
        for node_index in range(1, triangulation.NbNodes() + 1):
            point = _triangulation_node(triangulation, node_index).Transformed(transform)
            coordinates = (float(point.X()), float(point.Y()), float(point.Z()))
            if not np.all(np.isfinite(coordinates)):
                raise CadLabelError("Tessellation produced non-finite coordinates")
            key = tuple(int(round(value / merge_tolerance)) for value in coordinates)
            global_index = vertex_map.get(key)
            if global_index is None:
                global_index = len(vertices)
                vertex_map[key] = global_index
                vertices.append(coordinates)
            local_to_global[node_index] = global_index

        surface_type = ocp["BRepAdaptor_Surface"](face, True).GetType()
        class_id = _surface_class(surface_type, ocp)
        reversed_face = face.Orientation() == ocp["TopAbs_REVERSED"]
        for triangle_index in range(1, triangulation.NbTriangles() + 1):
            node_ids = list(_triangulation_triangle(triangulation, triangle_index))
            if reversed_face:
                node_ids[1], node_ids[2] = node_ids[2], node_ids[1]
            vertex_ids = tuple(local_to_global[node_id] for node_id in node_ids)
            if len(set(vertex_ids)) < 3:
                continue
            faces.append(vertex_ids)
            surface_labels.append(class_id)
            instance_labels.append(face_id)
        explorer.Next()
        face_id += 1

    if missing_triangulations:
        raise CadLabelError(
            f"Tessellation omitted {missing_triangulations} BRep face(s); labels are incomplete"
        )
    if not faces:
        raise CadLabelError("Tessellation produced no triangles")
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.fix_normals(multibody=True)
    if not mesh.is_watertight:
        raise CadLabelError("Tessellated solid is not watertight")
    return build_sample(mesh, surface_labels, instance_labels)


def step_to_sample(path: str | Path, **kwargs: float | None) -> TriangleGraphSample:
    return tessellate_labeled_shape(read_step(path), **kwargs)
