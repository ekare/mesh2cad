"""Lightweight browser preview tessellation for reconstructed OpenCascade shapes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


class PreviewError(RuntimeError):
    """Raised when a reconstructed shape cannot be tessellated for display."""


@dataclass(frozen=True)
class ShapePreview:
    positions: npt.NDArray[np.float32]
    total_triangles: int
    shown_triangles: int
    deviations: npt.NDArray[np.float32] | None = None
    deviation_scale: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "positions": self.positions.reshape(-1).tolist(),
            "total_triangles": self.total_triangles,
            "shown_triangles": self.shown_triangles,
            "sampled": self.shown_triangles < self.total_triangles,
            "deviations": (
                None if self.deviations is None else self.deviations.reshape(-1).tolist()
            ),
            "deviation_scale": self.deviation_scale,
        }


def _ocp() -> dict[str, Any]:
    try:
        from OCP.Bnd import Bnd_Box
        from OCP.BRep import BRep_Tool
        from OCP.BRepBndLib import BRepBndLib
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopLoc import TopLoc_Location
        from OCP.TopoDS import TopoDS
    except ImportError as exc:  # pragma: no cover - exercised by minimal installs
        raise PreviewError("STEP preview requires cadquery-ocp") from exc
    return {
        "Bnd_Box": Bnd_Box,
        "BRep_Tool": BRep_Tool,
        "BRepBndLib": BRepBndLib,
        "BRepMesh_IncrementalMesh": BRepMesh_IncrementalMesh,
        "TopAbs_FACE": TopAbs_FACE,
        "TopAbs_REVERSED": TopAbs_REVERSED,
        "TopExp_Explorer": TopExp_Explorer,
        "TopLoc_Location": TopLoc_Location,
        "TopoDS": TopoDS,
    }


def _node(triangulation: Any, index: int) -> Any:
    if hasattr(triangulation, "Node"):
        return triangulation.Node(index)
    return triangulation.InternalNodes().Value(index)


def _triangle(triangulation: Any, index: int) -> list[int]:
    if hasattr(triangulation, "Triangle"):
        triangle = triangulation.Triangle(index)
    else:
        triangle = triangulation.InternalTriangles().Value(index)
    return [int(value) for value in triangle.Get()]


def _shape_diagonal(shape: Any, ocp: dict[str, Any]) -> float:
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
                ],
                dtype=np.float64,
            )
        )
    )
    if not math.isfinite(diagonal) or diagonal <= np.finfo(float).eps:
        raise PreviewError("The reconstructed shape has a degenerate bounding box")
    return diagonal


def tessellate_shape(
    shape: Any,
    *,
    relative_deflection: float = 2e-3,
    angular_deflection: float = 0.2,
    max_triangles: int | None = 75_000,
) -> ShapePreview:
    """Tessellate a BRep for display without limiting the actual reconstruction."""
    if max_triangles is not None and max_triangles <= 0:
        raise ValueError("max_triangles must be positive")
    if relative_deflection <= 0.0:
        raise ValueError("relative_deflection must be positive")

    ocp = _ocp()
    diagonal = _shape_diagonal(shape, ocp)
    mesher = ocp["BRepMesh_IncrementalMesh"](
        shape,
        diagonal * relative_deflection,
        False,
        float(angular_deflection),
        True,
    )
    mesher.Perform()
    if not mesher.IsDone():
        raise PreviewError("OpenCascade preview tessellation failed")

    triangulations: list[tuple[Any, Any, bool]] = []
    total_triangles = 0
    explorer = ocp["TopExp_Explorer"](shape, ocp["TopAbs_FACE"])
    while explorer.More():
        face = ocp["TopoDS"].Face_s(explorer.Current())
        location = ocp["TopLoc_Location"]()
        triangulation = ocp["BRep_Tool"].Triangulation_s(face, location)
        if triangulation is None:
            raise PreviewError("A reconstructed face has no preview triangulation")
        triangulations.append(
            (triangulation, location.Transformation(), face.Orientation() == ocp["TopAbs_REVERSED"])
        )
        total_triangles += int(triangulation.NbTriangles())
        explorer.Next()

    if total_triangles == 0:
        raise PreviewError("The reconstructed shape produced no preview triangles")

    stride = 1 if max_triangles is None else max(1, math.ceil(total_triangles / max_triangles))
    positions: list[tuple[float, float, float]] = []
    global_triangle = 0
    for triangulation, transform, reversed_face in triangulations:
        for triangle_index in range(1, triangulation.NbTriangles() + 1):
            selected = global_triangle % stride == 0
            global_triangle += 1
            if not selected:
                continue
            node_ids = _triangle(triangulation, triangle_index)
            if reversed_face:
                node_ids[1], node_ids[2] = node_ids[2], node_ids[1]
            triangle_positions = []
            for node_id in node_ids:
                point = _node(triangulation, node_id).Transformed(transform)
                coordinates = (float(point.X()), float(point.Y()), float(point.Z()))
                if not np.all(np.isfinite(coordinates)):
                    raise PreviewError("Preview tessellation produced non-finite coordinates")
                triangle_positions.append(coordinates)
            if len(set(triangle_positions)) == 3:
                positions.extend(triangle_positions)

    position_array = np.asarray(positions, dtype=np.float32).reshape(-1, 3, 3)
    if len(position_array) == 0:
        raise PreviewError("Preview tessellation produced only degenerate triangles")
    return ShapePreview(
        positions=position_array,
        total_triangles=total_triangles,
        shown_triangles=len(position_array),
    )
