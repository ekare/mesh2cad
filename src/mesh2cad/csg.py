"""Small constructive-solid-geometry tree used by topology reconstruction.

The tree keeps recognition separate from the CAD kernel. Large models can be
recognized in overlapping chunks and reduced into an explicit boolean graph
before OpenCascade evaluates the final exact solid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np


@dataclass(frozen=True)
class Box:
    minimum: np.ndarray
    dimensions: np.ndarray
    axes: np.ndarray | None = None


@dataclass(frozen=True)
class Cylinder:
    axis_point: np.ndarray
    axis: np.ndarray
    radius: float
    height: float


BooleanOperation = Literal["cut", "fuse", "common"]


@dataclass(frozen=True)
class Boolean:
    operation: BooleanOperation
    left: CsgNode
    right: CsgNode


CsgNode: TypeAlias = Box | Cylinder | Boolean


class CsgEvaluationError(RuntimeError):
    """Raised when OpenCascade cannot evaluate a CSG node."""


def evaluate(node: CsgNode, ocp: dict[str, Any]) -> Any:
    """Evaluate a kernel-independent CSG tree with an OpenCascade backend."""
    gp_pnt = ocp["gp_Pnt"]
    if isinstance(node, Box):
        if node.axes is None:
            builder = ocp["BRepPrimAPI_MakeBox"](
                gp_pnt(*node.minimum),
                float(node.dimensions[0]),
                float(node.dimensions[1]),
                float(node.dimensions[2]),
            )
        else:
            axes = np.asarray(node.axes, dtype=np.float64)
            if axes.shape != (3, 3) or not np.allclose(axes.T @ axes, np.eye(3), atol=1e-7):
                raise CsgEvaluationError("Oriented box axes must be a 3x3 orthonormal frame")
            if float(np.linalg.det(axes)) < 0.0:
                raise CsgEvaluationError("Oriented box axes must form a right-handed frame")
            frame = ocp["gp_Ax2"](
                gp_pnt(*node.minimum),
                ocp["gp_Dir"](*axes[:, 2]),
                ocp["gp_Dir"](*axes[:, 0]),
            )
            builder = ocp["BRepPrimAPI_MakeBox"](
                frame,
                float(node.dimensions[0]),
                float(node.dimensions[1]),
                float(node.dimensions[2]),
            )
        builder.Build()
        if not builder.IsDone():
            raise CsgEvaluationError("OpenCascade could not build a box primitive")
        return builder.Shape()

    if isinstance(node, Cylinder):
        builder = ocp["BRepPrimAPI_MakeCylinder"](
            ocp["gp_Ax2"](gp_pnt(*node.axis_point), ocp["gp_Dir"](*node.axis)),
            float(node.radius),
            float(node.height),
        )
        builder.Build()
        if not builder.IsDone():
            raise CsgEvaluationError("OpenCascade could not build a cylinder primitive")
        return builder.Shape()

    operations = {
        "cut": ocp["BRepAlgoAPI_Cut"],
        "fuse": ocp["BRepAlgoAPI_Fuse"],
        "common": ocp["BRepAlgoAPI_Common"],
    }
    builder = operations[node.operation](evaluate(node.left, ocp), evaluate(node.right, ocp))
    builder.Build()
    if not builder.IsDone():
        raise CsgEvaluationError(f"OpenCascade boolean {node.operation} failed")
    return builder.Shape()


def count_operations(node: CsgNode, operation: BooleanOperation) -> int:
    """Count boolean operations in a graph for diagnostics and tests."""
    if not isinstance(node, Boolean):
        return 0
    return (
        int(node.operation == operation)
        + count_operations(node.left, operation)
        + count_operations(node.right, operation)
    )
