"""Optional mesh and STEP simplification helpers used by the local web app."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import trimesh

from mesh2cad.export import export_step
from mesh2cad.topology import _count_subshapes, _load_ocp, simplify_shape


class SimplificationError(RuntimeError):
    """Raised when a requested simplification cannot be completed safely."""


def simplify_mesh_file(
    input_path: str | Path,
    output_path: str | Path,
    target_faces: int,
) -> dict[str, int]:
    source = Path(input_path)
    destination = Path(output_path)
    loaded = trimesh.load(source, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise SimplificationError("Input does not contain a single triangle mesh")
    before = len(loaded.faces)
    target = max(4, min(int(target_faces), before))
    if target == before:
        shutil.copyfile(source, destination)
    else:
        try:
            simplified = loaded.simplify_quadric_decimation(face_count=target)
        except (ImportError, ModuleNotFoundError) as exc:
            raise SimplificationError(
                "Mesh simplification needs the optional fast-simplification package"
            ) from exc
        if len(simplified.faces) == 0 or not simplified.is_watertight and loaded.is_watertight:
            raise SimplificationError("Simplification produced an unusable mesh")
        destination.parent.mkdir(parents=True, exist_ok=True)
        simplified.export(destination, file_type="stl")
    return {"input_faces": before, "output_faces": target}


def _step_shape(path: Path, ocp: dict[str, Any]) -> Any:
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone or reader.TransferRoots() == 0:
        raise SimplificationError("OpenCascade could not read the STEP file")
    shape = reader.OneShape()
    if shape.IsNull():
        raise SimplificationError("STEP file contains no usable shape")
    return shape


def simplify_step_file(input_path: str | Path, output_path: str | Path) -> dict[str, int]:
    ocp = _load_ocp()
    source = Path(input_path)
    shape = _step_shape(source, ocp)
    before = _count_subshapes(shape, ocp["TopAbs_FACE"], ocp["TopExp_Explorer"])
    simplified = simplify_shape(shape, ocp, 1.0)
    after = _count_subshapes(simplified, ocp["TopAbs_FACE"], ocp["TopExp_Explorer"])
    export_step(simplified, output_path)
    return {"input_faces": before, "output_faces": after}
