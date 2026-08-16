"""STEP export through OpenCascade."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mesh2cad.topology import BackendUnavailable, TopologyError


def export_step(shape: Any, path: str | Path) -> Path:
    try:
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    except ImportError as exc:
        raise BackendUnavailable("OpenCascade STEP writer is unavailable") from exc

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = STEPControl_Writer()
    transfer_status = writer.Transfer(shape, STEPControl_AsIs)
    if transfer_status != IFSelect_RetDone:
        raise TopologyError("OpenCascade failed to transfer the BRep to the STEP writer")
    write_status = writer.Write(str(destination))
    if write_status != IFSelect_RetDone:
        raise TopologyError(f"OpenCascade failed to write STEP file: {destination}")
    return destination
