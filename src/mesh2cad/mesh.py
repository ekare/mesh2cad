"""Mesh loading, cleaning, and adjacency extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from mesh2cad.quality import MeshQualityReport, assess_mesh
from mesh2cad.types import FloatArray, IntArray


class MeshError(ValueError):
    """Raised when an input cannot be used as a triangle mesh."""


@dataclass(frozen=True)
class MeshData:
    mesh: trimesh.Trimesh
    adjacency: IntArray
    neighbours: tuple[IntArray, ...]
    scale: float
    quality: MeshQualityReport

    @property
    def vertices(self) -> FloatArray:
        return np.asarray(self.mesh.vertices, dtype=np.float64)

    @property
    def faces(self) -> IntArray:
        return np.asarray(self.mesh.faces, dtype=np.int64)


def _as_single_mesh(loaded: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene) and loaded.geometry:
        return trimesh.util.concatenate(tuple(loaded.geometry.values()))
    raise MeshError("Input does not contain a triangle mesh")


def prepare_mesh(mesh: trimesh.Trimesh) -> MeshData:
    """Return a validated copy plus face adjacency and per-face neighbours."""
    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        raise MeshError("Mesh is empty")

    clean = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=True,
        validate=True,
    )
    clean.remove_unreferenced_vertices()
    clean.fix_normals(multibody=True)

    if not np.isfinite(clean.vertices).all():
        raise MeshError("Mesh contains non-finite coordinates")
    if len(clean.faces) == 0:
        raise MeshError("Mesh has no non-degenerate faces after cleaning")

    adjacency = np.asarray(clean.face_adjacency, dtype=np.int64)
    neighbour_lists: list[list[int]] = [[] for _ in range(len(clean.faces))]
    for left, right in adjacency:
        neighbour_lists[int(left)].append(int(right))
        neighbour_lists[int(right)].append(int(left))
    neighbours = tuple(np.asarray(items, dtype=np.int64) for items in neighbour_lists)

    extent = np.asarray(clean.extents, dtype=np.float64)
    scale = float(np.linalg.norm(extent))
    if scale <= np.finfo(float).eps:
        raise MeshError("Mesh bounding box is degenerate")
    quality = assess_mesh(
        clean,
        source_vertices=len(mesh.vertices),
        source_faces=len(mesh.faces),
    )
    return MeshData(
        mesh=clean,
        adjacency=adjacency,
        neighbours=neighbours,
        scale=scale,
        quality=quality,
    )


def load_mesh(path: str | Path) -> MeshData:
    """Load an STL or another trimesh-supported surface format."""
    source = Path(path)
    if not source.is_file():
        raise MeshError(f"Input file does not exist: {source}")
    loaded = trimesh.load(source, force="mesh", process=False)
    return prepare_mesh(_as_single_mesh(loaded))
