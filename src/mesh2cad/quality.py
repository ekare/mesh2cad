"""Scale-independent structural and sampling diagnostics for triangle meshes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class MeshQualityReport:
    source_vertices: int
    source_faces: int
    vertices: int
    faces: int
    connected_components: int
    watertight: bool
    winding_consistent: bool
    volume: bool
    boundary_edges: int
    non_manifold_edges: int
    degenerate_faces: int
    triangle_quality_p05: float
    normalized_edge_length_median: float
    normalized_edge_length_p95: float
    structural_health: float
    repair_recommended: bool
    coarse_sampling_hint: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_vertices": self.source_vertices,
            "source_faces": self.source_faces,
            "vertices": self.vertices,
            "faces": self.faces,
            "connected_components": self.connected_components,
            "watertight": self.watertight,
            "winding_consistent": self.winding_consistent,
            "volume": self.volume,
            "boundary_edges": self.boundary_edges,
            "non_manifold_edges": self.non_manifold_edges,
            "degenerate_faces": self.degenerate_faces,
            "triangle_quality_p05": self.triangle_quality_p05,
            "normalized_edge_length_median": self.normalized_edge_length_median,
            "normalized_edge_length_p95": self.normalized_edge_length_p95,
            "structural_health": self.structural_health,
            "repair_recommended": self.repair_recommended,
            "coarse_sampling_hint": self.coarse_sampling_hint,
        }


def _component_count(face_count: int, adjacency: np.ndarray) -> int:
    if face_count == 0:
        return 0
    if len(adjacency) == 0:
        return face_count
    rows = np.concatenate((adjacency[:, 0], adjacency[:, 1]))
    columns = np.concatenate((adjacency[:, 1], adjacency[:, 0]))
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
        shape=(face_count, face_count),
    )
    count, _labels = connected_components(graph, directed=False)
    return int(count)


def assess_mesh(
    mesh: trimesh.Trimesh,
    *,
    source_vertices: int | None = None,
    source_faces: int | None = None,
) -> MeshQualityReport:
    """Measure repair-relevant structure without claiming that coarse sampling is damage."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Mesh quality requires non-empty vertices and faces")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Mesh quality requires finite vertices")
    scale = float(np.linalg.norm(mesh.extents))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        raise ValueError("Mesh quality requires a non-degenerate bounding box")

    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    degenerate_threshold = max(scale * scale * 1e-14, np.finfo(float).eps)
    degenerate_faces = int(np.count_nonzero(areas <= degenerate_threshold))

    sorted_edges = np.sort(np.asarray(mesh.edges, dtype=np.int64), axis=1)
    unique_edges, edge_counts = np.unique(sorted_edges, axis=0, return_counts=True)
    boundary_edges = int(np.count_nonzero(edge_counts == 1))
    non_manifold_edges = int(np.count_nonzero(edge_counts > 2))
    edge_vectors = vertices[unique_edges[:, 1]] - vertices[unique_edges[:, 0]]
    normalized_lengths = np.linalg.norm(edge_vectors, axis=1) / scale

    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    triangle_edges = np.stack(
        (
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ),
        axis=1,
    )
    quality = 4.0 * np.sqrt(3.0) * areas / np.maximum(
        np.square(triangle_edges).sum(axis=1), np.finfo(float).eps
    )
    quality = np.clip(quality, 0.0, 1.0)

    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64).reshape(-1, 2)
    component_count = _component_count(len(faces), adjacency)
    watertight = bool(mesh.is_watertight)
    winding_consistent = bool(mesh.is_winding_consistent)
    is_volume = bool(mesh.is_volume)
    structural_health = (
        0.30 * float(watertight)
        + 0.20 * float(winding_consistent)
        + 0.15 * float(is_volume)
        + 0.15 * float(boundary_edges == 0)
        + 0.10 * float(non_manifold_edges == 0)
        + 0.05 * float(degenerate_faces == 0)
        + 0.05 * float(component_count == 1)
    )
    repair_recommended = bool(
        not watertight
        or not winding_consistent
        or boundary_edges > 0
        or non_manifold_edges > 0
        or degenerate_faces > 0
    )
    edge_p95 = float(np.percentile(normalized_lengths, 95.0))
    return MeshQualityReport(
        source_vertices=int(source_vertices if source_vertices is not None else len(vertices)),
        source_faces=int(source_faces if source_faces is not None else len(faces)),
        vertices=len(vertices),
        faces=len(faces),
        connected_components=component_count,
        watertight=watertight,
        winding_consistent=winding_consistent,
        volume=is_volume,
        boundary_edges=boundary_edges,
        non_manifold_edges=non_manifold_edges,
        degenerate_faces=degenerate_faces,
        triangle_quality_p05=float(np.percentile(quality, 5.0)),
        normalized_edge_length_median=float(np.median(normalized_lengths)),
        normalized_edge_length_p95=edge_p95,
        structural_health=float(structural_health),
        repair_recommended=repair_recommended,
        coarse_sampling_hint=edge_p95 >= 0.10,
    )
