"""Triangle graph features and portable NPZ training samples."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import trimesh

SURFACE_CLASSES = (
    "plane",
    "cylinder",
    "cone",
    "sphere",
    "torus",
    "bspline",
    "other",
)
SURFACE_TO_ID = {name: index for index, name in enumerate(SURFACE_CLASSES)}
FEATURE_NAMES = (
    "centroid_x",
    "centroid_y",
    "centroid_z",
    "normal_x",
    "normal_y",
    "normal_z",
    "log_relative_area",
    "mean_dihedral",
    "max_dihedral",
    "edge_short",
    "edge_middle",
    "edge_long",
)


@dataclass(frozen=True)
class TriangleGraphSample:
    vertices: npt.NDArray[np.float32]
    faces: npt.NDArray[np.int64]
    features: npt.NDArray[np.float32]
    adjacency: npt.NDArray[np.int64]
    surface_labels: npt.NDArray[np.int64]
    face_instance_labels: npt.NDArray[np.int64]
    boundary_labels: npt.NDArray[np.float32]


def triangle_features(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Build scale-normalized node features and undirected face adjacency."""
    face_count = len(mesh.faces)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64).copy()
    scale = float(np.linalg.norm(mesh.extents))
    if scale <= np.finfo(float).eps:
        raise ValueError("Cannot featurize a degenerate mesh")

    centroids = (np.asarray(mesh.triangles_center) - np.asarray(mesh.centroid)) / scale
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    log_area = np.log(np.maximum(areas / np.mean(areas), 1e-12))[:, None]

    mean_dihedral = np.zeros(face_count, dtype=np.float64)
    max_dihedral = np.zeros(face_count, dtype=np.float64)
    counts = np.zeros(face_count, dtype=np.float64)
    if len(adjacency):
        dots = np.einsum("ij,ij->i", normals[adjacency[:, 0]], normals[adjacency[:, 1]])
        angles = np.arccos(np.clip(dots, -1.0, 1.0))
        for column in (0, 1):
            indices = adjacency[:, column]
            np.add.at(mean_dihedral, indices, angles)
            np.add.at(counts, indices, 1.0)
            np.maximum.at(max_dihedral, indices, angles)
    mean_dihedral /= np.maximum(counts, 1.0)

    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    edge_lengths = np.stack(
        (
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ),
        axis=1,
    )
    edge_lengths.sort(axis=1)
    edge_lengths /= scale
    features = np.column_stack(
        (
            centroids,
            normals,
            log_area,
            mean_dihedral,
            max_dihedral,
            edge_lengths,
        )
    )
    return features.astype(np.float32), adjacency


def build_sample(
    mesh: trimesh.Trimesh,
    surface_labels: npt.ArrayLike,
    face_instance_labels: npt.ArrayLike,
) -> TriangleGraphSample:
    surface_array = np.asarray(surface_labels, dtype=np.int64)
    instance_array = np.asarray(face_instance_labels, dtype=np.int64)
    if len(surface_array) != len(mesh.faces) or len(instance_array) != len(mesh.faces):
        raise ValueError("Every triangle must have a surface and instance label")
    features, adjacency = triangle_features(mesh)
    boundary = (instance_array[adjacency[:, 0]] != instance_array[adjacency[:, 1]]).astype(
        np.float32
    )
    return TriangleGraphSample(
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        features=features,
        adjacency=adjacency,
        surface_labels=surface_array,
        face_instance_labels=instance_array,
        boundary_labels=boundary,
    )


def save_sample(sample: TriangleGraphSample, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            vertices=sample.vertices,
            faces=sample.faces,
            features=sample.features,
            adjacency=sample.adjacency,
            surface_labels=sample.surface_labels,
            face_instance_labels=sample.face_instance_labels,
            boundary_labels=sample.boundary_labels,
            surface_classes=np.asarray(SURFACE_CLASSES),
            feature_names=np.asarray(FEATURE_NAMES),
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    return destination


def load_sample(path: str | Path) -> TriangleGraphSample:
    with np.load(path, allow_pickle=False) as data:
        return TriangleGraphSample(
            vertices=data["vertices"],
            faces=data["faces"],
            features=data["features"],
            adjacency=data["adjacency"],
            surface_labels=data["surface_labels"],
            face_instance_labels=data["face_instance_labels"],
            boundary_labels=data["boundary_labels"],
        )
