import numpy as np
import trimesh

from mesh2cad.learning.segment import segmentation_from_prediction
from mesh2cad.mesh import prepare_mesh
from mesh2cad.segmentation import SegmentationConfig


def test_prediction_instances_are_refit_as_verified_planes():
    data = prepare_mesh(trimesh.creation.box(extents=(2.0, 3.0, 4.0)))
    _, instances = np.unique(
        np.round(data.mesh.face_normals, decimals=6), axis=0, return_inverse=True
    )
    prediction = {
        "surface_labels": np.zeros(len(data.mesh.faces), dtype=np.int64),
        "instance_labels": instances,
    }

    result = segmentation_from_prediction(
        data,
        prediction,
        config=SegmentationConfig(min_plane_faces=2, min_plane_area_fraction=0.0),
    )

    assert len(result.plane_regions) == 6
    assert len(result.unassigned_faces) == 0
    assert max(region.fit.max_error for region in result.plane_regions) < 1e-12


def test_prediction_with_bad_geometry_is_rejected():
    data = prepare_mesh(trimesh.creation.box())
    prediction = {
        "surface_labels": np.zeros(len(data.mesh.faces), dtype=np.int64),
        "instance_labels": np.zeros(len(data.mesh.faces), dtype=np.int64),
    }

    result = segmentation_from_prediction(
        data,
        prediction,
        config=SegmentationConfig(min_plane_faces=2, min_plane_area_fraction=0.0),
    )

    assert not result.regions
    assert len(result.unassigned_faces) == len(data.mesh.faces)


def test_predicted_instances_recover_supported_geometry_from_wrong_semantics():
    mesh = trimesh.creation.box(extents=(2.0, 3.0, 4.0))
    vertices, faces = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
    vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    data = prepare_mesh(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))
    _, instances = np.unique(
        np.round(data.mesh.face_normals, decimals=6), axis=0, return_inverse=True
    )
    prediction = {
        "surface_labels": np.full(len(data.mesh.faces), 1, dtype=np.int64),
        "instance_labels": instances,
    }

    result = segmentation_from_prediction(
        data,
        prediction,
        config=SegmentationConfig(min_plane_faces=2, min_plane_area_fraction=0.0),
    )

    assert len(result.plane_regions) == 6
    assert len(result.cylinder_regions) == 0
    assert len(result.unassigned_faces) == 0
