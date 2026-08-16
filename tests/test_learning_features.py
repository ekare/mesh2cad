import numpy as np
import pytest

from mesh2cad.learning import features
from mesh2cad.learning.features import FEATURE_NAMES, SURFACE_TO_ID, build_sample, save_sample
from mesh2cad.mesh import prepare_mesh
from mesh2cad.segmentation import segment_surfaces
from mesh2cad.synthetic import cube_with_hole


def test_triangle_graph_features_and_boundaries_align():
    data = prepare_mesh(cube_with_hole(sections_per_side=8))
    segmentation = segment_surfaces(data)
    instances = segmentation.face_region_ids
    semantics = np.full(len(data.mesh.faces), SURFACE_TO_ID["other"], dtype=np.int64)
    for region in segmentation.regions:
        semantics[region.face_indices] = SURFACE_TO_ID[region.kind]

    sample = build_sample(data.mesh, semantics, instances)

    assert sample.features.shape == (len(data.mesh.faces), len(FEATURE_NAMES))
    assert np.isfinite(sample.features).all()
    assert sample.adjacency.shape[1] == 2
    assert sample.boundary_labels.shape == (len(sample.adjacency),)
    assert np.count_nonzero(sample.boundary_labels) > 0


def test_sample_save_never_exposes_an_incomplete_target(tmp_path, monkeypatch):
    data = prepare_mesh(cube_with_hole(sections_per_side=8))
    segmentation = segment_surfaces(data)
    sample = build_sample(data.mesh, segmentation.face_region_ids, segmentation.face_region_ids)
    destination = tmp_path / "sample.npz"

    def interrupted_write(stream, **_arrays):
        stream.write(b"incomplete")
        raise OSError("simulated interruption")

    monkeypatch.setattr(features.np, "savez_compressed", interrupted_write)

    with pytest.raises(OSError, match="simulated interruption"):
        save_sample(sample, destination)

    assert not destination.exists()
    assert destination.with_suffix(".npz.part").read_bytes() == b"incomplete"
