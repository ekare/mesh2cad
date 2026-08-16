from mesh2cad.learning.cad_labels import step_to_sample
from mesh2cad.learning.features import SURFACE_TO_ID
from mesh2cad.pipeline import convert
from mesh2cad.synthetic import write_cube_with_hole


def test_step_tessellation_preserves_surface_labels(tmp_path):
    source = write_cube_with_hole(tmp_path / "input.stl", sections_per_side=8)
    step = tmp_path / "part.step"
    convert(source, step)

    sample = step_to_sample(step, linear_deflection=0.2)

    labels = set(sample.surface_labels.tolist())
    assert SURFACE_TO_ID["plane"] in labels
    assert SURFACE_TO_ID["cylinder"] in labels
    assert len(set(sample.face_instance_labels.tolist())) == 7
    assert sample.boundary_labels.sum() > 0
