import json

from mesh2cad.mesh import load_mesh
from mesh2cad.pipeline import analyze, convert
from mesh2cad.segmentation import segment_surfaces
from mesh2cad.synthetic import write_cube_with_hole


def test_end_to_end_writes_valid_step_and_metrics(tmp_path):
    stl_path = write_cube_with_hole(tmp_path / "input.stl", sections_per_side=12)
    step_path = tmp_path / "output.step"
    report_path = tmp_path / "metrics.json"

    result = convert(stl_path, step_path, report_path=report_path)

    assert step_path.is_file()
    assert step_path.stat().st_size > 500
    assert "ISO-10303-21" in step_path.read_text(encoding="utf-8", errors="ignore")[:100]
    assert result.topology is not None
    assert result.topology.valid
    assert result.topology.solid_count == 1
    assert result.topology.cylinders_cut == 1
    assert result.metrics.brep_vertex_rmse is not None
    assert result.metrics.brep_vertex_rmse < 1e-7
    assert result.metrics.geometric_fidelity is not None
    # The analytic cylinder is intentionally compared to the STL's flat chords. At the strict
    # adaptive tolerance it is highly faithful but not falsely reported as identical.
    assert 0.8 < result.metrics.geometric_fidelity < 1.0
    assert result.metrics.bidirectional_p95_error is not None
    assert result.metrics.volume_relative_error is not None

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["segmentation"]["plane_count"] == 6
    assert report["segmentation"]["cylinder_count"] == 1
    assert report["metrics"]["assigned_area_fraction"] == 1.0
    assert report["metrics"]["geometric_fidelity"] > 0.8
    assert report["cad_quality"]["representation"] == "analytic-csg"


def test_explicit_neural_mode_bypasses_deterministic_selection(tmp_path, monkeypatch):
    stl_path = write_cube_with_hole(tmp_path / "input.stl", sections_per_side=8)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test checkpoint placeholder")
    expected = segment_surfaces(load_mesh(stl_path))
    calls = []

    def fake_segment(data, checkpoint_path, **kwargs):
        calls.append((data, checkpoint_path, kwargs))
        return expected

    monkeypatch.setattr("mesh2cad.learning.segment.segment_with_model", fake_segment)

    result = analyze(
        stl_path,
        checkpoint_path=checkpoint,
        segmentation_mode="neural",
        device_name="cpu",
    )

    assert result.segmentation_engine == "neural"
    assert result.segmentation is expected
    assert len(calls) == 1
    assert calls[0][1] == checkpoint


def test_neural_mode_requires_checkpoint(tmp_path):
    stl_path = write_cube_with_hole(tmp_path / "input.stl", sections_per_side=8)

    try:
        analyze(stl_path, segmentation_mode="neural")
    except ValueError as exc:
        assert "requires a checkpoint" in str(exc)
    else:
        raise AssertionError("Neural mode unexpectedly ran without a checkpoint")
