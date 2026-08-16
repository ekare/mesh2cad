from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from mesh2cad.learned_repair import NeuralAnalyticRepairer, _fidelity_metrics
from mesh2cad.quality import assess_mesh
from mesh2cad.repair import ConservativeRepairer, RepairError, repair_file


def test_quality_report_detects_open_boundary_and_coarse_sampling() -> None:
    closed = trimesh.creation.box(extents=(2.0, 3.0, 4.0))
    closed_report = assess_mesh(closed)
    open_mesh = closed.copy()
    open_mesh.update_faces(np.arange(len(open_mesh.faces)) != 0)
    open_mesh.remove_unreferenced_vertices()
    open_report = assess_mesh(open_mesh)

    assert closed_report.watertight
    assert closed_report.boundary_edges == 0
    assert closed_report.repair_recommended is False
    assert closed_report.coarse_sampling_hint is True
    assert not open_report.watertight
    assert open_report.boundary_edges == 3
    assert open_report.repair_recommended is True
    assert open_report.structural_health < closed_report.structural_health


def test_conservative_repair_removes_duplicate_face_without_inventing_geometry() -> None:
    box = trimesh.creation.box()
    damaged = trimesh.Trimesh(
        vertices=box.vertices.copy(),
        faces=np.vstack((box.faces, box.faces[0])),
        process=False,
    )

    result = ConservativeRepairer().repair(damaged)

    assert "remove_duplicate_faces" in result.actions
    assert len(result.mesh.faces) == len(box.faces)
    assert result.after.watertight
    assert result.after.repair_recommended is False


def test_repair_file_writes_stl_and_auditable_report(tmp_path) -> None:
    source = tmp_path / "source.stl"
    output = tmp_path / "repaired.stl"
    trimesh.creation.box().export(source)

    result = repair_file(source, output)

    assert output.is_file()
    report = json.loads(output.with_suffix(".repair.json").read_text(encoding="utf-8"))
    assert report["repairer"] == "conservative"
    assert report["after"]["watertight"] is True
    assert report["after"]["repair_recommended"] is False
    assert result.after.structural_health == 1.0


def test_neural_repair_requires_an_explicit_checkpoint() -> None:
    with pytest.raises(RepairError, match="requires --checkpoint"):
        NeuralAnalyticRepairer(None).repair(trimesh.creation.box())


def test_neural_analytic_repair_accepts_only_gated_candidate(tmp_path, monkeypatch) -> None:
    import mesh2cad.learned_repair as learned_repair

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"inference checkpoint")
    source = trimesh.creation.box()
    candidate = trimesh.creation.box()
    proposal = SimpleNamespace(
        metrics=SimpleNamespace(assigned_area_fraction=1.0, assigned_face_fraction=1.0),
        segmentation=object(),
    )
    monkeypatch.setattr(learned_repair, "analyze", lambda *args, **kwargs: proposal)
    monkeypatch.setattr(learned_repair, "recognize_csg", lambda *args, **kwargs: object())
    monkeypatch.setattr(learned_repair, "_load_ocp", lambda: {})
    monkeypatch.setattr(learned_repair, "evaluate", lambda *args, **kwargs: object())
    monkeypatch.setattr(learned_repair, "_validate_shape", lambda *args, **kwargs: None)
    monkeypatch.setattr(learned_repair, "_shape_mesh", lambda *args, **kwargs: candidate)
    monkeypatch.setattr(
        learned_repair,
        "_fidelity_metrics",
        lambda *args, **kwargs: {
            "symmetric_rms_relative": 0.0,
            "symmetric_p95_relative": 0.0,
            "symmetric_p99_relative": 0.0,
            "symmetric_max_relative": 0.0,
            "relative_area_error": 0.0,
            "relative_volume_error": 0.0,
            "sample_count_each_direction": 64.0,
        },
    )

    result = NeuralAnalyticRepairer(checkpoint, sample_count=64).repair(source)

    assert result.abstained is False
    assert result.confidence == 1.0
    assert "learned_surface_segmentation" in result.actions
    assert result.metrics is not None
    assert result.metrics["output_triangles"] == len(candidate.faces)
    assert result.checkpoint_sha256 is not None
    assert len(result.checkpoint_sha256) == 64


def test_repair_metrics_do_not_claim_volume_fidelity_for_open_source() -> None:
    candidate = trimesh.creation.box()
    source = candidate.copy()
    source.update_faces(np.arange(len(source.faces)) != 0)
    source.remove_unreferenced_vertices()

    metrics = _fidelity_metrics(source, candidate, sample_count=32)

    assert source.is_watertight is False
    assert metrics["relative_volume_error"] is None
