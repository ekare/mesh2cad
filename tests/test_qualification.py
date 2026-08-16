from __future__ import annotations

import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

from mesh2cad.learning.qualification import qualify_model  # noqa: E402


def _artifacts(tmp_path, *, worsened_fraction: float = 0.0):
    checkpoint = tmp_path / "public.pt"
    torch.save(
        {
            "checkpoint_kind": "mesh2cad-inference",
            "training_summary": {"training_complete": True},
            "quality": {
                "test": {
                    "supported_surface_macro_iou": 0.6,
                    "boundary_f1": 0.96,
                }
            },
        },
        checkpoint,
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    reconstruction = tmp_path / "reconstruction.json"
    repair = tmp_path / "repair.json"
    reconstruction.write_text(
        json.dumps(
            {
                "checkpoint_sha256": digest,
                "summary": {
                    "cases": 24,
                    "modes": {
                        "neural": {
                            "success_rate": 1.0,
                            "analytic_csg_rate": 1.0,
                            "mean_geometric_fidelity": 0.96,
                        }
                    },
                    "neural_not_worse_than_deterministic": {"fraction": 1.0},
                },
            }
        ),
        encoding="utf-8",
    )
    repair.write_text(
        json.dumps(
            {
                "checkpoint_sha256": digest,
                "summary": {
                    "overall": {
                        "cases": 24,
                        "watertight_outputs": 24,
                        "worsened_fraction": worsened_fraction,
                        "acceptance_rate": 0.75,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return checkpoint, reconstruction, repair


def test_model_qualification_passes_all_explicit_gates(tmp_path) -> None:
    checkpoint, reconstruction, repair = _artifacts(tmp_path)

    report = qualify_model(checkpoint, reconstruction, repair)

    assert report["passed"] is True
    assert report["failed_gates"] == []
    assert all(gate["passed"] for gate in report["gates"].values())


def test_model_qualification_reports_failed_repair_gate(tmp_path) -> None:
    checkpoint, reconstruction, repair = _artifacts(tmp_path, worsened_fraction=0.1)

    report = qualify_model(checkpoint, reconstruction, repair)

    assert report["passed"] is False
    assert report["failed_gates"] == ["repair_worsened_fraction"]


def test_model_qualification_rejects_mismatched_checkpoint(tmp_path) -> None:
    checkpoint, reconstruction, repair = _artifacts(tmp_path)
    payload = json.loads(reconstruction.read_text(encoding="utf-8"))
    payload["checkpoint_sha256"] = "0" * 64
    reconstruction.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint hash"):
        qualify_model(checkpoint, reconstruction, repair)
