from __future__ import annotations

import json

import pytest

from mesh2cad.dataset.automate import AUTOMATE_ARCHIVE_SHA256
from mesh2cad.dataset.provenance import build_automate_provenance
from mesh2cad.learning.split_export import export_split_manifest


def _write_manifests(tmp_path, *, status: str = "complete"):
    ingest = tmp_path / "ingest.json"
    preparation = tmp_path / "preparation.json"
    ingest.write_text(
        json.dumps(
            {
                "dataset": "AutoMate",
                "license": "CC0-1.0",
                "archive": {
                    "path": "/private/cache/step.zip",
                    "verification_performed": False,
                },
                "destination": "/private/cache/automate",
                "step_files": 447_442,
                "uncompressed_bytes": 74_088_040_746,
            }
        ),
        encoding="utf-8",
    )
    preparation.write_text(
        json.dumps(
            {
                "status": status,
                "processed": 447_442 if status == "complete" else 100,
                "completed": 447_440 if status == "complete" else 98,
                "failed": 2,
                "remaining": 0 if status == "complete" else 447_342,
                "configuration": {
                    "relative_deflection": 0.001,
                    "angular_deflection": 0.1,
                    "max_triangles": None,
                    "workers": 5,
                    "min_free_gib": 100,
                },
                "failures": [
                    {
                        "path": "/private/cache/a.step",
                        "error": "CadLabelError: STEP contains an invalid BRep",
                    },
                    {
                        "path": "/private/cache/b.step",
                        "error": "CadLabelError: Tessellated solid is not watertight",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return ingest, preparation


def test_public_automate_provenance_strips_paths_and_aggregates_failures(tmp_path) -> None:
    ingest, preparation = _write_manifests(tmp_path)

    result = build_automate_provenance(ingest, preparation)
    serialized = json.dumps(result)

    assert result["published_archive"]["sha256"] == AUTOMATE_ARCHIVE_SHA256
    assert result["local_acquisition"]["archive_verification_performed"] is False
    assert result["preparation"]["accepted"] == 447_440
    assert result["preparation"]["failure_categories"] == {
        "invalid_brep": 1,
        "not_watertight": 1,
    }
    assert result["preparation"]["configuration"]["max_triangles"] is None
    assert result["preparation"]["uncapped_triangles"] is True
    assert "/private/" not in serialized
    assert "a.step" not in serialized
    assert "workers" not in result["preparation"]["configuration"]


def test_public_provenance_refuses_incomplete_preparation_by_default(tmp_path) -> None:
    ingest, preparation = _write_manifests(tmp_path, status="running")

    with pytest.raises(ValueError, match="incomplete"):
        build_automate_provenance(ingest, preparation)

    result = build_automate_provenance(ingest, preparation, allow_partial=True)
    assert result["preparation"]["status"] == "running"
    assert result["preparation"]["remaining"] == 447_342
    assert result["release_scope"] == "internal_partial_audit"


def test_completed_stage_checkpoint_makes_partial_subset_provenance_publishable(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    ingest, preparation = _write_manifests(tmp_path, status="running")
    checkpoint = tmp_path / "stage.pt"
    torch.save(
        {
            "training_complete": True,
            "history": [{"epoch": 1}],
            "best_epoch": 1,
            "seed": 17,
            "max_samples_per_source": 4,
            "dataset_roots": ["/private/prepared"],
            "training_files": ["/private/prepared/a.npz", "/private/prepared/b.npz"],
            "validation_files": ["/private/prepared/c.npz"],
            "test_files": ["/private/prepared/d.npz"],
        },
        checkpoint,
    )
    split_manifest = tmp_path / "automate-v0.1-split.json"
    split_report = export_split_manifest(checkpoint, split_manifest)

    result = build_automate_provenance(
        ingest,
        preparation,
        training_checkpoint=checkpoint,
        split_manifest=split_manifest,
    )
    serialized = json.dumps(result)

    assert result["release_scope"] == "staged_training_subset"
    assert result["preparation"]["status"] == "running"
    assert result["preparation"]["snapshot_complete_for_training_subset"] is True
    assert result["training_subset"]["selected_samples"] == 4
    assert result["training_subset"]["training_samples"] == 2
    assert len(result["training_subset"]["identifier_sha256"]["selected"]) == 64
    assert result["training_subset"]["public_split_manifest"] == {
        "file": "automate-v0.1-split.json",
        "sha256": split_report["sha256"],
        "schema_version": 1,
    }
    assert "/private/" not in serialized
    assert "a.npz" not in serialized


def test_stage_provenance_rejects_incomplete_checkpoint(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    ingest, preparation = _write_manifests(tmp_path, status="running")
    checkpoint = tmp_path / "stage.pt"
    torch.save({"training_complete": False}, checkpoint)

    with pytest.raises(ValueError, match="completed training"):
        build_automate_provenance(
            ingest,
            preparation,
            training_checkpoint=checkpoint,
        )
