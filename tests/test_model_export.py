from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mesh2cad.learning.features import FEATURE_NAMES, SURFACE_CLASSES  # noqa: E402
from mesh2cad.learning.model_export import export_inference_checkpoint  # noqa: E402


def _checkpoint(path, *, complete: bool) -> None:
    torch.save(
        {
            "model_state": {"weight": torch.arange(4)},
            "model": {"hidden_size": 16, "layers": 2},
            "surface_classes": SURFACE_CLASSES,
            "feature_names": FEATURE_NAMES,
            "amp": True,
            "partition": {"max_core_faces": 100, "halo_hops": 2},
            "augmentation": {"random_rotation": True},
            "history": [
                {
                    "epoch": 1,
                    "train_loss": 0.75,
                    "validation": {"surface_macro_iou": 0.5, "boundary_f1": 0.8},
                }
            ],
            "best_epoch": 1,
            "best_score": 1.3,
            "test": {"surface_macro_iou": 0.48},
            "training_complete": complete,
            "training_files": ["/home/private/training/a.npz", "/home/private/training/b.npz"],
            "validation_files": ["/home/private/validation/c.npz"],
            "test_files": ["/home/private/test/d.npz"],
            "dataset_roots": ["/home/private/dataset"],
            "optimizer_state": {"private": True},
            "torch_random_state": torch.get_rng_state(),
            "dataset_statistics": {"surface_counts": [1, 2, 3]},
            "seed": 17,
            "max_samples_per_source": 4,
        },
        path,
    )


def test_public_model_export_strips_paths_and_recovery_state(tmp_path) -> None:
    source = tmp_path / "training.pt"
    output = tmp_path / "public.pt"
    _checkpoint(source, complete=True)

    report = export_inference_checkpoint(
        source,
        output,
        provenance={"dataset": "AutoMate", "license": "CC0-1.0"},
    )
    exported = torch.load(output, map_location="cpu", weights_only=False)

    assert exported["checkpoint_kind"] == "mesh2cad-inference"
    assert exported["provenance"]["license"] == "CC0-1.0"
    assert exported["training_summary"]["training_samples"] == 2
    assert exported["quality"]["validation"]["surface_macro_iou"] == 0.5
    assert exported["training_summary"]["split_identity"]["counts"] == {
        "training": 2,
        "validation": 1,
        "test": 1,
        "selected": 4,
    }
    assert len(exported["training_summary"]["split_identity"]["sha256"]["selected"]) == 64
    assert "optimizer_state" not in exported
    assert "training_files" not in exported
    assert "dataset_roots" not in exported
    assert b"/home/private" not in output.read_bytes()
    assert report["checkpoint"] == "public.pt"
    assert len(report["sha256"]) == 64


def test_public_model_export_rejects_incomplete_training(tmp_path) -> None:
    source = tmp_path / "recovery.pt"
    _checkpoint(source, complete=False)

    with pytest.raises(ValueError, match="incomplete"):
        export_inference_checkpoint(source, tmp_path / "public.pt")


def test_public_model_export_validates_staged_provenance_identity(tmp_path) -> None:
    source = tmp_path / "training.pt"
    output = tmp_path / "public.pt"
    _checkpoint(source, complete=True)
    provenance = {
        "release_scope": "staged_training_subset",
        "training_subset": {
            "selection_seed": 17,
            "selection_limit_per_source": 4,
            "selected_samples": 4,
            "training_samples": 2,
            "validation_samples": 1,
            "test_samples": 1,
        },
    }

    export_inference_checkpoint(source, output, provenance=provenance)
    provenance["training_subset"]["test_samples"] = 2
    with pytest.raises(ValueError, match="test_samples"):
        export_inference_checkpoint(source, output, provenance=provenance)
