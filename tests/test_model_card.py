from __future__ import annotations

import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

from mesh2cad.learning.model_card import build_model_card  # noqa: E402


def test_model_card_is_path_free_and_binds_benchmark_hashes(tmp_path) -> None:
    checkpoint_path = tmp_path / "public.pt"
    checkpoint = {
        "checkpoint_kind": "mesh2cad-inference",
        "mesh2cad_version": "0.1.0",
        "model_license": "MIT",
        "model": {"hidden_size": 128, "layers": 5},
        "surface_classes": ("plane", "cylinder"),
        "training_summary": {
            "completed_epochs": 12,
            "training_samples": 12_800,
            "validation_samples": 1_600,
            "test_samples": 1_600,
        },
        "quality": {
            "best_epoch": 8,
            "validation": {
                "surface_macro_iou": 0.6,
                "supported_surface_macro_iou": 0.7,
                "surface_accuracy": 0.8,
                "boundary_f1": 0.9,
            },
            "test": {
                "surface_macro_iou": 0.59,
                "supported_surface_macro_iou": 0.69,
                "surface_accuracy": 0.79,
                "boundary_f1": 0.89,
            },
        },
        "provenance": {
            "dataset": {
                "name": "AutoMate",
                "source": "https://doi.org/10.5061/dryad.2547d7wvw",
                "license": "CC0-1.0",
            },
            "training_subset": {"selected_samples": 16_000, "selection_seed": 17},
        },
    }
    torch.save(checkpoint, checkpoint_path)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    export_report = tmp_path / "export.json"
    export_report.write_text(
        json.dumps({"checkpoint": "mesh2cad-automate-v0.1.pt", "sha256": digest}),
        encoding="utf-8",
    )
    reconstruction = tmp_path / "reconstruction.json"
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
                            "mean_geometric_fidelity": 0.95,
                        },
                        "auto": {
                            "success_rate": 1.0,
                            "analytic_csg_rate": 1.0,
                            "mean_geometric_fidelity": 0.96,
                        },
                    },
                    "neural_not_worse_than_deterministic": {
                        "fraction": 1.0,
                        "comparable_cases": 24,
                        "absolute_fidelity_tolerance": 0.001,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    card = build_model_card(
        checkpoint_path,
        export_report,
        reconstruction_report=reconstruction,
    )

    assert "Prepared together with ChatGPT." in card
    assert "16,000" in card
    assert "CC0-1.0" in card
    assert "60.00%" in card
    assert digest in card
    assert str(tmp_path) not in card


def test_model_card_rejects_mismatched_benchmark(tmp_path) -> None:
    checkpoint_path = tmp_path / "public.pt"
    torch.save({"checkpoint_kind": "mesh2cad-inference"}, checkpoint_path)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    export_report = tmp_path / "export.json"
    benchmark = tmp_path / "benchmark.json"
    export_report.write_text(json.dumps({"sha256": digest}), encoding="utf-8")
    benchmark.write_text(json.dumps({"checkpoint_sha256": "0" * 64}), encoding="utf-8")

    with pytest.raises(ValueError, match="benchmark checkpoint hash"):
        build_model_card(
            checkpoint_path,
            export_report,
            reconstruction_report=benchmark,
        )
