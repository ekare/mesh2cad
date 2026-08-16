from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from mesh2cad.learning.split_export import export_split_manifest  # noqa: E402


def test_split_export_is_exact_deterministic_and_path_free(tmp_path) -> None:
    checkpoint = tmp_path / "private.pt"
    torch.save(
        {
            "training_complete": True,
            "seed": 17,
            "max_samples_per_source": 4,
            "training_files": ["/private/z.npz", "/private/a.npz"],
            "validation_files": ["/private/c.npz"],
            "test_files": ["/private/d.npz"],
        },
        checkpoint,
    )
    first = tmp_path / "split-1.json"
    second = tmp_path / "split-2.json"

    first_report = export_split_manifest(checkpoint, first)
    second_report = export_split_manifest(checkpoint, second)
    payload = json.loads(first.read_text(encoding="utf-8"))

    assert payload["partitions"]["train"] == ["a.npz", "z.npz"]
    assert payload["selection"]["selected_samples"] == 4
    assert first.read_bytes() == second.read_bytes()
    assert first_report["sha256"] == second_report["sha256"]
    assert b"/private/" not in first.read_bytes()


def test_split_export_refuses_incomplete_training(tmp_path) -> None:
    checkpoint = tmp_path / "private.pt"
    torch.save({"training_complete": False}, checkpoint)

    with pytest.raises(ValueError, match="completed training"):
        export_split_manifest(checkpoint, tmp_path / "split.json")
