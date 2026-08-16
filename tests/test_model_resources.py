from __future__ import annotations

from pathlib import Path

from mesh2cad.model_resources import (
    BUNDLED_MODEL_FILENAME,
    bundled_checkpoint,
    default_checkpoint,
    default_segmentation_engine,
)


def test_environment_checkpoint_defaults_to_direct_neural(monkeypatch, tmp_path) -> None:
    model = tmp_path / "reviewed.pt"
    monkeypatch.setenv("MESH2CAD_CHECKPOINT", str(model))

    assert default_checkpoint() == model
    assert default_segmentation_engine(model) == "neural"


def test_explicit_engine_environment_wins(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MESH2CAD_SEGMENTATION_ENGINE", "deterministic")

    assert default_segmentation_engine(tmp_path / "reviewed.pt") == "deterministic"


def test_frozen_bundle_is_discovered_without_personal_paths(monkeypatch, tmp_path) -> None:
    model = tmp_path / "mesh2cad" / "models" / BUNDLED_MODEL_FILENAME
    model.parent.mkdir(parents=True)
    model.write_bytes(b"public inference checkpoint")
    monkeypatch.setattr("mesh2cad.model_resources.sys._MEIPASS", str(tmp_path), raising=False)

    assert bundled_checkpoint() == Path(model)
