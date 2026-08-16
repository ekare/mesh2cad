from __future__ import annotations

import json
import zipfile

import pytest

from mesh2cad.dataset.automate import AutoMateError, ingest_automate


def test_ingest_automate_extracts_atomically_and_resumes(tmp_path) -> None:
    archive = tmp_path / "step.zip"
    step = b"ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as compressed:
        compressed.writestr("step/part-a.step", step)
        compressed.writestr("step/nested/part-b.stp", step)

    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "README.md").write_text("AutoMate\n", encoding="utf-8")
    destination = tmp_path / "dataset"

    first = ingest_automate(
        archive,
        destination,
        metadata_root=metadata,
        verify_archive=False,
        progress_every=1,
    )
    second = ingest_automate(
        archive,
        destination,
        verify_archive=False,
        progress_every=0,
    )

    assert first["extracted"] == 2
    assert first["skipped_existing"] == 0
    assert second["extracted"] == 0
    assert second["skipped_existing"] == 2
    assert (destination / "step" / "part-a.step").read_bytes() == step
    assert (destination / "step" / "nested" / "part-b.stp").read_bytes() == step
    assert not list(destination.rglob("*.part"))
    manifest = json.loads((destination / "ingest-manifest.json").read_text(encoding="utf-8"))
    assert manifest["license"] == "CC0-1.0"
    assert manifest["step_files"] == 2


def test_ingest_automate_rejects_path_traversal_before_extraction(tmp_path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as compressed:
        compressed.writestr("step/good.step", b"good")
        compressed.writestr("../escape.step", b"bad")

    destination = tmp_path / "dataset"
    with pytest.raises(AutoMateError, match="Unsafe ZIP member"):
        ingest_automate(archive, destination, verify_archive=False)

    assert not (destination / "step" / "good.step").exists()
    assert not (tmp_path / "escape.step").exists()
