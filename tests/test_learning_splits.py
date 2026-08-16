from __future__ import annotations

import json

import pytest

from mesh2cad.learning.train import _dataset_files, _split_files_from_manifest


def test_dataset_files_samples_each_source_deterministically(tmp_path) -> None:
    roots = [tmp_path / "one", tmp_path / "two"]
    for root in roots:
        root.mkdir()
        for index in range(5):
            (root / f"sample-{index}.npz").touch()

    first, resolved = _dataset_files(
        roots, seed=23, max_samples_per_source=2
    )
    second, _ = _dataset_files(roots, seed=23, max_samples_per_source=2)

    assert first == second
    assert len(first) == 4
    assert {path.parent for path in first} == set(roots)
    assert resolved == [str(root.resolve()) for root in roots]


def test_explicit_split_manifest_resolves_step_names_to_npz(tmp_path) -> None:
    files = [tmp_path / name for name in ("a.npz", "b.npz", "c.npz")]
    manifest = tmp_path / "splits.json"
    manifest.write_text(
        json.dumps(
            {
                "partitions": {
                    "train": ["a.step"],
                    "val": ["b.step"],
                    "test": ["c.step"],
                }
            }
        ),
        encoding="utf-8",
    )

    train, validation, test = _split_files_from_manifest(files, manifest)

    assert train == [files[0]]
    assert validation == [files[1]]
    assert test == [files[2]]


def test_explicit_split_manifest_rejects_leakage(tmp_path) -> None:
    files = [tmp_path / "same.npz"]
    manifest = tmp_path / "splits.json"
    manifest.write_text(
        json.dumps({"train": ["same.npz"], "validation": ["same.npz"], "test": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="leaks"):
        _split_files_from_manifest(files, manifest)


def test_explicit_split_manifest_allows_rejected_unprepared_samples(tmp_path) -> None:
    files = [tmp_path / "accepted.npz"]
    manifest = tmp_path / "splits.json"
    manifest.write_text(
        json.dumps(
            {
                "train": ["accepted.step", "rejected.step"],
                "validation": [],
                "test": [],
            }
        ),
        encoding="utf-8",
    )

    train, validation, test = _split_files_from_manifest(files, manifest)

    assert train == files
    assert validation == []
    assert test == []


def test_explicit_split_manifest_rejects_prepared_orphans(tmp_path) -> None:
    files = [tmp_path / "orphan.npz"]
    manifest = tmp_path / "splits.json"
    manifest.write_text(
        json.dumps({"train": ["another.step"], "validation": [], "test": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="absent from split manifest"):
        _split_files_from_manifest(files, manifest)
