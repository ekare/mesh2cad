from __future__ import annotations

import json

import pytest

from mesh2cad.dataset.parquet_step import ParquetStepError, materialize_step_parquet


def test_materialize_step_parquet_streams_and_resumes(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "source.parquet"
    step = b"ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n"
    table = pa.table({"part_id": ["part/one", "part-two"], "step_bytes": [step, step]})
    pq.write_table(table, source)

    output = tmp_path / "step"
    first = materialize_step_parquet([source], output, batch_size=1)
    second = materialize_step_parquet([tmp_path], output, batch_size=1)

    assert first["written"] == 2
    assert first["rejected"] == 0
    assert second["written"] == 0
    assert second["skipped"] == 2
    assert second["skipped_existing"] == 2
    assert second["duplicates"] == 0
    assert (output / "part_one.step").read_bytes() == step
    manifest = json.loads((output / "materialize-manifest.json").read_text(encoding="utf-8"))
    assert manifest["skipped"] == 2


def test_materialize_step_parquet_rejects_unknown_columns(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"name": ["x"], "blob": [b"not step"]}), source)

    with pytest.raises(ParquetStepError, match="expected columns"):
        materialize_step_parquet([source], tmp_path / "out")


def test_materialize_step_parquet_reports_cross_source_duplicates(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    step = b"ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n"
    train = tmp_path / "train"
    val = tmp_path / "val"
    train.mkdir()
    val.mkdir()
    pq.write_table(pa.table({"uuid": ["same"], "step_file": [step]}), train / "a.parquet")
    pq.write_table(pa.table({"uuid": ["same"], "step_file": [step]}), val / "b.parquet")

    output = tmp_path / "out"
    summary = materialize_step_parquet([train, val], output)

    assert summary["written"] == 1
    assert summary["duplicates"] == 1
    records = [
        json.loads(line)
        for line in (output / "materialize-index.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["status"] for record in records] == ["written", "duplicate"]
    assert records[1]["source_partition"] == "val"
    partitions = json.loads(
        (output / "materialize-partitions.json").read_text(encoding="utf-8")
    )["partitions"]
    assert partitions == {"train": ["same.step"]}
