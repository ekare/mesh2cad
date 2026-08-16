from __future__ import annotations

from mesh2cad.learning.repair_benchmark import (
    CORRUPTION_FAMILIES,
    _aggregate,
    _paired_case,
)


def test_paired_corruptions_share_world_frame_and_include_expected_damage() -> None:
    records = [_paired_case(index, seed=23) for index in range(len(CORRUPTION_FAMILIES))]

    assert [metadata["family"] for _source, _reference, metadata in records] == list(
        CORRUPTION_FAMILIES
    )
    for source, reference, _metadata in records:
        assert source.bounds.shape == reference.bounds.shape == (2, 3)
        assert source.extents.min() > 0.0
        assert reference.is_watertight
    missing_source = records[2][0]
    duplicate_source, duplicate_reference, _ = records[3]
    assert missing_source.is_watertight is False
    assert len(duplicate_source.faces) == len(duplicate_reference.faces) + 1


def test_repair_benchmark_aggregation_keeps_worsened_fraction_visible() -> None:
    cases = [
        {
            "corruption": "identity",
            "abstained": False,
            "after_watertight": True,
            "truth_p95_worsened": False,
            "truth_before": {"symmetric_p95_relative": 0.0},
            "truth_after": {"symmetric_p95_relative": 1e-5},
        },
        {
            "corruption": "noise",
            "abstained": True,
            "after_watertight": False,
            "truth_p95_worsened": True,
            "truth_before": {"symmetric_p95_relative": 1e-3},
            "truth_after": {"symmetric_p95_relative": 2e-3},
        },
    ]

    summary = _aggregate(cases)

    assert summary["overall"]["acceptance_rate"] == 0.5
    assert summary["overall"]["worsened_fraction"] == 0.5
    assert summary["by_corruption"]["identity"]["worsened"] == 0
    assert summary["by_corruption"]["noise"]["worsened"] == 1
