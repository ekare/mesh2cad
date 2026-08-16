from __future__ import annotations

import json

from mesh2cad.learning.benchmark import MODES, _summary, benchmark


def _result(fidelity: float, *, representation: str = "analytic-csg"):
    return {
        "success": True,
        "metrics": {"geometric_fidelity": fidelity},
        "cad_quality": {"representation": representation},
    }


def test_benchmark_summary_compares_direct_neural_with_deterministic() -> None:
    cases = [
        {
            "results": {
                "deterministic": _result(0.9),
                "neural": _result(0.91),
                "auto": _result(0.91),
            }
        },
        {
            "results": {
                "deterministic": _result(0.95),
                "neural": _result(0.8, representation="faithful-mesh"),
                "auto": _result(0.95),
            }
        },
    ]

    result = _summary(cases)

    assert result["modes"]["neural"]["success_rate"] == 1.0
    assert result["modes"]["neural"]["analytic_csg_rate"] == 0.5
    assert result["neural_not_worse_than_deterministic"]["count"] == 1
    assert result["neural_not_worse_than_deterministic"]["fraction"] == 0.5
    assert (
        result["neural_not_worse_than_deterministic"]["absolute_fidelity_tolerance"]
        == 1e-3
    )


def test_benchmark_report_contains_hashes_but_no_local_input_paths(tmp_path, monkeypatch) -> None:
    import mesh2cad.learning.benchmark as benchmark_module

    checkpoint = tmp_path / "checkpoint.pt"
    source = tmp_path / "private-input.stl"
    output = tmp_path / "report.json"
    checkpoint.write_bytes(b"checkpoint")
    source.write_bytes(b"mesh")
    monkeypatch.setattr(
        benchmark_module,
        "_mode_result",
        lambda *args, **kwargs: _result(1.0),
    )

    report = benchmark([source], checkpoint, output_path=output, device_name="cpu")
    serialized = output.read_text(encoding="utf-8")

    assert set(report["cases"][0]["results"]) == set(MODES)
    assert len(report["checkpoint_sha256"]) == 64
    assert len(report["cases"][0]["source_sha256"]) == 64
    assert str(source) not in serialized
    assert "private-input.stl" not in serialized
    assert json.loads(serialized)["summary"]["cases"] == 1
