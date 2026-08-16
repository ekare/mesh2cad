"""Apply explicit v0.1 release gates to a public model and its benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

MIN_TEST_ANALYTIC_IOU = 0.50
MIN_TEST_BOUNDARY_F1 = 0.94
MIN_BENCHMARK_CASES = 24
MIN_RECONSTRUCTION_RATE = 0.95
MIN_NEURAL_NOT_WORSE_RATE = 0.95
MIN_REPAIR_WATERTIGHT_RATE = 0.95
MAX_REPAIR_WORSENED_RATE = 0.0


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('Model qualification needs `pip install -e ".[ml]"`') from exc
    return torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return payload


def _gate(actual: Any, operator: str, threshold: Any, passed: bool) -> dict[str, Any]:
    return {
        "actual": actual,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(passed),
    }


def qualify_model(
    checkpoint_path: str | Path,
    reconstruction_report: str | Path,
    repair_report: str | Path,
) -> dict[str, Any]:
    checkpoint_file = Path(checkpoint_path)
    checkpoint_hash = _sha256(checkpoint_file)
    checkpoint = _torch().load(checkpoint_file, map_location="cpu", weights_only=False)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("checkpoint_kind") != "mesh2cad-inference"
    ):
        raise ValueError("checkpoint is not a public mesh2cad inference artifact")
    reconstruction = _read_json(reconstruction_report)
    repair = _read_json(repair_report)
    if reconstruction.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("reconstruction report checkpoint hash does not match")
    if repair.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("repair report checkpoint hash does not match")

    test = checkpoint.get("quality", {}).get("test") or {}
    reconstruction_summary = reconstruction.get("summary", {})
    neural = reconstruction_summary.get("modes", {}).get("neural", {})
    comparison = reconstruction_summary.get("neural_not_worse_than_deterministic", {})
    repair_summary = repair.get("summary", {}).get("overall", {})
    repair_cases = int(repair_summary.get("cases", 0))
    watertight_rate = float(repair_summary.get("watertight_outputs", 0)) / max(1, repair_cases)
    gates = {
        "training_complete": _gate(
            checkpoint.get("training_summary", {}).get("training_complete"),
            "is",
            True,
            checkpoint.get("training_summary", {}).get("training_complete") is True,
        ),
        "test_plane_cylinder_macro_iou": _gate(
            test.get("supported_surface_macro_iou"),
            ">=",
            MIN_TEST_ANALYTIC_IOU,
            float(test.get("supported_surface_macro_iou", 0.0)) >= MIN_TEST_ANALYTIC_IOU,
        ),
        "test_boundary_f1": _gate(
            test.get("boundary_f1"),
            ">=",
            MIN_TEST_BOUNDARY_F1,
            float(test.get("boundary_f1", 0.0)) >= MIN_TEST_BOUNDARY_F1,
        ),
        "reconstruction_cases": _gate(
            reconstruction_summary.get("cases"),
            ">=",
            MIN_BENCHMARK_CASES,
            int(reconstruction_summary.get("cases", 0)) >= MIN_BENCHMARK_CASES,
        ),
        "direct_neural_success_rate": _gate(
            neural.get("success_rate"),
            ">=",
            MIN_RECONSTRUCTION_RATE,
            float(neural.get("success_rate", 0.0)) >= MIN_RECONSTRUCTION_RATE,
        ),
        "direct_neural_analytic_csg_rate": _gate(
            neural.get("analytic_csg_rate"),
            ">=",
            MIN_RECONSTRUCTION_RATE,
            float(neural.get("analytic_csg_rate", 0.0)) >= MIN_RECONSTRUCTION_RATE,
        ),
        "neural_not_worse_than_deterministic": _gate(
            comparison.get("fraction"),
            ">=",
            MIN_NEURAL_NOT_WORSE_RATE,
            float(comparison.get("fraction", 0.0)) >= MIN_NEURAL_NOT_WORSE_RATE,
        ),
        "repair_cases": _gate(
            repair_cases,
            ">=",
            MIN_BENCHMARK_CASES,
            repair_cases >= MIN_BENCHMARK_CASES,
        ),
        "repair_watertight_output_rate": _gate(
            watertight_rate,
            ">=",
            MIN_REPAIR_WATERTIGHT_RATE,
            watertight_rate >= MIN_REPAIR_WATERTIGHT_RATE,
        ),
        "repair_worsened_fraction": _gate(
            repair_summary.get("worsened_fraction"),
            "<=",
            MAX_REPAIR_WORSENED_RATE,
            float(repair_summary.get("worsened_fraction", 1.0))
            <= MAX_REPAIR_WORSENED_RATE,
        ),
    }
    failures = [name for name, gate in gates.items() if not gate["passed"]]
    return {
        "schema_version": 1,
        "checkpoint": checkpoint_file.name,
        "checkpoint_sha256": checkpoint_hash,
        "passed": not failures,
        "failed_gates": failures,
        "gates": gates,
        "non_gating_observations": {
            "repair_acceptance_rate": repair_summary.get("acceptance_rate"),
            "repair_mean_truth_p95_before": repair_summary.get("mean_truth_p95_before"),
            "repair_mean_truth_p95_after": repair_summary.get("mean_truth_p95_after"),
            "direct_neural_mean_geometric_fidelity": neural.get("mean_geometric_fidelity"),
        },
    }


def write_qualification(report: dict[str, Any], output_path: str | Path) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply mesh2cad v0.1 model release gates")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--reconstruction-report", type=Path, required=True)
    parser.add_argument("--repair-report", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = qualify_model(
            args.checkpoint,
            args.reconstruction_report,
            args.repair_report,
        )
        write_qualification(report, args.output)
    except (json.JSONDecodeError, OSError, RuntimeError, ValueError) as exc:
        print(f"mesh2cad-qualify-model: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
