"""Generate a release model card from verified, path-free artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('Model-card generation needs `pip install -e ".[ml]"`') from exc
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


def _percent(value: Any) -> str:
    return "not reported" if value is None else f"{float(value) * 100:.2f}%"


def _decimal(value: Any) -> str:
    return "not reported" if value is None else f"{float(value):.6f}"


def _validate_benchmark(
    report: dict[str, Any] | None, checkpoint_sha256: str, label: str
) -> dict[str, Any] | None:
    if report is None:
        return None
    if report.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"{label} benchmark checkpoint hash does not match the public model")
    return report


def build_model_card(
    checkpoint_path: str | Path,
    export_report: str | Path,
    *,
    reconstruction_report: str | Path | None = None,
    repair_report: str | Path | None = None,
) -> str:
    """Build an English model card while refusing mismatched evaluation artifacts."""
    torch = _torch()
    checkpoint_file = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("checkpoint_kind") != "mesh2cad-inference"
    ):
        raise ValueError("checkpoint is not a public mesh2cad inference artifact")
    report = _read_json(export_report)
    checkpoint_sha256 = _sha256(checkpoint_file)
    if report.get("sha256") != checkpoint_sha256:
        raise ValueError("export report hash does not match the public checkpoint")
    reconstruction = _validate_benchmark(
        _read_json(reconstruction_report) if reconstruction_report is not None else None,
        checkpoint_sha256,
        "reconstruction",
    )
    repair = _validate_benchmark(
        _read_json(repair_report) if repair_report is not None else None,
        checkpoint_sha256,
        "repair",
    )

    provenance = checkpoint.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("checkpoint provenance must be an object")
    dataset = provenance.get("dataset", {})
    subset = provenance.get("training_subset", {})
    summary = checkpoint.get("training_summary", {})
    quality = checkpoint.get("quality", {})
    validation = quality.get("validation") or {}
    test = quality.get("test") or {}
    model = checkpoint.get("model", {})
    surface_classes = ", ".join(str(value) for value in checkpoint.get("surface_classes", ()))
    source_name = str(dataset.get("name", "not reported"))
    source_url = str(dataset.get("source", ""))
    source_label = f"[{source_name}]({source_url})" if source_url else source_name

    lines = [
        "# mesh2cad AutoMate v0.1 model card",
        "",
        "## Model summary",
        "",
        "This is the first staged neural surface-segmentation model for mesh2cad. It predicts a "
        "surface class for each STL triangle and a boundary probability for adjacent triangles. "
        "The deterministic geometry layer verifies the proposal, fits analytic planes and "
        "cylinders, constructs a BRep with OpenCascade, and measures the result against the input.",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Public checkpoint | `{report.get('checkpoint', checkpoint_file.name)}` |",
        f"| SHA-256 | `{checkpoint_sha256}` |",
        f"| mesh2cad version | `{checkpoint.get('mesh2cad_version', 'unknown')}` |",
        f"| Model license | `{checkpoint.get('model_license', 'not reported')}` |",
        f"| Best-checkpoint objective | `{quality.get('selection_metric', 'not reported')}` |",
        f"| Architecture | TriangleGraphNet, hidden size {model.get('hidden_size', 'unknown')}, "
        f"{model.get('layers', 'unknown')} message-passing layers |",
        f"| Predicted surface classes | {surface_classes or 'not reported'} |",
        "| Analytic reconstruction in v0.1 | planes and cylinders |",
        "| Author | Prepared together with ChatGPT. |",
        "",
        "## Training data and provenance",
        "",
        f"The checkpoint was trained on a deterministic staged subset of {source_label}, published "
        f"under `{dataset.get('license', 'not reported')}`. It does not claim training on the full "
        "upstream corpus.",
        "",
        "| Split | Samples |",
        "|---|---:|",
        f"| Training | {int(summary.get('training_samples', 0)):,} |",
        f"| Validation | {int(summary.get('validation_samples', 0)):,} |",
        f"| Held-out test | {int(summary.get('test_samples', 0)):,} |",
        f"| Total selected | {int(subset.get('selected_samples', 0)):,} |",
        "",
        f"Selection seed: `{subset.get('selection_seed', 'not reported')}`. Completed epochs: "
        f"`{summary.get('completed_epochs', 'not reported')}`. Best epoch: "
        f"`{quality.get('best_epoch', 'not reported')}`. The embedded provenance contains "
        "path-free "
        "SHA-256 identities for the exact split membership.",
        "",
        "## Held-out metrics",
        "",
        "| Metric | Validation (best epoch) | Held-out test |",
        "|---|---:|---:|",
        f"| Surface macro IoU | {_percent(validation.get('surface_macro_iou'))} | "
        f"{_percent(test.get('surface_macro_iou'))} |",
        f"| Plane/cylinder macro IoU | {_percent(validation.get('supported_surface_macro_iou'))} | "
        f"{_percent(test.get('supported_surface_macro_iou'))} |",
        f"| Surface accuracy | {_percent(validation.get('surface_accuracy'))} | "
        f"{_percent(test.get('surface_accuracy'))} |",
        f"| Boundary F1 | {_percent(validation.get('boundary_f1'))} | "
        f"{_percent(test.get('boundary_f1'))} |",
        "",
    ]

    if reconstruction is not None:
        benchmark_summary = reconstruction.get("summary", {})
        modes = benchmark_summary.get("modes", {})
        neural = modes.get("neural", {})
        automatic = modes.get("auto", {})
        comparison = benchmark_summary.get("neural_not_worse_than_deterministic", {})
        lines.extend(
            [
                "## End-to-end synthetic reconstruction benchmark",
                "",
                f"Cases: `{benchmark_summary.get('cases', 0)}` arbitrarily rotated and coarsened "
                "cube-with-through-hole meshes.",
                "",
                "| Metric | Direct neural | Auto |",
                "|---|---:|---:|",
                f"| Successful conversion | {_percent(neural.get('success_rate'))} | "
                f"{_percent(automatic.get('success_rate'))} |",
                f"| Analytic CSG output | {_percent(neural.get('analytic_csg_rate'))} | "
                f"{_percent(automatic.get('analytic_csg_rate'))} |",
                f"| Mean geometric fidelity | {_decimal(neural.get('mean_geometric_fidelity'))} | "
                f"{_decimal(automatic.get('mean_geometric_fidelity'))} |",
                "",
                f"Direct neural fidelity was within "
                f"`{comparison.get('absolute_fidelity_tolerance', 'not reported')}` of "
                "deterministic "
                f"reconstruction on {_percent(comparison.get('fraction'))} of "
                f"`{comparison.get('comparable_cases', 0)}` comparable cases.",
                "",
            ]
        )

    if repair is not None:
        repair_summary = repair.get("summary", {}).get("overall", {})
        lines.extend(
            [
                "## Paired synthetic repair benchmark",
                "",
                "| Metric | Result |",
                "|---|---:|",
                f"| Cases | {int(repair_summary.get('cases', 0))} |",
                f"| Accepted repairs | {_percent(repair_summary.get('acceptance_rate'))} |",
                f"| Watertight outputs | {int(repair_summary.get('watertight_outputs', 0))} |",
                f"| Repairs made worse beyond tolerance | "
                f"{_percent(repair_summary.get('worsened_fraction'))} |",
                f"| Mean relative P95 before | "
                f"{_decimal(repair_summary.get('mean_truth_p95_before'))} |",
                f"| Mean relative P95 after | "
                f"{_decimal(repair_summary.get('mean_truth_p95_after'))} |",
                "",
            ]
        )

    lines.extend(
        [
            "## Intended use",
            "",
            "Use this model as an experimental proposal stage for simple mechanical triangle "
            "meshes dominated by planar faces and circular through-hole cylinders. Always inspect "
            "the generated STEP, the topology report, and the bidirectional fidelity metrics.",
            "",
            "## Limitations and out-of-scope use",
            "",
            "- v0.1 analytically reconstructs planes and cylinders; the classifier's other surface "
            "labels do not imply complete cone, sphere, torus, or free-form BRep reconstruction.",
            "- Training is a 16K staged subset, not the full AutoMate corpus.",
            "- Synthetic reconstruction and repair cases do not establish performance on every "
            "scanner, tessellator, topology, unit system, or damaged production mesh.",
            "- The repair path may abstain. An accepted repair can still be wrong.",
            "- Do not use the model or its output as the sole basis for safety-critical, medical, "
            "aerospace, automotive, structural, regulated, high-value, or production decisions.",
            "",
            "## Disclaimer",
            "",
            "This experimental model and all generated outputs are provided as is, without a "
            "guarantee of correctness, dimensional accuracy, manufacturability, or fitness for any "
            "purpose. You are solely responsible for independently validating every result and for "
            "all consequences of using, modifying, manufacturing from, distributing, or relying on "
            "it. See the project `DISCLAIMER.md` for the complete terms.",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a verified mesh2cad model card")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--export-report", type=Path, required=True)
    parser.add_argument("--reconstruction-report", type=Path)
    parser.add_argument("--repair-report", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        card = build_model_card(
            args.checkpoint,
            args.export_report,
            reconstruction_report=args.reconstruction_report,
            repair_report=args.repair_report,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".part")
        temporary.write_text(card, encoding="utf-8")
        temporary.replace(args.output)
    except (json.JSONDecodeError, OSError, RuntimeError, ValueError) as exc:
        print(f"mesh2cad-model-card: error: {exc}", file=sys.stderr)
        return 2
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
