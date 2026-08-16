"""Export path-free, inference-only checkpoints for public distribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from mesh2cad import __version__


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('Model export needs `pip install -e ".[ml]"`') from exc
    return torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _best_validation(checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    history = checkpoint.get("history", [])
    best_epoch = int(checkpoint.get("best_epoch", 0))
    if not isinstance(history, list) or best_epoch < 1 or best_epoch > len(history):
        return None
    record = history[best_epoch - 1]
    validation = record.get("validation") if isinstance(record, dict) else None
    return validation if isinstance(validation, dict) else None


def _split_identity(checkpoint: dict[str, Any]) -> dict[str, Any]:
    digests: dict[str, str] = {}
    counts: dict[str, int] = {}
    selected: set[str] = set()
    for public_name, field in (
        ("training", "training_files"),
        ("validation", "validation_files"),
        ("test", "test_files"),
    ):
        values = checkpoint.get(field, ())
        names = [Path(str(value)).name for value in values]
        if len(names) != len(set(names)):
            raise ValueError(f"checkpoint field {field!r} contains duplicate samples")
        if selected.intersection(names):
            raise ValueError("checkpoint train/validation/test partitions overlap")
        selected.update(names)
        digest = hashlib.sha256()
        for name in sorted(names, key=str.casefold):
            digest.update(name.encode("utf-8"))
            digest.update(b"\n")
        counts[public_name] = len(names)
        digests[public_name] = digest.hexdigest()
    combined = hashlib.sha256()
    for name in sorted(selected, key=str.casefold):
        combined.update(name.encode("utf-8"))
        combined.update(b"\n")
    digests["selected"] = combined.hexdigest()
    return {
        "method": "SHA-256 of sorted prepared-sample basenames, each followed by LF",
        "counts": {**counts, "selected": len(selected)},
        "sha256": digests,
    }


def _validate_subset_provenance(
    checkpoint: dict[str, Any], provenance: dict[str, Any], identity: dict[str, Any]
) -> None:
    subset = provenance.get("training_subset")
    if subset is None:
        return
    if not isinstance(subset, dict):
        raise ValueError("provenance training_subset must be an object")
    counts = identity["counts"]
    for name, key in (
        ("training", "training_samples"),
        ("validation", "validation_samples"),
        ("test", "test_samples"),
        ("selected", "selected_samples"),
    ):
        if int(subset.get(key, -1)) != int(counts[name]):
            raise ValueError(f"provenance {key} does not match the checkpoint")
    declared_digests = subset.get("identifier_sha256")
    if declared_digests is not None and declared_digests != identity["sha256"]:
        raise ValueError("provenance split identifier digests do not match the checkpoint")
    declared_seed = subset.get("selection_seed")
    if declared_seed is not None and int(declared_seed) != int(checkpoint.get("seed", -1)):
        raise ValueError("provenance selection seed does not match the checkpoint")
    declared_limit = subset.get("selection_limit_per_source")
    checkpoint_limit = checkpoint.get("max_samples_per_source")
    if declared_limit is not None and int(declared_limit) != int(checkpoint_limit or -1):
        raise ValueError("provenance selection limit does not match the checkpoint")


def export_inference_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    provenance: dict[str, Any] | None = None,
    model_license: str = "MIT",
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Write an atomic checkpoint without optimizer, RNG, or machine-local path state."""
    torch = _torch()
    source_path = Path(source)
    output_path = Path(destination)
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("training checkpoint must contain a mapping")
    if not checkpoint.get("training_complete", False) and not allow_incomplete:
        raise ValueError("refusing to export an incomplete training checkpoint")
    required = ("model_state", "model", "surface_classes", "feature_names")
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise ValueError(f"checkpoint is missing public inference fields: {', '.join(missing)}")
    for name in ("training_files", "validation_files", "test_files"):
        values = checkpoint.get(name, ())
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"checkpoint field {name!r} must be a sequence")
    split_identity = _split_identity(checkpoint)
    _validate_subset_provenance(checkpoint, provenance or {}, split_identity)

    history = checkpoint.get("history", [])
    best_epoch = int(checkpoint.get("best_epoch", 0))
    best_record = (
        history[best_epoch - 1]
        if isinstance(history, list) and 1 <= best_epoch <= len(history)
        else {}
    )
    public_checkpoint = {
        "checkpoint_kind": "mesh2cad-inference",
        "schema_version": 1,
        "mesh2cad_version": __version__,
        "model_license": model_license,
        "model_state": checkpoint["model_state"],
        "model": checkpoint["model"],
        "surface_classes": tuple(checkpoint["surface_classes"]),
        "feature_names": tuple(checkpoint["feature_names"]),
        "amp": bool(checkpoint.get("amp", False)),
        "partition": checkpoint.get("partition", {}),
        "augmentation": checkpoint.get("augmentation", {}),
        "quality": {
            "selection_metric": checkpoint.get("selection_metric", "legacy-all-surfaces"),
            "best_epoch": best_epoch,
            "best_score": float(checkpoint.get("best_score", -1.0)),
            "best_train_loss": (
                best_record.get("train_loss") if isinstance(best_record, dict) else None
            ),
            "validation": _best_validation(checkpoint),
            "test": checkpoint.get("test"),
        },
        "training_summary": {
            "training_complete": bool(checkpoint.get("training_complete", False)),
            "completed_epochs": len(history) if isinstance(history, list) else 0,
            "training_samples": len(checkpoint.get("training_files", ())),
            "validation_samples": len(checkpoint.get("validation_files", ())),
            "test_samples": len(checkpoint.get("test_files", ())),
            "dataset_statistics": checkpoint.get("dataset_statistics", {}),
            "split_identity": split_identity,
        },
        "provenance": provenance or {},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    with temporary.open("wb") as stream:
        torch.save(public_checkpoint, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output_path)
    return {
        "schema_version": 1,
        "checkpoint": output_path.name,
        "sha256": _sha256(output_path),
        "bytes": output_path.stat().st_size,
        "model_license": model_license,
        "training_complete": public_checkpoint["training_summary"]["training_complete"],
        "training_samples": public_checkpoint["training_summary"]["training_samples"],
        "validation_samples": public_checkpoint["training_summary"]["validation_samples"],
        "test_samples": public_checkpoint["training_summary"]["test_samples"],
        "best_epoch": best_epoch,
        "best_score": public_checkpoint["quality"]["best_score"],
        "split_identity": split_identity,
        "source_sha256": _sha256(source_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a path-free inference checkpoint for public distribution"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, help="public dataset provenance JSON")
    parser.add_argument("--report", type=Path, help="write an external hash/quality report")
    parser.add_argument("--model-license", default="MIT")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="export a recovery checkpoint for private testing (never a public release)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        provenance = (
            json.loads(args.provenance.read_text(encoding="utf-8"))
            if args.provenance is not None
            else None
        )
        if provenance is not None and not isinstance(provenance, dict):
            raise ValueError("provenance JSON must contain an object")
        report = export_inference_checkpoint(
            args.checkpoint,
            args.output,
            provenance=provenance,
            model_license=args.model_license,
            allow_incomplete=args.allow_incomplete,
        )
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (json.JSONDecodeError, OSError, RuntimeError, ValueError) as exc:
        print(f"mesh2cad-export-model: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
