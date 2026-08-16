"""Create a path-free public AutoMate preparation provenance record."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mesh2cad.dataset.automate import (
    AUTOMATE_ARCHIVE_BYTES,
    AUTOMATE_ARCHIVE_MD5,
    AUTOMATE_ARCHIVE_SHA256,
    AUTOMATE_STEP_FILES,
    AUTOMATE_UNCOMPRESSED_BYTES,
)


def _read_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return payload


def _failure_category(error: str) -> str:
    normalized = error.casefold()
    if "invalid brep" in normalized:
        return "invalid_brep"
    if "not watertight" in normalized:
        return "not_watertight"
    if "omitted" in normalized and "face" in normalized:
        return "incomplete_tessellation"
    if "no solid" in normalized:
        return "no_solid"
    if "memory" in normalized:
        return "memory_limit"
    if "timeout" in normalized or "timed out" in normalized:
        return "timeout"
    return "other"


def _identifier_digest(values: Sequence[str]) -> str:
    """Hash path-free prepared-sample basenames in a stable order."""
    digest = hashlib.sha256()
    for name in sorted(values, key=str.casefold):
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bind_split_manifest(path: str | Path, subset: dict[str, Any]) -> dict[str, Any]:
    split_path = Path(path)
    payload = _read_object(split_path)
    partitions = payload.get("partitions")
    if not isinstance(partitions, dict):
        raise ValueError("split manifest must contain a partitions object")
    all_names: set[str] = set()
    aliases = (("training", "train"), ("validation", "validation"), ("test", "test"))
    for public_name, manifest_name in aliases:
        values = partitions.get(manifest_name)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"split manifest partition {manifest_name!r} must be a string list")
        if any(
            Path(value).name != value or Path(value).suffix.casefold() != ".npz"
            for value in values
        ):
            raise ValueError("split manifest must contain path-free NPZ basenames")
        if len(values) != len(set(values)) or all_names.intersection(values):
            raise ValueError("split manifest partitions contain duplicate or overlapping samples")
        all_names.update(values)
        if len(values) != int(subset[f"{public_name}_samples"]):
            raise ValueError(f"split manifest {manifest_name} count does not match the checkpoint")
        if _identifier_digest(values) != subset["identifier_sha256"][public_name]:
            raise ValueError(
                f"split manifest {manifest_name} identity does not match the checkpoint"
            )
    if _identifier_digest(list(all_names)) != subset["identifier_sha256"]["selected"]:
        raise ValueError("split manifest selected identity does not match the checkpoint")
    return {
        "file": split_path.name,
        "sha256": _file_sha256(split_path),
        "schema_version": int(payload.get("schema_version", 0)),
    }


def _training_subset(checkpoint_path: str | Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('Staged provenance needs `pip install -e ".[ml]"`') from exc

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("training checkpoint must contain a mapping")
    if not checkpoint.get("training_complete", False):
        raise ValueError("staged public provenance requires completed training")

    split_names: dict[str, list[str]] = {}
    all_names: set[str] = set()
    for public_name, field in (
        ("training", "training_files"),
        ("validation", "validation_files"),
        ("test", "test_files"),
    ):
        values = checkpoint.get(field)
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"checkpoint field {field!r} must be a sequence")
        names = [Path(str(value)).name for value in values]
        if any(not name or Path(name).suffix.casefold() != ".npz" for name in names):
            raise ValueError(f"checkpoint field {field!r} contains an invalid sample identifier")
        if len(names) != len(set(names)):
            raise ValueError(f"checkpoint field {field!r} contains duplicate samples")
        overlap = all_names.intersection(names)
        if overlap:
            raise ValueError("checkpoint train/validation/test partitions overlap")
        all_names.update(names)
        split_names[public_name] = names

    selected_samples = len(all_names)
    if selected_samples < 1 or not split_names["training"]:
        raise ValueError("training checkpoint contains no selected training samples")
    history = checkpoint.get("history", [])
    if not isinstance(history, list):
        raise ValueError("checkpoint history must be a list")
    max_samples = checkpoint.get("max_samples_per_source")
    seed = checkpoint.get("seed")
    dataset_roots = checkpoint.get("dataset_roots", [])
    if not isinstance(dataset_roots, (list, tuple)):
        raise ValueError("checkpoint dataset_roots must be a sequence")

    return {
        "kind": "deterministic_staged_subset",
        "training_complete": True,
        "completed_epochs": len(history),
        "best_epoch": int(checkpoint.get("best_epoch", 0)),
        "source_count": len(dataset_roots),
        "selection_seed": int(seed) if seed is not None else None,
        "selection_limit_per_source": int(max_samples) if max_samples is not None else None,
        "selected_samples": selected_samples,
        "training_samples": len(split_names["training"]),
        "validation_samples": len(split_names["validation"]),
        "test_samples": len(split_names["test"]),
        "selection_algorithm": (
            "seeded per-source sampling followed by a seeded global shuffle and split"
        ),
        "identifier_digest_method": (
            "SHA-256 of sorted prepared-sample basenames, each followed by LF"
        ),
        "identifier_sha256": {
            **{
                name: _identifier_digest(values)
                for name, values in split_names.items()
            },
            "selected": _identifier_digest(list(all_names)),
        },
    }


def build_automate_provenance(
    ingest_manifest: str | Path,
    preparation_manifest: str | Path,
    *,
    allow_partial: bool = False,
    split_seed: int = 17,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    training_checkpoint: str | Path | None = None,
    split_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Strip machine paths and detailed failed filenames from operational manifests."""
    ingest = _read_object(ingest_manifest)
    preparation = _read_object(preparation_manifest)
    if ingest.get("dataset") != "AutoMate":
        raise ValueError("ingest manifest is not for AutoMate")
    if ingest.get("license") != "CC0-1.0":
        raise ValueError("AutoMate ingest manifest does not record CC0-1.0")
    status = str(preparation.get("status", "unknown"))
    if status != "complete" and not allow_partial and training_checkpoint is None:
        raise ValueError("preparation is incomplete; pass --allow-partial only for internal audits")

    processed = int(preparation.get("processed", 0))
    accepted = int(preparation.get("completed", 0))
    rejected = int(preparation.get("failed", 0))
    remaining = int(preparation.get("remaining", max(0, AUTOMATE_STEP_FILES - processed)))
    if accepted < 0 or rejected < 0 or processed < 0 or remaining < 0:
        raise ValueError("preparation counts cannot be negative")
    if accepted + rejected != processed:
        raise ValueError("accepted plus rejected must equal processed")
    recorded_step_files = int(ingest.get("step_files", 0))
    if processed + remaining != recorded_step_files:
        raise ValueError("processed plus remaining must equal the acquired STEP inventory")
    if status == "complete" and remaining != 0:
        raise ValueError("complete preparation cannot have remaining samples")

    failures = preparation.get("failures", [])
    if not isinstance(failures, list):
        raise ValueError("preparation failures must be a list")
    categories = Counter(
        _failure_category(str(record.get("error", "")))
        for record in failures
        if isinstance(record, dict)
    )
    if sum(categories.values()) != rejected:
        raise ValueError("failure ledger count does not match the rejected count")

    configuration = preparation.get("configuration", {})
    if not isinstance(configuration, dict):
        raise ValueError("preparation configuration must be an object")
    public_configuration = {
        key: configuration.get(key)
        for key in (
            "linear_deflection",
            "relative_deflection",
            "angular_deflection",
            "max_triangles",
            "selection_limit",
            "sample_count",
            "seed",
            "shuffle",
            "job_timeout_seconds",
        )
        if key in configuration
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "AutoMate",
            "source": "https://doi.org/10.5061/dryad.2547d7wvw",
            "license": "CC0-1.0",
            "publication_date": "2023-03-27",
            "record_id": "104123",
        },
        "published_archive": {
            "bytes": AUTOMATE_ARCHIVE_BYTES,
            "uncompressed_bytes": AUTOMATE_UNCOMPRESSED_BYTES,
            "step_files": AUTOMATE_STEP_FILES,
            "md5": AUTOMATE_ARCHIVE_MD5,
            "sha256": AUTOMATE_ARCHIVE_SHA256,
        },
        "local_acquisition": {
            "archive_verification_performed": bool(
                ingest.get("archive", {}).get("verification_performed", False)
            ),
            "recorded_step_files": recorded_step_files,
            "recorded_uncompressed_bytes": int(ingest.get("uncompressed_bytes", 0)),
        },
        "preparation": {
            "status": status,
            "processed": processed,
            "accepted": accepted,
            "rejected": rejected,
            "remaining": remaining,
            "failure_categories": dict(sorted(categories.items())),
            "configuration": public_configuration,
            "uncapped_triangles": (
                configuration.get("max_triangles") is None
                if "max_triangles" in configuration
                else None
            ),
        },
        "split_policy": {
            "seed": split_seed,
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
            "ownership": "source STEP file before augmentation",
        },
    }
    if training_checkpoint is not None:
        subset = _training_subset(training_checkpoint)
        if accepted < int(subset["selected_samples"]):
            raise ValueError("preparation snapshot contains fewer accepted samples than training")
        result["release_scope"] = "staged_training_subset"
        result["training_subset"] = subset
        if split_manifest is not None:
            subset["public_split_manifest"] = _bind_split_manifest(split_manifest, subset)
        result["preparation"]["snapshot_complete_for_training_subset"] = True
    else:
        if split_manifest is not None:
            raise ValueError("a public split manifest requires --training-checkpoint")
        result["release_scope"] = (
            "complete_prepared_corpus" if status == "complete" else "internal_partial_audit"
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a path-free public AutoMate preparation provenance JSON"
    )
    parser.add_argument("ingest_manifest", type=Path)
    parser.add_argument("preparation_manifest", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--training-checkpoint",
        type=Path,
        help=(
            "completed staged checkpoint; permits a transparent public subset provenance record "
            "while full preparation is still in progress"
        ),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help="path-free exact split exported from the same completed checkpoint",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        provenance = build_automate_provenance(
            args.ingest_manifest,
            args.preparation_manifest,
            allow_partial=args.allow_partial,
            split_seed=args.split_seed,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            training_checkpoint=args.training_checkpoint,
            split_manifest=args.split_manifest,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".part")
        temporary.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"mesh2cad-automate-provenance: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(provenance, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
