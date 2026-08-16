"""Export exact path-free train/validation/test membership from a private checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('Split export needs `pip install -e ".[ml]"`') from exc
    return torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_split_manifest(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = _torch().load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("training checkpoint must contain a mapping")
    if not checkpoint.get("training_complete", False):
        raise ValueError("public split export requires completed training")
    partitions: dict[str, list[str]] = {}
    owners: set[str] = set()
    for public_name, field in (
        ("train", "training_files"),
        ("validation", "validation_files"),
        ("test", "test_files"),
    ):
        values = checkpoint.get(field)
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"checkpoint field {field!r} must be a sequence")
        names = sorted((Path(str(value)).name for value in values), key=str.casefold)
        if any(not name or Path(name).suffix.casefold() != ".npz" for name in names):
            raise ValueError(f"checkpoint field {field!r} contains an invalid sample identifier")
        if len(names) != len(set(names)):
            raise ValueError(f"checkpoint field {field!r} contains duplicate samples")
        if owners.intersection(names):
            raise ValueError("checkpoint train/validation/test partitions overlap")
        owners.update(names)
        partitions[public_name] = names
    if not partitions["train"]:
        raise ValueError("training checkpoint contains no training samples")
    return {
        "schema_version": 1,
        "dataset": "AutoMate",
        "selection": {
            "seed": checkpoint.get("seed"),
            "limit_per_source": checkpoint.get("max_samples_per_source"),
            "selected_samples": len(owners),
        },
        "partitions": partitions,
    }


def export_split_manifest(checkpoint_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    payload = build_split_manifest(checkpoint_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    return {
        "schema_version": 1,
        "file": destination.name,
        "sha256": _sha256(destination),
        "bytes": destination.stat().st_size,
        "training_samples": len(payload["partitions"]["train"]),
        "validation_samples": len(payload["partitions"]["validation"]),
        "test_samples": len(payload["partitions"]["test"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a path-free training split manifest")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = export_split_manifest(args.checkpoint, args.output)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"mesh2cad-export-split: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
