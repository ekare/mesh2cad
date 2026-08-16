"""Stream STEP payloads from public CAD Parquet datasets onto disk."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


class ParquetStepError(RuntimeError):
    """Raised when a Parquet corpus does not contain usable STEP records."""


STEP_COLUMNS = ("step_bytes", "step_file", "step")
ID_COLUMNS = ("part_id", "uuid", "id", "file_id")


def _safe_stem(value: object) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return stem[:180] or "part"


def _pick_column(names: Iterable[str], requested: str, candidates: Sequence[str]) -> str:
    available = set(names)
    if requested != "auto":
        if requested not in available:
            raise ParquetStepError(f"Column {requested!r} is not present")
        return requested
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise ParquetStepError(f"None of the expected columns are present: {', '.join(candidates)}")


def _step_bytes(value: Any) -> bytes:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if not isinstance(value, bytes):
        raise ParquetStepError(f"STEP payload has unsupported type {type(value).__name__}")
    if b"ISO-10303-21" not in value[:256].upper():
        raise ParquetStepError("STEP payload does not contain an ISO-10303-21 header")
    return value


def materialize_step_parquet(
    sources: Sequence[str | Path],
    destination: str | Path,
    *,
    step_column: str = "auto",
    id_column: str = "auto",
    batch_size: int = 256,
    limit: int | None = None,
) -> dict[str, Any]:
    """Extract STEP byte columns without loading a complete Parquet file into memory."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ParquetStepError('Parquet extraction needs `pip install -e ".[dataset]"`') from exc

    requested_paths = [Path(source).expanduser().resolve() for source in sources]
    if not requested_paths:
        raise ParquetStepError("At least one Parquet source is required")
    missing = [str(path) for path in requested_paths if not path.exists()]
    if missing:
        raise ParquetStepError(f"Parquet source not found: {missing[0]}")
    paths: list[Path] = []
    for path in requested_paths:
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.parquet")))
        elif path.is_file():
            paths.append(path)
    paths = sorted(dict.fromkeys(paths))
    if not paths:
        raise ParquetStepError("No Parquet files were found in the requested sources")
    if batch_size < 1:
        raise ParquetStepError("batch_size must be positive")
    if limit is not None and limit < 1:
        raise ParquetStepError("limit must be positive when provided")

    target = Path(destination).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_existing = 0
    duplicates = 0
    rejected = 0
    considered = 0
    errors: list[dict[str, str]] = []
    seen_outputs: dict[str, str] = {}
    partition_files: dict[str, list[str]] = {}
    index_path = target / "materialize-index.jsonl"
    index_partial = index_path.with_suffix(".jsonl.part")

    with index_partial.open("w", encoding="utf-8") as index_stream:
        for source in paths:
            parquet = pq.ParquetFile(source)
            names = parquet.schema_arrow.names
            payload_name = _pick_column(names, step_column, STEP_COLUMNS)
            identifier_name = _pick_column(names, id_column, ID_COLUMNS)
            for batch in parquet.iter_batches(
                batch_size=batch_size,
                columns=[identifier_name, payload_name],
            ):
                identifiers = batch.column(0).to_pylist()
                payloads = batch.column(1).to_pylist()
                for identifier, payload in zip(identifiers, payloads, strict=True):
                    if limit is not None and considered >= limit:
                        break
                    considered += 1
                    output = target / f"{_safe_stem(identifier)}.step"
                    output_key = output.name.casefold()
                    record = {
                        "id": str(identifier),
                        "output": output.name,
                        "source": str(source),
                        "source_partition": source.parent.name,
                    }
                    owner = seen_outputs.get(output_key)
                    if owner is not None:
                        duplicates += 1
                        record.update({"status": "duplicate", "first_source": owner})
                        index_stream.write(json.dumps(record, sort_keys=True) + "\n")
                        continue
                    seen_outputs[output_key] = str(source)
                    if output.exists() and output.stat().st_size > 0:
                        skipped_existing += 1
                        partition_files.setdefault(source.parent.name, []).append(output.name)
                        record["status"] = "skipped_existing"
                        index_stream.write(json.dumps(record, sort_keys=True) + "\n")
                        continue
                    try:
                        data = _step_bytes(payload)
                        partial = output.with_suffix(".step.part")
                        partial.write_bytes(data)
                        partial.replace(output)
                        written += 1
                        partition_files.setdefault(source.parent.name, []).append(output.name)
                        record["status"] = "written"
                    except (OSError, ParquetStepError) as exc:
                        rejected += 1
                        record.update({"status": "rejected", "error": str(exc)})
                        errors.append({"id": str(identifier), "error": str(exc)})
                    index_stream.write(json.dumps(record, sort_keys=True) + "\n")
                if limit is not None and considered >= limit:
                    break
            if limit is not None and considered >= limit:
                break
    index_partial.replace(index_path)
    partition_manifest = {
        "schema_version": 1,
        "partitions": {
            name: sorted(files) for name, files in sorted(partition_files.items())
        },
    }
    (target / "materialize-partitions.json").write_text(
        json.dumps(partition_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "sources": [str(path) for path in paths],
        "destination": str(target),
        "considered": considered,
        "written": written,
        "skipped": skipped_existing + duplicates,
        "skipped_existing": skipped_existing,
        "duplicates": duplicates,
        "rejected": rejected,
        "partition_counts": {
            name: len(files) for name, files in sorted(partition_files.items())
        },
        "limit": limit,
        "errors": errors,
    }
    (target / "materialize-manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream STEP files from CAD Parquet datasets")
    parser.add_argument(
        "source",
        type=Path,
        nargs="+",
        help="one or more Parquet files or directories searched recursively",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--step-column", default="auto")
    parser.add_argument("--id-column", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, help="optional staging limit; geometry is not reduced")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = materialize_step_parquet(
            args.source,
            args.output,
            step_column=args.step_column,
            id_column=args.id_column,
            batch_size=args.batch_size,
            limit=args.limit,
        )
    except (OSError, ParquetStepError) as exc:
        print(f"mesh2cad-parquet-step: error: {exc}", file=sys.stderr)
        return 2
    console_summary = {
        key: value for key, value in summary.items() if key not in {"sources", "errors"}
    }
    console_summary["source_count"] = len(summary["sources"])
    console_summary["error_count"] = len(summary["errors"])
    print(json.dumps(console_summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
