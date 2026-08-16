"""Verify and resumably extract the public AutoMate STEP archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import sys
import zipfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

AUTOMATE_ARCHIVE_BYTES = 13_211_245_622
AUTOMATE_ARCHIVE_MD5 = "7eb29a422507cc881bd4b70416ec723b"
AUTOMATE_ARCHIVE_SHA256 = "6cd36bf6591e8c6ab78e19a20205c218c8a24d971e4bdfc85b95375d33ed45e8"
AUTOMATE_STEP_FILES = 447_442
AUTOMATE_UNCOMPRESSED_BYTES = 74_088_040_746


class AutoMateError(RuntimeError):
    """Raised when an AutoMate archive cannot be safely ingested."""


def _archive_digests(path: Path, block_size: int = 16 * 1024 * 1024) -> tuple[str, str]:
    md5 = hashlib.md5()  # noqa: S324 - AutoMate publishes an MD5 checksum
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            md5.update(block)
            sha256.update(block)
    return md5.hexdigest(), sha256.hexdigest()


def _validated_member(info: zipfile.ZipInfo) -> PurePosixPath | None:
    path = PurePosixPath(info.filename)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise AutoMateError(f"Unsafe ZIP member path: {info.filename!r}")
    if path.parts[0] != "step":
        raise AutoMateError(f"Unexpected AutoMate ZIP member: {info.filename!r}")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise AutoMateError(f"Symbolic links are not allowed in the archive: {info.filename!r}")
    if info.is_dir():
        return None
    if path.suffix.casefold() not in {".step", ".stp"}:
        raise AutoMateError(f"Unexpected non-STEP archive member: {info.filename!r}")
    return path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def _copy_metadata(metadata_root: Path, destination: Path) -> list[str]:
    copied: list[str] = []
    for name in ("README.md", "parts.parquet"):
        source = metadata_root / name
        if source.is_file():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".part")
            shutil.copyfile(source, partial)
            partial.replace(target)
            copied.append(name)
    provenance = metadata_root / "provenance"
    if provenance.is_dir():
        target_root = destination / "provenance"
        target_root.mkdir(parents=True, exist_ok=True)
        for source in sorted(provenance.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(provenance)
            target = target_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".part")
            shutil.copyfile(source, partial)
            partial.replace(target)
            copied.append(str(Path("provenance") / relative))
    return copied


def ingest_automate(
    archive: str | Path,
    destination: str | Path,
    *,
    metadata_root: str | Path | None = None,
    verify_archive: bool = True,
    progress_every: int = 5_000,
) -> dict[str, Any]:
    """Validate and atomically extract AutoMate, safely resuming completed members."""
    archive_path = Path(archive).expanduser().resolve()
    target_root = Path(destination).expanduser().resolve()
    if not archive_path.is_file():
        raise AutoMateError(f"AutoMate archive not found: {archive_path}")
    if progress_every < 0:
        raise ValueError("progress_every cannot be negative")

    archive_size = archive_path.stat().st_size
    digests: dict[str, str] = {}
    if verify_archive:
        if archive_size != AUTOMATE_ARCHIVE_BYTES:
            raise AutoMateError(
                f"Archive size mismatch: expected {AUTOMATE_ARCHIVE_BYTES}, received {archive_size}"
            )
        md5, sha256 = _archive_digests(archive_path)
        digests = {"md5": md5, "sha256": sha256}
        if md5 != AUTOMATE_ARCHIVE_MD5 or sha256 != AUTOMATE_ARCHIVE_SHA256:
            raise AutoMateError("AutoMate archive checksum mismatch")

    target_root.mkdir(parents=True, exist_ok=True)
    progress_path = target_root / "ingest-progress.json"
    started_at = datetime.now(timezone.utc).isoformat()
    extracted = 0
    skipped_existing = 0
    extracted_bytes = 0
    member_count = 0
    member_bytes = 0

    try:
        compressed = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise AutoMateError(f"Cannot open AutoMate ZIP archive: {exc}") from exc

    with compressed:
        # Validate the complete central directory before writing any payload.
        for info in compressed.infolist():
            if _validated_member(info) is not None:
                member_count += 1
                member_bytes += info.file_size
        if verify_archive and (
            member_count != AUTOMATE_STEP_FILES or member_bytes != AUTOMATE_UNCOMPRESSED_BYTES
        ):
            raise AutoMateError(
                "AutoMate archive inventory does not match the published acquisition manifest"
            )

        for info in compressed.infolist():
            member = _validated_member(info)
            if member is None:
                continue
            output = target_root.joinpath(*member.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.is_file() and output.stat().st_size == info.file_size:
                skipped_existing += 1
            else:
                partial = output.with_suffix(output.suffix + ".part")
                try:
                    with compressed.open(info) as source, partial.open("wb") as destination_stream:
                        shutil.copyfileobj(source, destination_stream, length=8 * 1024 * 1024)
                    if partial.stat().st_size != info.file_size:
                        raise AutoMateError(f"Extracted size mismatch for {info.filename!r}")
                    partial.replace(output)
                except (OSError, EOFError, zipfile.BadZipFile) as exc:
                    raise AutoMateError(f"Failed to extract {info.filename!r}: {exc}") from exc
                extracted += 1
                extracted_bytes += info.file_size

            processed = extracted + skipped_existing
            if progress_every and (processed % progress_every == 0 or processed == member_count):
                progress = {
                    "schema_version": 1,
                    "status": "extracting" if processed < member_count else "extracted",
                    "archive": str(archive_path),
                    "destination": str(target_root),
                    "processed": processed,
                    "total": member_count,
                    "extracted": extracted,
                    "skipped_existing": skipped_existing,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                _write_json_atomic(progress_path, progress)
                print(json.dumps(progress, sort_keys=True), flush=True)

    copied_metadata: list[str] = []
    if metadata_root is not None:
        copied_metadata = _copy_metadata(Path(metadata_root).expanduser().resolve(), target_root)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "AutoMate",
        "license": "CC0-1.0",
        "source": "https://doi.org/10.5061/dryad.2547d7wvw",
        "status": "complete",
        "archive": {
            "path": str(archive_path),
            "bytes": archive_size,
            "verification_performed": verify_archive,
            **digests,
        },
        "destination": str(target_root),
        "step_files": member_count,
        "uncompressed_bytes": member_bytes,
        "extracted": extracted,
        "skipped_existing": skipped_existing,
        "extracted_bytes_this_run": extracted_bytes,
        "metadata_files": copied_metadata,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(target_root / "ingest-manifest.json", manifest)
    _write_json_atomic(progress_path, {**manifest, "status": "complete"})
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and resumably extract the official AutoMate STEP archive"
    )
    parser.add_argument("archive", type=Path, help="official step.zip archive")
    parser.add_argument("destination", type=Path, help="AutoMate dataset root")
    parser.add_argument(
        "--metadata-root",
        type=Path,
        help="optional directory containing README.md, parts.parquet, and provenance",
    )
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="skip the full-archive checksum pass (individual ZIP members still use CRC checks)",
    )
    parser.add_argument("--progress-every", type=int, default=5_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = ingest_automate(
            args.archive,
            args.destination,
            metadata_root=args.metadata_root,
            verify_archive=not args.skip_checksum,
            progress_every=args.progress_every,
        )
    except (AutoMateError, OSError, ValueError) as exc:
        print(f"mesh2cad-automate: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
