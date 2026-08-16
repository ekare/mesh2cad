"""Download official ABC dataset shards with resume and integrity checks."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

ABC_BASE = "https://deep-geometry.github.io/abc-dataset/data"
SUPPORTED_FORMATS = ("step", "obj", "feat", "stl2", "meta", "stat", "ofs", "para")


class DatasetError(RuntimeError):
    """Raised for download, checksum, or extraction failures."""


def _read_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "mesh2cad/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def chunk_entry(file_format: str, chunk: int) -> tuple[str, str]:
    if file_format not in SUPPORTED_FORMATS:
        raise DatasetError(f"Unsupported ABC format: {file_format}")
    lines = _read_text(f"{ABC_BASE}/{file_format}_v00.txt").splitlines()
    if chunk < 0 or chunk >= len(lines):
        raise DatasetError(f"ABC chunk must be between 0 and {len(lines) - 1}")
    url, filename = lines[chunk].split(maxsplit=1)
    return url, filename


def expected_checksum(filename: str) -> str:
    pattern = re.compile(rf"^{re.escape(filename)}:\s*([0-9a-fA-F]{{32}})\s*$", re.MULTILINE)
    match = pattern.search(_read_text(f"{ABC_BASE}/md5.yml"))
    if match is None:
        raise DatasetError(f"No official checksum found for {filename}")
    return match.group(1).lower()


def file_md5(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - the publisher's manifest uses MD5
    with Path(path).open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, destination: str | Path, *, max_attempts: int = 10) -> Path:
    """Download to a .part file and resume when the server supports ranges."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    retryable = (
        urllib.error.URLError,
        TimeoutError,
        ConnectionError,
        http.client.HTTPException,
    )
    for attempt in range(1, max_attempts + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "mesh2cad/0.1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                append = offset > 0 and response.status == 206
                mode = "ab" if append else "wb"
                with partial.open(mode) as stream:
                    while block := response.read(8 * 1024 * 1024):
                        stream.write(block)
        except retryable as exc:
            if attempt == max_attempts:
                raise DatasetError(
                    f"Download failed after {max_attempts} attempts; partial file retained at "
                    f"{partial}"
                ) from exc
            time.sleep(min(2**attempt, 30))
            continue
        partial.replace(target)
        return target
    raise AssertionError("unreachable")


def extract_archive(archive: str | Path, destination: str | Path) -> Path:
    try:
        import py7zr
    except ImportError as exc:
        raise DatasetError('Archive extraction needs `pip install -e ".[dataset]"`') from exc
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive, mode="r") as compressed:
        compressed.extractall(path=target)
    return target


def download_chunk(
    root: str | Path,
    *,
    chunk: int = 0,
    file_format: str = "step",
    extract: bool = False,
) -> Path:
    root_path = Path(root).expanduser().resolve()
    url, filename = chunk_entry(file_format, chunk)
    archive = root_path / "v00" / "archives" / filename
    if not archive.exists():
        download_file(url, archive)
    actual = file_md5(archive)
    expected = expected_checksum(filename)
    if actual != expected:
        raise DatasetError(
            f"Checksum mismatch for {archive}: expected {expected}, received {actual}"
        )
    if extract:
        return extract_archive(archive, root_path / "v00" / file_format)
    return archive


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download an official ABC dataset shard")
    parser.add_argument("root", type=Path, help="dataset root directory")
    parser.add_argument("--chunk", type=int, default=0, help="10k-model shard index")
    parser.add_argument("--format", choices=SUPPORTED_FORMATS, default="step")
    parser.add_argument("--extract", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = download_chunk(
            args.root,
            chunk=args.chunk,
            file_format=args.format,
            extract=args.extract,
        )
    except (OSError, DatasetError) as exc:
        print(f"mesh2cad-abc: error: {exc}", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
