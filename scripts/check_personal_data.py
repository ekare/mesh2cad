"""Fail when release files contain the current user's obvious absolute paths."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--forbidden-marker",
        action="append",
        default=[],
        help="additional private hostname, mount label, or identifier to reject",
    )
    args = parser.parse_args()
    username = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    needles: set[bytes] = set()
    if username:
        needles.update(
            {
                f"C:\\Users\\{username}".encode().lower(),
                f"C:/Users/{username}".encode().lower(),
                f"/home/{username}".encode().lower(),
            }
        )
    needles.update(marker.encode().lower() for marker in args.forbidden_marker if marker)
    if not needles:
        return 0
    failures: list[str] = []
    for path in args.root.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".zip", ".sha256"}:
            continue
        data = path.read_bytes().lower()
        if any(needle in data for needle in needles):
            failures.append(str(path.relative_to(args.root)))
    if failures:
        raise RuntimeError("Personal/private marker found in: " + ", ".join(failures))
    print("Personal/private marker scan passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
