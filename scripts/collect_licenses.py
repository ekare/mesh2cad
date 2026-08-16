"""Collect exact license files for dependencies bundled in a binary release."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import shutil
import sys
from pathlib import Path

PYTHON_DISTRIBUTIONS = (
    "numpy",
    "scipy",
    "trimesh",
    "cadquery-ocp",
    "cadquery-ocp-proxy",
    "vtk",
    "matplotlib",
    "contourpy",
    "cycler",
    "fonttools",
    "kiwisolver",
    "pillow",
    "pyparsing",
    "python-dateutil",
    "six",
    "fastapi",
    "starlette",
    "pydantic",
    "pydantic-core",
    "annotated-types",
    "annotated-doc",
    "typing-extensions",
    "typing-inspection",
    "uvicorn",
    "click",
    "h11",
    "python-multipart",
    "torch",
    "filelock",
    "fsspec",
    "jinja2",
    "markupsafe",
    "networkx",
    "sympy",
    "mpmath",
    "anyio",
    "idna",
    "sniffio",
    "certifi",
    "pyinstaller",
)

LICENSE_WORDS = ("license", "licence", "copying", "notice", "copyright")


def _license_files(distribution: metadata.Distribution) -> list[Path]:
    matches = []
    for entry in distribution.files or ():
        if any(word in entry.name.lower() for word in LICENSE_WORDS):
            source = Path(distribution.locate_file(entry))
            if source.is_file():
                matches.append(source)
    return sorted(set(matches))


def collect(destination: Path, repository: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    inventory: list[str] = []
    missing: list[str] = []

    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if not python_license.is_file():
        missing.append("Python runtime LICENSE.txt")
    else:
        shutil.copy2(python_license, destination / "PYTHON-LICENSE.txt")
        inventory.append(f"Python {sys.version.split()[0]}: PYTHON-LICENSE.txt")

    package_root = destination / "python-packages"
    for name in PYTHON_DISTRIBUTIONS:
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        files = _license_files(distribution)
        if not files:
            if name == "cadquery-ocp-proxy":
                inventory.append(
                    f"{name} {distribution.version}: covered by cadquery-ocp Apache-2.0 file"
                )
                continue
            missing.append(f"{name} {distribution.version}")
            continue
        target = package_root / f"{name}-{distribution.version}"
        target.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(files, start=1):
            relative_name = source.name if len(files) == 1 else f"{index:03d}-{source.name}"
            shutil.copy2(source, target / relative_name)
        inventory.append(f"{name} {distribution.version}: {len(files)} file(s)")

    occt_source = repository / "third_party_licenses" / "occt"
    occt_files = sorted(path for path in occt_source.glob("*.txt") if path.is_file())
    if len(occt_files) < 2:
        missing.append("OCCT LGPL license and special exception")
    else:
        occt_target = destination / "occt"
        occt_target.mkdir(parents=True, exist_ok=True)
        for source in occt_files:
            shutil.copy2(source, occt_target / source.name)
        inventory.append(f"Open CASCADE Technology: {len(occt_files)} file(s)")

    if missing:
        raise RuntimeError("Missing required license material: " + ", ".join(missing))

    inventory.sort(key=str.casefold)
    (destination / "INVENTORY.txt").write_text(
        "mesh2cad binary third-party license inventory\n"
        "Generated from the exact build environment.\n\n"
        + "\n".join(inventory)
        + "\n",
        encoding="utf-8",
    )
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args()
    inventory = collect(args.destination.resolve(), args.repository.resolve())
    print(f"Collected {len(inventory)} license entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
