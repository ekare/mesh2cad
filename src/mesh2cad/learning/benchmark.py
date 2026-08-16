"""Path-free end-to-end reconstruction benchmark for reviewed checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import trimesh

from mesh2cad.pipeline import convert
from mesh2cad.synthetic import cube_with_hole

MODES = ("deterministic", "neural", "auto")
FIDELITY_REGRESSION_TOLERANCE = 1e-3


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synthetic_cases(root: Path, count: int, seed: int) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("synthetic_cases must be positive")
    randomizer = np.random.default_rng(seed)
    cases: list[dict[str, Any]] = []
    for index in range(count):
        size = randomizer.uniform([7.0, 7.0, 3.0], [16.0, 15.0, 9.0])
        radius = float(randomizer.uniform(0.7, min(size[:2]) * 0.24))
        sections = (8, 12, 18, 28)[index % 4]
        mesh = cube_with_hole(
            size=tuple(float(value) for value in size),
            radius=radius,
            sections_per_side=sections,
        )
        axis = randomizer.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = float(randomizer.uniform(0.0, math.pi))
        transform = trimesh.transformations.rotation_matrix(angle, axis)
        translation = randomizer.uniform(-20.0, 20.0, size=3)
        transform[:3, 3] = translation
        mesh.apply_transform(transform)
        path = root / f"synthetic-{index:03d}.stl"
        mesh.export(path, file_type="stl")
        cases.append(
            {
                "path": path,
                "kind": "synthetic_cube_with_through_hole",
                "parameters": {
                    "size": size.tolist(),
                    "radius": radius,
                    "sections_per_side": sections,
                    "rotation_radians": angle,
                    "translation": translation.tolist(),
                },
            }
        )
    return cases


def _mode_result(
    source: Path,
    output: Path,
    *,
    mode: str,
    checkpoint_path: Path,
    device_name: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = convert(
            source,
            output,
            checkpoint_path=checkpoint_path,
            device_name=device_name,
            segmentation_mode=mode,
        )
    except Exception as exc:
        return {
            "success": False,
            "elapsed_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
        }
    return {
        "success": True,
        "elapsed_seconds": time.perf_counter() - started,
        "segmentation_engine": result.segmentation_engine,
        "plane_regions": len(result.segmentation.plane_regions),
        "cylinder_regions": len(result.segmentation.cylinder_regions),
        "unassigned_faces": len(result.segmentation.unassigned_faces),
        "metrics": result.metrics.to_dict(),
        "cad_quality": result.cad_quality(),
        "topology": result.topology.to_dict() if result.topology is not None else None,
    }


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"cases": len(cases), "modes": {}}
    for mode in MODES:
        records = [case["results"][mode] for case in cases]
        successes = [record for record in records if record["success"]]
        fidelities = [
            float(record["metrics"]["geometric_fidelity"])
            for record in successes
            if record.get("metrics", {}).get("geometric_fidelity") is not None
        ]
        analytic = [
            record
            for record in successes
            if record.get("cad_quality", {}).get("representation") == "analytic-csg"
        ]
        summary["modes"][mode] = {
            "successes": len(successes),
            "success_rate": len(successes) / max(1, len(records)),
            "analytic_csg_rate": len(analytic) / max(1, len(records)),
            "mean_geometric_fidelity": (
                float(np.mean(fidelities)) if fidelities else None
            ),
            "minimum_geometric_fidelity": min(fidelities, default=None),
        }

    comparable = 0
    neural_not_worse = 0
    for case in cases:
        deterministic = case["results"]["deterministic"]
        neural = case["results"]["neural"]
        if not deterministic["success"] or not neural["success"]:
            continue
        deterministic_fidelity = deterministic.get("metrics", {}).get("geometric_fidelity")
        neural_fidelity = neural.get("metrics", {}).get("geometric_fidelity")
        if deterministic_fidelity is None or neural_fidelity is None:
            continue
        comparable += 1
        neural_not_worse += (
            float(neural_fidelity) + FIDELITY_REGRESSION_TOLERANCE
            >= float(deterministic_fidelity)
        )
    summary["neural_not_worse_than_deterministic"] = {
        "comparable_cases": comparable,
        "count": neural_not_worse,
        "fraction": neural_not_worse / max(1, comparable),
        "absolute_fidelity_tolerance": FIDELITY_REGRESSION_TOLERANCE,
    }
    return summary


def benchmark(
    inputs: Sequence[str | Path],
    checkpoint_path: str | Path,
    *,
    output_path: str | Path,
    device_name: str = "auto",
    synthetic_cases: int = 0,
    seed: int = 17,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint does not exist: {checkpoint}")
    destination = Path(output_path)
    with TemporaryDirectory(prefix="mesh2cad-benchmark-") as directory:
        workspace = Path(directory)
        cases = [
            {"path": Path(path).resolve(), "kind": "external_mesh", "parameters": None}
            for path in inputs
        ]
        if synthetic_cases:
            cases.extend(_synthetic_cases(workspace, synthetic_cases, seed))
        if not cases:
            raise ValueError("Provide at least one input or --synthetic-cases")

        results: list[dict[str, Any]] = []
        for index, case in enumerate(cases):
            source = Path(case["path"])
            if not source.is_file():
                raise ValueError(f"Benchmark input does not exist: {source}")
            mode_results = {
                mode: _mode_result(
                    source,
                    workspace / f"case-{index:04d}-{mode}.step",
                    mode=mode,
                    checkpoint_path=checkpoint,
                    device_name=device_name,
                )
                for mode in MODES
            }
            results.append(
                {
                    "case_id": f"case-{index:04d}",
                    "kind": case["kind"],
                    "source_sha256": _fingerprint(source),
                    "parameters": case["parameters"],
                    "results": mode_results,
                }
            )

    report = {
        "schema_version": 1,
        "checkpoint_sha256": _fingerprint(checkpoint),
        "device": device_name,
        "seed": seed,
        "summary": _summary(results),
        "cases": results,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare deterministic, direct-neural, and auto STL-to-STEP reconstruction"
    )
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--synthetic-cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = benchmark(
            args.inputs,
            args.checkpoint,
            output_path=args.output,
            device_name=args.device,
            synthetic_cases=args.synthetic_cases,
            seed=args.seed,
        )
    except (OSError, ValueError) as exc:
        print(f"mesh2cad-benchmark: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
