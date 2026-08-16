"""Paired synthetic benchmark for the learned analytic repair plugin."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from mesh2cad.learned_repair import NeuralAnalyticRepairer, _fidelity_metrics
from mesh2cad.quality import assess_mesh
from mesh2cad.synthetic import cube_with_hole

CORRUPTION_FAMILIES = ("identity", "coarse", "missing_face", "duplicate_face", "quantized", "noise")


def _mesh_fingerprint(mesh: trimesh.Trimesh) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(mesh.vertices, dtype=np.float64).tobytes())
    digest.update(np.asarray(mesh.faces, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _paired_case(
    index: int, *, seed: int
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, dict[str, Any]]:
    randomizer = np.random.default_rng(seed + index * 1009)
    size = randomizer.uniform([8.0, 8.0, 4.0], [16.0, 15.0, 9.0])
    radius = float(randomizer.uniform(0.8, min(size[:2]) * 0.22))
    family = CORRUPTION_FAMILIES[index % len(CORRUPTION_FAMILIES)]
    reference = cube_with_hole(
        size=tuple(float(value) for value in size),
        radius=radius,
        sections_per_side=32,
    )
    source = (
        cube_with_hole(
            size=tuple(float(value) for value in size),
            radius=radius,
            sections_per_side=8,
        )
        if family == "coarse"
        else reference.copy()
    )
    if family == "missing_face":
        keep = np.ones(len(source.faces), dtype=bool)
        keep[int(randomizer.integers(0, len(source.faces)))] = False
        source.update_faces(keep)
        source.remove_unreferenced_vertices()
    elif family == "duplicate_face":
        source = trimesh.Trimesh(
            vertices=source.vertices.copy(),
            faces=np.vstack((source.faces, source.faces[0])),
            process=False,
        )
    elif family == "quantized":
        grid = float(np.linalg.norm(source.extents)) * 5e-4
        source.vertices = np.round(np.asarray(source.vertices) / grid) * grid
    elif family == "noise":
        sigma = float(np.linalg.norm(source.extents)) * 1e-4
        source.vertices = np.asarray(source.vertices) + randomizer.normal(
            scale=sigma, size=source.vertices.shape
        )

    axis = randomizer.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = float(randomizer.uniform(0.0, math.pi))
    transform = trimesh.transformations.rotation_matrix(angle, axis)
    translation = randomizer.uniform(-15.0, 15.0, size=3)
    transform[:3, 3] = translation
    reference.apply_transform(transform)
    source.apply_transform(transform)
    return source, reference, {
        "family": family,
        "size": size.tolist(),
        "radius": radius,
        "rotation_radians": angle,
        "translation": translation.tolist(),
    }


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_family[str(case["corruption"])].append(case)

    def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
        accepted = [record for record in records if not record["abstained"]]
        worsened = [record for record in records if record["truth_p95_worsened"]]
        return {
            "cases": len(records),
            "accepted": len(accepted),
            "acceptance_rate": len(accepted) / max(1, len(records)),
            "watertight_outputs": sum(record["after_watertight"] for record in records),
            "worsened": len(worsened),
            "worsened_fraction": len(worsened) / max(1, len(records)),
            "mean_truth_p95_before": float(
                np.mean([record["truth_before"]["symmetric_p95_relative"] for record in records])
            ),
            "mean_truth_p95_after": float(
                np.mean([record["truth_after"]["symmetric_p95_relative"] for record in records])
            ),
        }

    return {
        "overall": summarize(cases),
        "by_corruption": {
            family: summarize(records) for family, records in sorted(by_family.items())
        },
    }


def benchmark_repair(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    cases: int = 12,
    seed: int = 17,
    device_name: str = "auto",
    sample_count: int = 512,
    worsening_tolerance: float = 1e-3,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint does not exist: {checkpoint}")
    if cases < 1:
        raise ValueError("cases must be positive")
    repairer = NeuralAnalyticRepairer(
        checkpoint,
        device_name=device_name,
        sample_count=sample_count,
    )
    records: list[dict[str, Any]] = []
    for index in range(cases):
        source, reference, parameters = _paired_case(index, seed=seed)
        before_truth = _fidelity_metrics(source, reference, sample_count=sample_count)
        result = repairer.repair(source)
        after_truth = _fidelity_metrics(result.mesh, reference, sample_count=sample_count)
        records.append(
            {
                "case_id": f"case-{index:04d}",
                "corruption": parameters.pop("family"),
                "parameters": parameters,
                "source_sha256": _mesh_fingerprint(source),
                "reference_sha256": _mesh_fingerprint(reference),
                "abstained": result.abstained,
                "confidence": result.confidence,
                "actions": list(result.actions),
                "before_quality": assess_mesh(source).to_dict(),
                "after_quality": result.after.to_dict(),
                "after_watertight": bool(result.mesh.is_watertight),
                "truth_before": before_truth,
                "truth_after": after_truth,
                "truth_p95_worsened": bool(
                    float(after_truth["symmetric_p95_relative"])
                    > float(before_truth["symmetric_p95_relative"]) + worsening_tolerance
                ),
                "repair_gate_metrics": result.metrics,
            }
        )
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "schema_version": 1,
        "checkpoint_sha256": checkpoint_hash,
        "device": device_name,
        "seed": seed,
        "sample_count_each_direction": sample_count,
        "worsening_tolerance": worsening_tolerance,
        "summary": _aggregate(records),
        "cases": records,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark learned repair against paired clean synthetic BRep tessellations"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-count", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = benchmark_repair(
            args.checkpoint,
            args.output,
            cases=args.cases,
            seed=args.seed,
            device_name=args.device,
            sample_count=args.sample_count,
        )
    except (OSError, ValueError) as exc:
        print(f"mesh2cad-repair-benchmark: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
