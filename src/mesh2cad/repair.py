"""Auditable mesh-repair plugin contract and conservative built-in cleanup."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Protocol

import numpy as np
import trimesh

from mesh2cad.model_resources import default_checkpoint
from mesh2cad.quality import MeshQualityReport, assess_mesh


class RepairError(RuntimeError):
    """Raised when a repairer cannot produce a usable mesh."""


@dataclass(frozen=True)
class RepairResult:
    mesh: trimesh.Trimesh
    repairer: str
    before: MeshQualityReport
    after: MeshQualityReport
    actions: tuple[str, ...]
    confidence: float | None = None
    abstained: bool = False
    metrics: dict[str, object] | None = None
    checkpoint_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "repairer": self.repairer,
            "actions": list(self.actions),
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "confidence": self.confidence,
            "abstained": self.abstained,
            "metrics": self.metrics,
            "checkpoint_sha256": self.checkpoint_sha256,
        }


class MeshRepairer(Protocol):
    name: str

    def repair(self, mesh: trimesh.Trimesh) -> RepairResult: ...


def _single_mesh(loaded: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene) and loaded.geometry:
        return trimesh.util.concatenate(tuple(loaded.geometry.values()))
    raise RepairError("Input does not contain a triangle mesh")


class ConservativeRepairer:
    """Remove invalid duplication and fix winding without inventing missing surface geometry."""

    name = "conservative"

    def repair(self, mesh: trimesh.Trimesh) -> RepairResult:
        source_vertices = len(mesh.vertices)
        source_faces = len(mesh.faces)
        working = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices, dtype=np.float64),
            faces=np.asarray(mesh.faces, dtype=np.int64),
            process=False,
            validate=False,
        )
        before_view = working.copy()
        before_view.merge_vertices()
        before = assess_mesh(
            before_view,
            source_vertices=source_vertices,
            source_faces=source_faces,
        )
        actions: list[str] = []

        original_vertices = len(working.vertices)
        working.merge_vertices()
        if len(working.vertices) != original_vertices:
            actions.append("merge_duplicate_vertices")

        unique = working.unique_faces()
        if int(np.count_nonzero(unique)) != len(working.faces):
            working.update_faces(unique)
            actions.append("remove_duplicate_faces")

        nondegenerate = working.nondegenerate_faces()
        if int(np.count_nonzero(nondegenerate)) != len(working.faces):
            working.update_faces(nondegenerate)
            actions.append("remove_degenerate_faces")

        referenced_before = len(working.vertices)
        working.remove_unreferenced_vertices()
        if len(working.vertices) != referenced_before:
            actions.append("remove_unreferenced_vertices")

        winding_before = bool(working.is_winding_consistent)
        working.fix_normals(multibody=True)
        if not winding_before and working.is_winding_consistent:
            actions.append("orient_face_winding")

        after = assess_mesh(
            working,
            source_vertices=source_vertices,
            source_faces=source_faces,
        )
        return RepairResult(
            mesh=working,
            repairer=self.name,
            before=before,
            after=after,
            actions=tuple(actions),
        )


def available_repairers(
    *,
    checkpoint_path: str | Path | None = None,
    device_name: str = "auto",
) -> dict[str, MeshRepairer]:
    from mesh2cad.learned_repair import NeuralAnalyticRepairer

    repairers: dict[str, MeshRepairer] = {"conservative": ConservativeRepairer()}
    repairers["neural-analytic"] = NeuralAnalyticRepairer(
        checkpoint_path,
        device_name=device_name,
    )
    for entry_point in entry_points(group="mesh2cad.repairers"):
        candidate = entry_point.load()()
        repairers[entry_point.name] = candidate
    return repairers


def repair_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    repairer_name: str = "conservative",
    report_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    device_name: str = "auto",
) -> RepairResult:
    source = Path(input_path)
    destination = Path(output_path)
    loaded = trimesh.load(source, force="mesh", process=False)
    repairer = available_repairers(
        checkpoint_path=checkpoint_path,
        device_name=device_name,
    ).get(repairer_name)
    if repairer is None:
        raise RepairError(f"Unknown repairer: {repairer_name}")
    result = repairer.repair(_single_mesh(loaded))
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.mesh.export(destination, file_type="stl")
    resolved_report = Path(report_path) if report_path is not None else destination.with_suffix(
        ".repair.json"
    )
    resolved_report.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    checkpoint = default_checkpoint()
    parser = argparse.ArgumentParser(
        prog="mesh2cad-repair",
        description="Measure and repair a triangle mesh through an auditable plugin contract.",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--repairer",
        default="neural-analytic" if checkpoint is not None else "conservative",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=checkpoint,
        help="learned checkpoint; a reviewed model may already be bundled",
    )
    parser.add_argument("--device", default="auto", help="neural device: auto, cuda, or cpu")
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = repair_file(
            args.input,
            args.output,
            repairer_name=args.repairer,
            report_path=args.report,
            checkpoint_path=args.checkpoint,
            device_name=args.device,
        )
    except (OSError, ValueError, RepairError) as exc:
        print(f"mesh2cad-repair: error: {exc}", file=sys.stderr)
        return 2
    print(f"Repairer: {result.repairer}")
    print(f"Actions: {', '.join(result.actions) if result.actions else 'none'}")
    if result.confidence is not None:
        print(f"Confidence: {result.confidence:.1%}")
    print(f"Abstained: {'yes' if result.abstained else 'no'}")
    print(
        "Structural health: "
        f"{result.before.structural_health:.1%} -> {result.after.structural_health:.1%}"
    )
    print(f"Repair still recommended: {'yes' if result.after.repair_recommended else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
