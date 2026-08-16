"""Learned analytic mesh repair backed by the triangle segmentation network."""

from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import trimesh

from mesh2cad.csg import evaluate
from mesh2cad.mesh import load_mesh
from mesh2cad.pipeline import analyze
from mesh2cad.preview import tessellate_shape
from mesh2cad.quality import assess_mesh
from mesh2cad.repair import ConservativeRepairer, RepairError, RepairResult
from mesh2cad.topology import _load_ocp, _validate_shape, recognize_csg


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape_mesh(
    shape: Any,
    *,
    relative_deflection: float,
    angular_deflection: float,
) -> trimesh.Trimesh:
    preview = tessellate_shape(
        shape,
        relative_deflection=relative_deflection,
        angular_deflection=angular_deflection,
        max_triangles=None,
    )
    triangles = np.asarray(preview.positions, dtype=np.float64)
    candidate = trimesh.Trimesh(
        vertices=triangles.reshape(-1, 3),
        faces=np.arange(triangles.size // 3, dtype=np.int64).reshape(-1, 3),
        process=False,
        validate=False,
    )
    candidate.merge_vertices()
    candidate.update_faces(candidate.unique_faces())
    candidate.update_faces(candidate.nondegenerate_faces())
    candidate.remove_unreferenced_vertices()
    candidate.fix_normals(multibody=True)
    return candidate


def _point_distances(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(trimesh.proximity.closest_point(mesh, points)[1], dtype=np.float64)
    except (ImportError, ModuleNotFoundError):
        distances = []
        for start in range(0, len(points), 64):
            distances.append(
                trimesh.proximity.closest_point_naive(mesh, points[start : start + 64])[1]
            )
        return np.concatenate(distances)


def _fidelity_metrics(
    source: trimesh.Trimesh,
    candidate: trimesh.Trimesh,
    *,
    sample_count: int,
) -> dict[str, float | int | None]:
    scale = float(np.linalg.norm(source.extents))
    if scale <= np.finfo(float).eps:
        raise RepairError("Cannot measure a degenerate source mesh")
    source_points, _ = trimesh.sample.sample_surface(source, sample_count, seed=1701)
    candidate_points, _ = trimesh.sample.sample_surface(candidate, sample_count, seed=1702)
    distances = np.concatenate(
        (
            _point_distances(candidate, source_points),
            _point_distances(source, candidate_points),
        )
    )
    normalized = distances / scale
    source_area = max(float(source.area), np.finfo(float).eps)
    area_error = abs(float(candidate.area) - float(source.area)) / source_area
    volume_error = None
    if source.is_watertight and candidate.is_watertight:
        source_volume = max(abs(float(source.volume)), np.finfo(float).eps)
        volume_error = abs(abs(float(candidate.volume)) - abs(float(source.volume))) / source_volume
    return {
        "symmetric_rms_relative": float(np.sqrt(np.mean(np.square(normalized)))),
        "symmetric_p95_relative": float(np.quantile(normalized, 0.95)),
        "symmetric_p99_relative": float(np.quantile(normalized, 0.99)),
        "symmetric_max_relative": float(np.max(normalized)),
        "relative_area_error": area_error,
        "relative_volume_error": volume_error,
        "sample_count_each_direction": sample_count,
    }


class NeuralAnalyticRepairer:
    """Use learned surface labels to rebuild and fully retessellate supported analytic CSG."""

    name = "neural-analytic"

    def __init__(
        self,
        checkpoint_path: str | Path | None,
        *,
        device_name: str = "auto",
        relative_deflection: float = 5e-4,
        angular_deflection: float = 0.1,
        sample_count: int = 2_048,
        max_p95_relative: float = 0.025,
        max_deviation_relative: float = 0.06,
        max_area_error: float = 0.10,
        max_volume_error: float = 0.08,
    ) -> None:
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
        self.device_name = device_name
        self.relative_deflection = relative_deflection
        self.angular_deflection = angular_deflection
        self.sample_count = sample_count
        self.max_p95_relative = max_p95_relative
        self.max_deviation_relative = max_deviation_relative
        self.max_area_error = max_area_error
        self.max_volume_error = max_volume_error

    def _abstain(
        self,
        conservative: RepairResult,
        *,
        reason: str,
        checkpoint_hash: str,
        metrics: dict[str, object] | None = None,
    ) -> RepairResult:
        details = dict(metrics or {})
        details["abstention_reason"] = reason
        return RepairResult(
            mesh=conservative.mesh,
            repairer=self.name,
            before=conservative.before,
            after=conservative.after,
            actions=(*conservative.actions, "neural_abstained"),
            confidence=0.0,
            abstained=True,
            metrics=details,
            checkpoint_sha256=checkpoint_hash,
        )

    def repair(self, mesh: trimesh.Trimesh) -> RepairResult:
        if self.checkpoint_path is None:
            raise RepairError("The neural-analytic repairer requires --checkpoint")
        if not self.checkpoint_path.is_file():
            raise RepairError(f"Repair checkpoint does not exist: {self.checkpoint_path}")
        if self.sample_count < 32:
            raise RepairError("Neural repair fidelity sampling requires at least 32 points")

        checkpoint_hash = _sha256(self.checkpoint_path)
        conservative = ConservativeRepairer().repair(mesh)
        working = conservative.mesh
        try:
            with TemporaryDirectory(prefix="mesh2cad-neural-repair-") as directory:
                source_path = Path(directory) / "source.stl"
                working.export(source_path, file_type="stl")
                data = load_mesh(source_path)
                proposal = analyze(
                    source_path,
                    checkpoint_path=self.checkpoint_path,
                    device_name=self.device_name,
                    segmentation_mode="neural",
                )
                assigned_area = proposal.metrics.assigned_area_fraction
                if assigned_area < 0.90:
                    return self._abstain(
                        conservative,
                        reason="learned surface coverage is below 90%",
                        checkpoint_hash=checkpoint_hash,
                        metrics={"assigned_area_fraction": assigned_area},
                    )
                graph = recognize_csg(data, proposal.segmentation)
                ocp = _load_ocp()
                shape = evaluate(graph, ocp)
                _validate_shape(shape, ocp)
                candidate = _shape_mesh(
                    shape,
                    relative_deflection=self.relative_deflection,
                    angular_deflection=self.angular_deflection,
                )
        except Exception as exc:
            return self._abstain(
                conservative,
                reason=f"analytic reconstruction failed: {type(exc).__name__}",
                checkpoint_hash=checkpoint_hash,
            )

        metrics = _fidelity_metrics(working, candidate, sample_count=self.sample_count)
        after = assess_mesh(
            candidate,
            source_vertices=len(mesh.vertices),
            source_faces=len(mesh.faces),
        )
        accepted = bool(
            candidate.is_watertight
            and candidate.is_winding_consistent
            and after.structural_health >= conservative.after.structural_health
            and metrics["symmetric_p95_relative"] <= self.max_p95_relative
            and metrics["symmetric_max_relative"] <= self.max_deviation_relative
            and metrics["relative_area_error"] <= self.max_area_error
            and (
                metrics["relative_volume_error"] is None
                or metrics["relative_volume_error"] <= self.max_volume_error
            )
        )
        if not accepted:
            return self._abstain(
                conservative,
                reason="candidate failed structural or bidirectional fidelity gates",
                checkpoint_hash=checkpoint_hash,
                metrics=metrics,
            )

        fidelity_ratio = 1.0 - min(
            1.0, metrics["symmetric_p95_relative"] / self.max_p95_relative
        )
        confidence = float(
            np.clip(0.55 * proposal.metrics.assigned_area_fraction + 0.45 * fidelity_ratio, 0, 1)
        )
        return RepairResult(
            mesh=candidate,
            repairer=self.name,
            before=conservative.before,
            after=after,
            actions=(
                *conservative.actions,
                "learned_surface_segmentation",
                "analytic_csg_reconstruction",
                "full_resolution_retessellation",
            ),
            confidence=confidence,
            abstained=False,
            metrics={
                **metrics,
                "assigned_area_fraction": proposal.metrics.assigned_area_fraction,
                "assigned_face_fraction": proposal.metrics.assigned_face_fraction,
                "output_triangles": len(candidate.faces),
            },
            checkpoint_sha256=checkpoint_hash,
        )
