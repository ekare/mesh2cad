"""Public orchestration API for mesh analysis and STEP conversion."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Literal

from mesh2cad.export import export_step
from mesh2cad.mesh import load_mesh
from mesh2cad.metrics import analytic_surface_metrics, shape_fidelity
from mesh2cad.segmentation import SegmentationConfig, segment_surfaces
from mesh2cad.topology import reconstruct_brep
from mesh2cad.types import ConversionResult

SegmentationMode = Literal["auto", "deterministic", "neural"]


def _segmentation_quality(data, segmentation) -> float:
    metrics = analytic_surface_metrics(data, segmentation)
    relative_rmse = metrics.analytic_surface_rmse / data.scale
    return (
        0.75 * metrics.assigned_area_fraction
        + 0.25 * metrics.assigned_face_fraction
        - 0.1 * min(relative_rmse, 1.0)
    )


def _select_segmentation(
    data,
    *,
    config: SegmentationConfig | None,
    checkpoint_path: str | Path | None,
    device_name: str,
    boundary_threshold: float,
    segmentation_mode: SegmentationMode,
):
    if segmentation_mode not in {"auto", "deterministic", "neural"}:
        raise ValueError(f"Unsupported segmentation mode: {segmentation_mode}")
    if segmentation_mode == "neural" and checkpoint_path is None:
        raise ValueError("Neural segmentation mode requires a checkpoint")

    if segmentation_mode == "neural":
        from mesh2cad.learning.segment import segment_with_model

        return (
            segment_with_model(
                data,
                checkpoint_path,
                config=config,
                device_name=device_name,
                boundary_threshold=boundary_threshold,
            ),
            "neural",
        )

    deterministic = segment_surfaces(data, config)
    if checkpoint_path is None or segmentation_mode == "deterministic":
        return deterministic, "deterministic"

    deterministic_metrics = analytic_surface_metrics(data, deterministic)
    if (
        deterministic_metrics.assigned_area_fraction >= 0.98
        and deterministic_metrics.assigned_face_fraction >= 0.98
    ):
        return deterministic, "deterministic-high-confidence"

    from mesh2cad.learning.segment import segment_with_model

    neural = segment_with_model(
        data,
        checkpoint_path,
        config=config,
        device_name=device_name,
        boundary_threshold=boundary_threshold,
    )
    if _segmentation_quality(data, neural) > _segmentation_quality(data, deterministic):
        return neural, "neural"
    return deterministic, "deterministic-selected-over-neural"


def _write_report(result: ConversionResult, destination: str | Path) -> Path:
    report_path = Path(destination)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return report_path


def analyze(
    input_path: str | Path,
    *,
    config: SegmentationConfig | None = None,
    checkpoint_path: str | Path | None = None,
    device_name: str = "auto",
    boundary_threshold: float = 0.5,
    segmentation_mode: SegmentationMode = "auto",
) -> ConversionResult:
    """Load and segment a mesh without constructing a BRep."""
    source_argument = Path(input_path)
    source = source_argument.resolve()
    data = load_mesh(source)
    segmentation, segmentation_engine = _select_segmentation(
        data,
        config=config,
        checkpoint_path=checkpoint_path,
        device_name=device_name,
        boundary_threshold=boundary_threshold,
        segmentation_mode=segmentation_mode,
    )
    metrics = analytic_surface_metrics(data, segmentation)
    return ConversionResult(
        input_path=str(source_argument),
        output_path=None,
        report_path=None,
        mesh_vertices=len(data.mesh.vertices),
        mesh_faces=len(data.mesh.faces),
        mesh_bounds=data.mesh.bounds.copy(),
        segmentation=segmentation,
        metrics=metrics,
        mesh_quality=data.quality,
        segmentation_engine=segmentation_engine,
    )


def convert(
    input_path: str | Path,
    output_path: str | Path,
    *,
    report_path: str | Path | None = None,
    config: SegmentationConfig | None = None,
    checkpoint_path: str | Path | None = None,
    device_name: str = "auto",
    boundary_threshold: float = 0.5,
    segmentation_mode: SegmentationMode = "auto",
) -> ConversionResult:
    """Run neural-assisted or deterministic segmentation and verified STEP reconstruction."""
    source_argument = Path(input_path)
    destination_argument = Path(output_path)
    source = source_argument.resolve()
    destination = destination_argument.resolve()
    data = load_mesh(source)
    segmentation, segmentation_engine = _select_segmentation(
        data,
        config=config,
        checkpoint_path=checkpoint_path,
        device_name=device_name,
        boundary_threshold=boundary_threshold,
        segmentation_mode=segmentation_mode,
    )
    topology = reconstruct_brep(data, segmentation)
    export_step(topology.shape, destination)
    fidelity, comparison_preview = shape_fidelity(data, topology.shape)
    metrics = analytic_surface_metrics(
        data,
        segmentation,
        brep_distances=topology.vertex_distances,
        fidelity=fidelity,
        fidelity_tessellation_sampled=(
            comparison_preview.shown_triangles < comparison_preview.total_triangles
        ),
    )
    result = ConversionResult(
        input_path=str(source_argument),
        output_path=str(destination_argument),
        report_path=None,
        mesh_vertices=len(data.mesh.vertices),
        mesh_faces=len(data.mesh.faces),
        mesh_bounds=data.mesh.bounds.copy(),
        segmentation=segmentation,
        metrics=metrics,
        mesh_quality=data.quality,
        segmentation_engine=segmentation_engine,
        topology=topology.summary,
        _shape=topology.shape,
        _comparison_preview=comparison_preview,
    )
    if report_path is not None:
        report_argument = Path(report_path)
        result = replace(result, report_path=str(report_argument))
        _write_report(result, report_argument.resolve())
    return result
