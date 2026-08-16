"""Command-line interfaces for conversion and the synthetic demo."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from mesh2cad import __version__
from mesh2cad.model_resources import default_checkpoint, default_segmentation_engine
from mesh2cad.pipeline import convert
from mesh2cad.segmentation import SegmentationConfig
from mesh2cad.synthetic import write_cube_with_hole


def _parser() -> argparse.ArgumentParser:
    checkpoint = default_checkpoint()
    parser = argparse.ArgumentParser(
        prog="mesh2cad",
        description="Reconstruct a simple mechanical triangle mesh as an analytic STEP solid.",
    )
    parser.add_argument("input", type=Path, help="input STL mesh")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output STEP path")
    parser.add_argument(
        "--report",
        type=Path,
        help="metrics JSON path (default: OUTPUT.metrics.json)",
    )
    parser.add_argument(
        "--plane-angle",
        type=float,
        default=1.0,
        help="coplanar face-normal tolerance in degrees (default: 1.0)",
    )
    parser.add_argument(
        "--cylinder-angle",
        type=float,
        default=35.0,
        help="maximum neighbouring cylinder facet angle in degrees (default: 35.0)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=checkpoint,
        help="trained checkpoint; a reviewed model may already be bundled",
    )
    parser.add_argument(
        "--segmentation-engine",
        choices=("auto", "deterministic", "neural"),
        default=default_segmentation_engine(checkpoint),
        help=(
            "surface proposer: auto compares verified candidates, deterministic ignores the "
            "checkpoint, neural uses the checkpoint directly (default: auto)"
        ),
    )
    parser.add_argument("--device", default="auto", help="neural device: auto, cuda, or cpu")
    parser.add_argument(
        "--boundary-threshold",
        type=float,
        default=0.5,
        help="predicted patch-boundary probability threshold (default: 0.5)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = args.report or args.output.with_suffix(".metrics.json")
    config = SegmentationConfig(
        plane_normal_tolerance_degrees=args.plane_angle,
        cylinder_neighbour_angle_degrees=args.cylinder_angle,
    )
    try:
        result = convert(
            args.input,
            args.output,
            report_path=report,
            config=config,
            checkpoint_path=args.checkpoint,
            device_name=args.device,
            boundary_threshold=args.boundary_threshold,
            segmentation_mode=args.segmentation_engine,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"mesh2cad: error: {exc}", file=sys.stderr)
        return 2

    print(f"STEP: {result.output_path}")
    print(f"Report: {result.report_path}")
    print(
        "Surfaces: "
        f"{len(result.segmentation.plane_regions)} planes, "
        f"{len(result.segmentation.cylinder_regions)} cylinders"
    )
    print(
        "Analytic fit RMSE: "
        f"{result.metrics.analytic_surface_rmse:.6g} "
        f"(coverage {result.metrics.assigned_area_fraction:.2%})"
    )
    if result.metrics.brep_vertex_rmse is not None:
        print(f"BRep vertex RMSE: {result.metrics.brep_vertex_rmse:.6g}")
    if result.metrics.geometric_fidelity is not None:
        print(
            "Geometric fidelity: "
            f"{result.metrics.geometric_fidelity:.2%} "
            f"@ {result.metrics.fidelity_tolerance:.6g} model units; "
            f"P95 {result.metrics.bidirectional_p95_error:.6g}, "
            f"max {result.metrics.bidirectional_max_error:.6g}"
        )
    cad_quality = result.cad_quality()
    if cad_quality is not None:
        print(
            "CAD representation: "
            f"{cad_quality['representation']} "
            f"({cad_quality['brep_face_count']} BRep faces)"
        )
    return 0


def _demo_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mesh2cad-demo",
        description="Generate and reconstruct the cube-with-hole milestone demo.",
    )
    parser.add_argument("directory", nargs="?", type=Path, default=Path("demo-output"))
    parser.add_argument("--sections-per-side", type=int, default=24)
    return parser


def demo_main(argv: Sequence[str] | None = None) -> int:
    args = _demo_parser().parse_args(argv)
    directory = args.directory
    stl_path = directory / "cube_with_hole.stl"
    step_path = directory / "cube_with_hole.step"
    report_path = directory / "cube_with_hole.metrics.json"
    write_cube_with_hole(stl_path, sections_per_side=args.sections_per_side)
    return main([str(stl_path), "-o", str(step_path), "--report", str(report_path)])


if __name__ == "__main__":
    raise SystemExit(main())
