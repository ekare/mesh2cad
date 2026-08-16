"""Run a trained triangle-graph model and recover surface instances."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from mesh2cad.learning.features import (
    SURFACE_CLASSES,
    TriangleGraphSample,
    triangle_features,
)
from mesh2cad.learning.partition import partition_sample
from mesh2cad.mesh import MeshData, load_mesh


def connected_components(
    face_count: int,
    adjacency: np.ndarray,
    boundary_probability: np.ndarray,
    surface_labels: np.ndarray,
    *,
    threshold: float = 0.5,
    semantic_override_threshold: float | None = 0.1,
) -> np.ndarray:
    """Join non-boundary neighbours, allowing only very confident semantic-noise repair."""
    parent = np.arange(face_count, dtype=np.int64)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for edge, probability in zip(adjacency, boundary_probability, strict=True):
        left, right = int(edge[0]), int(edge[1])
        semantic_match = surface_labels[left] == surface_labels[right]
        confident_override = (
            semantic_override_threshold is not None
            and probability < semantic_override_threshold
        )
        if probability < threshold and (semantic_match or confident_override):
            union(left, right)
    roots = np.asarray([find(face) for face in range(face_count)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64)


def predict_mesh(
    data: MeshData,
    checkpoint_path: str | Path,
    *,
    device_name: str = "auto",
    boundary_threshold: float = 0.5,
    max_core_faces: int | None = None,
    halo_hops: int | None = None,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('Inference needs `pip install -e ".[ml]"`') from exc
    from mesh2cad.learning.model import TriangleGraphNet

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TriangleGraphNet(**checkpoint["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    features, adjacency = triangle_features(data.mesh)
    partition_settings = checkpoint.get("partition", {})
    if max_core_faces is None:
        max_core_faces = partition_settings.get("max_core_faces", 50_000)
    if halo_hops is None:
        halo_hops = partition_settings.get("halo_hops", 2)
    sample = TriangleGraphSample(
        vertices=np.asarray(data.mesh.vertices, dtype=np.float32),
        faces=np.asarray(data.mesh.faces, dtype=np.int64),
        features=features,
        adjacency=adjacency,
        surface_labels=np.zeros(len(features), dtype=np.int64),
        face_instance_labels=np.zeros(len(features), dtype=np.int64),
        boundary_labels=np.zeros(len(adjacency), dtype=np.float32),
    )
    partitions = partition_sample(sample, max_core_faces=max_core_faces, halo_hops=halo_hops)
    global_surface_logits = np.empty((len(features), len(SURFACE_CLASSES)), dtype=np.float32)
    global_boundary_logits = np.empty(len(adjacency), dtype=np.float32)
    use_amp = bool(checkpoint.get("amp", False)) and device.type == "cuda"
    with torch.no_grad():
        for partition in partitions:
            with torch.autocast(device_type=device.type, enabled=use_amp):
                surface_logits, boundary_logits = model(
                    torch.as_tensor(partition.features, dtype=torch.float32, device=device),
                    torch.as_tensor(partition.adjacency, dtype=torch.long, device=device),
                )
            core_ids = partition.global_node_ids[partition.core_node_mask]
            owned_edge_ids = partition.global_edge_ids[partition.edge_loss_mask]
            global_surface_logits[core_ids] = (
                surface_logits[partition.core_node_mask].float().cpu().numpy()
            )
            global_boundary_logits[owned_edge_ids] = (
                boundary_logits[partition.edge_loss_mask].float().cpu().numpy()
            )
    surface_labels = global_surface_logits.argmax(axis=1).astype(np.int64)
    clipped_boundary_logits = np.clip(global_boundary_logits, -80.0, 80.0)
    boundary_probability = 1.0 / (1.0 + np.exp(-clipped_boundary_logits))
    instance_labels = connected_components(
        len(data.mesh.faces),
        adjacency,
        boundary_probability,
        surface_labels,
        threshold=boundary_threshold,
    )
    return {
        "mesh": data.mesh,
        "adjacency": adjacency,
        "surface_labels": surface_labels,
        "boundary_probability": boundary_probability,
        "instance_labels": instance_labels,
        "device": str(device),
        "partitions": len(partitions),
    }


def predict(
    input_path: str | Path,
    checkpoint_path: str | Path,
    *,
    device_name: str = "auto",
    boundary_threshold: float = 0.5,
    max_core_faces: int | None = None,
    halo_hops: int | None = None,
) -> dict[str, Any]:
    """Load a mesh and run the neural triangle segmentation checkpoint."""
    return predict_mesh(
        load_mesh(input_path),
        checkpoint_path,
        device_name=device_name,
        boundary_threshold=boundary_threshold,
        max_core_faces=max_core_faces,
        halo_hops=halo_hops,
    )


def save_prediction(prediction: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mesh = prediction["mesh"]
    np.savez_compressed(
        destination,
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        adjacency=prediction["adjacency"],
        surface_labels=prediction["surface_labels"],
        boundary_probability=prediction["boundary_probability"],
        instance_labels=prediction["instance_labels"],
        surface_classes=np.asarray(SURFACE_CLASSES),
    )
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict analytic surface regions on an STL")
    parser.add_argument("input", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--boundary-threshold", type=float, default=0.5)
    parser.add_argument("--max-core-faces", type=int)
    parser.add_argument("--halo-hops", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        prediction = predict(
            args.input,
            args.checkpoint,
            device_name=args.device,
            boundary_threshold=args.boundary_threshold,
            max_core_faces=args.max_core_faces,
            halo_hops=args.halo_hops,
        )
        output = save_prediction(prediction, args.output)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"mesh2cad-predict: error: {exc}", file=sys.stderr)
        return 2
    summary = {
        "output": str(output.resolve()),
        "device": prediction["device"],
        "triangles": len(prediction["surface_labels"]),
        "instances": int(np.max(prediction["instance_labels"])) + 1,
        "partitions": prediction["partitions"],
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
