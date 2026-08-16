import json
from threading import Event

import numpy as np
import pytest
import trimesh

torch = pytest.importorskip("torch")

from mesh2cad.learning.features import (  # noqa: E402
    FEATURE_NAMES,
    SURFACE_CLASSES,
    build_sample,
    save_sample,
)
from mesh2cad.learning.model import TriangleGraphNet  # noqa: E402
from mesh2cad.learning.train import _cpu_cuda_rng_states, _selection_score, train  # noqa: E402


def test_triangle_graph_network_output_shapes():
    model = TriangleGraphNet(hidden_size=16, layers=2)
    features = torch.randn(5, len(FEATURE_NAMES))
    adjacency = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=torch.long)

    surface_logits, boundary_logits = model(features, adjacency)

    assert surface_logits.shape == (5, len(SURFACE_CLASSES))
    assert boundary_logits.shape == (4,)


def test_checkpoint_cuda_rng_states_are_restored_from_cpu_byte_tensors():
    states = [torch.arange(8, dtype=torch.uint8)]
    if torch.cuda.is_available():
        states = [state.cuda() for state in states]

    restored = _cpu_cuda_rng_states(states)

    assert len(restored) == 1
    assert restored[0].device.type == "cpu"
    assert restored[0].dtype == torch.uint8
    assert restored[0].tolist() == list(range(8))


def test_analytic_selection_prioritizes_plane_cylinder_iou_and_boundaries():
    metrics = {
        "surface_macro_iou": 0.9,
        "supported_surface_macro_iou": 0.6,
        "boundary_f1": 0.8,
    }

    assert _selection_score(metrics, "analytic") == pytest.approx(1.4)
    assert _selection_score(metrics, "all-surfaces") == pytest.approx(1.7)


def test_requested_stop_writes_atomic_recovery_checkpoint_and_paused_telemetry(tmp_path):
    dataset = tmp_path / "prepared"
    mesh = trimesh.creation.box()
    surface_labels = np.zeros(len(mesh.faces), dtype=np.int64)
    instance_labels = np.arange(len(mesh.faces), dtype=np.int64)
    sample = build_sample(mesh, surface_labels, instance_labels)
    for index in range(3):
        save_sample(sample, dataset / f"sample-{index}.npz")

    destination = tmp_path / "recovery.pt"
    telemetry = tmp_path / "training.json"
    stop = Event()
    stop.set()
    report = train(
        dataset,
        destination,
        epochs=2,
        hidden_size=8,
        layers=1,
        device_name="cpu",
        amp=False,
        max_core_faces=100,
        max_batch_faces=1_000,
        halo_hops=1,
        balance_samples=3,
        rotation_augmentation=False,
        prefetch_batches=0,
        checkpoint_every_batches=1,
        telemetry_path=telemetry,
        stop_event=stop,
    )

    checkpoint = torch.load(destination, map_location="cpu", weights_only=False)
    status = json.loads(telemetry.read_text(encoding="utf-8"))
    assert report["interrupted"] is True
    assert checkpoint["training_complete"] is False
    assert checkpoint["progress"]["phase"] == "interrupted"
    assert checkpoint["global_batch"] == 1
    assert status["status"] == "paused"
    assert status["phase"] == "interrupted"


def test_completed_epoch_writes_path_free_candidate_snapshot(tmp_path):
    dataset = tmp_path / "prepared"
    mesh = trimesh.creation.box()
    sample = build_sample(
        mesh,
        np.zeros(len(mesh.faces), dtype=np.int64),
        np.arange(len(mesh.faces), dtype=np.int64),
    )
    for index in range(3):
        save_sample(sample, dataset / f"sample-{index}.npz")
    snapshots = tmp_path / "epochs"

    train(
        dataset,
        tmp_path / "complete.pt",
        epochs=1,
        hidden_size=8,
        layers=1,
        device_name="cpu",
        amp=False,
        max_core_faces=100,
        max_batch_faces=1_000,
        halo_hops=1,
        balance_samples=3,
        rotation_augmentation=False,
        prefetch_batches=0,
        epoch_snapshots=snapshots,
    )

    candidate_path = snapshots / "epoch-001.pt"
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
    assert candidate["checkpoint_kind"] == "mesh2cad-training-candidate"
    assert candidate["selection_metric"] == "analytic"
    assert candidate["selection_score"] == pytest.approx(
        candidate["validation"]["supported_surface_macro_iou"]
        + candidate["validation"]["boundary_f1"]
    )
    assert str(tmp_path).encode() not in candidate_path.read_bytes()
