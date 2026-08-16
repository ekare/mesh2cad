"""Training CLI for the triangle-graph segmentation baseline."""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import signal
import sys
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from queue import Full, Queue
from threading import Event, Lock, Thread
from typing import Any

import numpy as np

from mesh2cad.learning.features import FEATURE_NAMES, SURFACE_CLASSES, load_sample
from mesh2cad.learning.partition import GraphPartition, partition_sample
from mesh2cad.learning.telemetry import publish_telemetry


def _torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch.nn import functional
    except ImportError as exc:
        raise RuntimeError('Training needs `pip install -e ".[ml]"`') from exc
    return torch, functional


def _device(requested: str, torch: Any) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access it")
    return device


def _cpu_cuda_rng_states(states: Sequence[Any]) -> list[Any]:
    """Normalize checkpoint-loaded CUDA RNG states for PyTorch's restore API."""
    return [state.detach().cpu() for state in states]


def _tensor_partition(partition: GraphPartition, torch: Any, device: Any) -> tuple[Any, ...]:
    return (
        torch.as_tensor(partition.features, dtype=torch.float32, device=device),
        torch.as_tensor(partition.adjacency, dtype=torch.long, device=device),
        torch.as_tensor(partition.surface_labels, dtype=torch.long, device=device),
        torch.as_tensor(partition.boundary_labels, dtype=torch.float32, device=device),
        torch.as_tensor(partition.core_node_mask, dtype=torch.bool, device=device),
        torch.as_tensor(partition.edge_loss_mask, dtype=torch.bool, device=device),
    )


def _merge_partitions(partitions: Sequence[GraphPartition]) -> GraphPartition:
    """Pack independent graph partitions into one disconnected training graph."""
    if not partitions:
        raise ValueError("cannot merge an empty partition collection")
    node_offsets = np.cumsum([0, *(len(partition.features) for partition in partitions[:-1])])
    adjacency = np.concatenate(
        [
            partition.adjacency + offset
            for partition, offset in zip(partitions, node_offsets, strict=True)
        ],
        axis=0,
    )
    return GraphPartition(
        features=np.concatenate([partition.features for partition in partitions], axis=0),
        adjacency=adjacency,
        surface_labels=np.concatenate(
            [partition.surface_labels for partition in partitions], axis=0
        ),
        boundary_labels=np.concatenate(
            [partition.boundary_labels for partition in partitions], axis=0
        ),
        core_node_mask=np.concatenate(
            [partition.core_node_mask for partition in partitions], axis=0
        ),
        edge_loss_mask=np.concatenate(
            [partition.edge_loss_mask for partition in partitions], axis=0
        ),
        # Training does not consume global IDs. Keep unique packed IDs for diagnostics/tests.
        global_node_ids=np.arange(
            sum(len(partition.features) for partition in partitions), dtype=np.int64
        ),
        global_edge_ids=np.arange(
            sum(len(partition.adjacency) for partition in partitions), dtype=np.int64
        ),
    )


def _random_rotation_matrix() -> np.ndarray:
    """Sample a right-handed orthonormal matrix from the global seeded NumPy RNG."""
    matrix, upper = np.linalg.qr(np.random.normal(size=(3, 3)))
    signs = np.sign(np.diag(upper))
    signs[signs == 0.0] = 1.0
    matrix *= signs
    if np.linalg.det(matrix) < 0.0:
        matrix[:, 0] *= -1.0
    return matrix


def _rotate_partition_features(
    partition: GraphPartition, rotation: np.ndarray
) -> GraphPartition:
    """Rotate centroid and normal features without altering graph geometry or labels."""
    rotation_array = np.asarray(rotation, dtype=np.float64)
    if rotation_array.shape != (3, 3):
        raise ValueError("rotation must be a 3x3 matrix")
    features = partition.features.copy()
    features[:, 0:3] = features[:, 0:3] @ rotation_array.T
    features[:, 3:6] = features[:, 3:6] @ rotation_array.T
    return replace(partition, features=features)


def _training_batches(
    files: Sequence[Path],
    *,
    max_core_faces: int | None,
    halo_hops: int,
    max_batch_faces: int,
    rotation_augmentation: bool = False,
    sample_workers: int = 1,
    sample_loader: Callable[..., tuple[GraphPartition, ...]] | None = None,
) -> Iterator[GraphPartition]:
    """Stream lossless mesh partitions and pack them without crossing mesh edges."""
    if max_batch_faces < 1:
        raise ValueError("max_batch_faces must be positive")
    pending: list[GraphPartition] = []
    pending_faces = 0
    for sample_partitions in _ordered_sample_partitions(
        files,
        max_core_faces=max_core_faces,
        halo_hops=halo_hops,
        workers=sample_workers,
        loader=sample_loader,
    ):
        rotation = _random_rotation_matrix() if rotation_augmentation else None
        for partition in sample_partitions:
            if rotation is not None:
                partition = _rotate_partition_features(partition, rotation)
            partition_faces = len(partition.features)
            if pending and pending_faces + partition_faces > max_batch_faces:
                yield _merge_partitions(pending)
                pending = []
                pending_faces = 0
            pending.append(partition)
            pending_faces += partition_faces
            # An individual partition can be larger than the packing target because halos are
            # lossless context. Yield it alone; never truncate it.
            if pending_faces >= max_batch_faces:
                yield _merge_partitions(pending)
                pending = []
                pending_faces = 0
    if pending:
        yield _merge_partitions(pending)


def _ordered_sample_partitions(
    files: Sequence[Path],
    *,
    max_core_faces: int | None,
    halo_hops: int,
    workers: int,
    loader: Callable[..., tuple[GraphPartition, ...]] | None = None,
) -> Iterator[tuple[GraphPartition, ...]]:
    """Load a bounded number of samples concurrently while preserving seeded file order."""
    if workers < 1:
        raise ValueError("sample_workers must be positive")
    resolved_loader = loader or _sample_partitions
    if workers == 1:
        for path in files:
            yield resolved_loader(
                path,
                max_core_faces=max_core_faces,
                halo_hops=halo_hops,
            )
        return

    def load(path: Path) -> tuple[GraphPartition, ...]:
        return resolved_loader(
            path,
            max_core_faces=max_core_faces,
            halo_hops=halo_hops,
        )

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mesh2cad-sample") as pool:
        iterator = iter(files)
        pending = deque()
        for _ in range(workers):
            try:
                pending.append(pool.submit(load, next(iterator)))
            except StopIteration:
                break
        while pending:
            future = pending.popleft()
            yield future.result()
            try:
                pending.append(pool.submit(load, next(iterator)))
            except StopIteration:
                pass


class SamplePartitionCache:
    """Bounded process-local LRU for decoded/partitioned samples across epochs."""

    def __init__(self, max_bytes: int) -> None:
        if max_bytes < 0:
            raise ValueError("sample cache size cannot be negative")
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self.hits = 0
        self.misses = 0
        self._items: OrderedDict[Path, tuple[tuple[GraphPartition, ...], int]] = OrderedDict()
        self._lock = Lock()

    @staticmethod
    def _partition_bytes(partitions: tuple[GraphPartition, ...]) -> int:
        return sum(
            array.nbytes
            for partition in partitions
            for array in (
                partition.features,
                partition.adjacency,
                partition.surface_labels,
                partition.boundary_labels,
                partition.core_node_mask,
                partition.edge_loss_mask,
                partition.global_node_ids,
                partition.global_edge_ids,
            )
        )

    def load(
        self,
        path: Path,
        *,
        max_core_faces: int | None,
        halo_hops: int,
    ) -> tuple[GraphPartition, ...]:
        with self._lock:
            cached = self._items.pop(path, None)
            if cached is not None:
                self._items[path] = cached
                self.hits += 1
                return cached[0]
            self.misses += 1
        partitions = _sample_partitions(
            path,
            max_core_faces=max_core_faces,
            halo_hops=halo_hops,
        )
        size = self._partition_bytes(partitions)
        if self.max_bytes == 0 or size > self.max_bytes:
            return partitions
        with self._lock:
            while self._items and self.current_bytes + size > self.max_bytes:
                _discarded_path, (_discarded, discarded_size) = self._items.popitem(last=False)
                self.current_bytes -= discarded_size
            self._items[path] = (partitions, size)
            self.current_bytes += size
        return partitions

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "max_bytes": self.max_bytes,
                "current_bytes": self.current_bytes,
                "items": len(self._items),
                "hits": self.hits,
                "misses": self.misses,
            }


def _prefetch_batches(
    batches: Iterator[GraphPartition], *, depth: int
) -> Iterator[GraphPartition]:
    """Prepare bounded CPU batches concurrently with the current accelerator step."""
    if depth < 1:
        yield from batches
        return
    queue: Queue[GraphPartition | object] = Queue(maxsize=depth)
    sentinel = object()
    stop = Event()
    errors: list[BaseException] = []

    def publish(item: GraphPartition | object) -> None:
        while not stop.is_set():
            try:
                queue.put(item, timeout=0.1)
                return
            except Full:
                continue

    def produce() -> None:
        try:
            for batch in batches:
                if stop.is_set():
                    return
                publish(batch)
        except BaseException as exc:
            errors.append(exc)
        finally:
            publish(sentinel)

    worker = Thread(target=produce, name="mesh2cad-batch-prefetch", daemon=True)
    worker.start()
    try:
        while True:
            item = queue.get()
            if item is sentinel:
                if errors:
                    raise errors[0]
                return
            if not isinstance(item, GraphPartition):
                raise RuntimeError("batch prefetch produced an invalid item")
            yield item
    finally:
        stop.set()
        worker.join(timeout=1.0)


def _sample_partitions(
    path: Path, *, max_core_faces: int | None, halo_hops: int
) -> tuple[GraphPartition, ...]:
    return partition_sample(load_sample(path), max_core_faces=max_core_faces, halo_hops=halo_hops)


def _dataset_files(
    datasets: str | Path | Sequence[str | Path],
    *,
    seed: int,
    max_samples_per_source: int | None,
) -> tuple[list[Path], list[str]]:
    roots = [datasets] if isinstance(datasets, (str, Path)) else list(datasets)
    if not roots:
        raise ValueError("at least one dataset directory is required")
    if max_samples_per_source is not None and max_samples_per_source < 1:
        raise ValueError("max_samples_per_source must be positive or None")
    selected: list[Path] = []
    resolved_roots: list[str] = []
    for index, value in enumerate(roots):
        root = Path(value)
        files = sorted(root.glob("*.npz"))
        if not files:
            raise RuntimeError(f"No .npz samples found in {root}")
        if max_samples_per_source is not None and len(files) > max_samples_per_source:
            source_random = random.Random(seed + index * 1_000_003)
            files = source_random.sample(files, max_samples_per_source)
        selected.extend(files)
        resolved_roots.append(str(root.resolve()))
    return selected, resolved_roots


def _split_files(
    files: list[Path],
    *,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[list[Path], list[Path], list[Path]]:
    if validation_fraction < 0.0 or test_fraction < 0.0:
        raise ValueError("split fractions cannot be negative")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation and test fractions must sum to less than one")
    if len(files) < 3:
        return files, [], []
    validation_count = max(1, int(round(validation_fraction * len(files))))
    test_count = max(1, int(round(test_fraction * len(files))))
    while validation_count + test_count >= len(files):
        if test_count > 1:
            test_count -= 1
        else:
            validation_count -= 1
    test_files = files[:test_count]
    validation_files = files[test_count : test_count + validation_count]
    training_files = files[test_count + validation_count :]
    return training_files, validation_files, test_files


def _split_files_from_manifest(
    files: list[Path], manifest: str | Path
) -> tuple[list[Path], list[Path], list[Path]]:
    manifest_path = Path(manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    partitions = payload.get("partitions", payload)
    if not isinstance(partitions, dict):
        raise ValueError("split manifest must contain a partitions object")
    aliases = {
        "train": ("train", "training"),
        "validation": ("validation", "val"),
        "test": ("test",),
    }
    available: dict[str, Path] = {}
    for path in files:
        key = path.stem.casefold()
        if key in available:
            raise ValueError(f"duplicate prepared sample identifier: {path.stem}")
        available[key] = path

    selected: dict[str, list[Path]] = {}
    owners: dict[str, str] = {}
    for canonical, names in aliases.items():
        entries: Any = []
        for name in names:
            if name in partitions:
                entries = partitions[name]
                break
        if not isinstance(entries, list):
            raise ValueError(f"split manifest partition {canonical!r} must be a list")
        resolved: list[Path] = []
        for entry in entries:
            key = Path(str(entry)).stem.casefold()
            previous = owners.get(key)
            if previous is not None and previous != canonical:
                raise ValueError(
                    "split manifest leaks "
                    f"{Path(str(entry)).name} across {previous} and {canonical}"
                )
            if previous is None:
                owners[key] = canonical
                path = available.get(key)
                if path is not None:
                    resolved.append(path)
        selected[canonical] = resolved
    orphaned = sorted(set(available) - set(owners))
    if orphaned:
        preview = ", ".join(orphaned[:5])
        suffix = "..." if len(orphaned) > 5 else ""
        raise ValueError(f"prepared samples are absent from split manifest: {preview}{suffix}")
    if not selected["train"]:
        raise ValueError("split manifest contains no training samples")
    return selected["train"], selected["validation"], selected["test"]


def _balancing_weights(
    files: list[Path], *, max_samples: int | None = 4_096, seed: int = 17
) -> tuple[np.ndarray, float, dict[str, Any]]:
    scanned_files = files
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("balance_samples must be positive or None")
        if len(files) > max_samples:
            scanned_files = random.Random(seed).sample(files, max_samples)
    class_counts = np.zeros(len(SURFACE_CLASSES), dtype=np.int64)
    positive_boundaries = 0
    boundary_count = 0
    for path in scanned_files:
        with np.load(path, allow_pickle=False) as sample:
            class_counts += np.bincount(sample["surface_labels"], minlength=len(SURFACE_CLASSES))
            positive_boundaries += int(sample["boundary_labels"].sum())
            boundary_count += len(sample["boundary_labels"])

    total = max(1, int(class_counts.sum()))
    class_weights = np.sqrt(total / (len(SURFACE_CLASSES) * np.maximum(class_counts, 1)))
    class_weights[class_counts == 0] = 0.0
    nonzero = class_weights > 0.0
    class_weights[nonzero] /= class_weights[nonzero].mean()
    class_weights = np.clip(class_weights, 0.1, 10.0)
    negative_boundaries = boundary_count - positive_boundaries
    boundary_positive_weight = (
        negative_boundaries / positive_boundaries if positive_boundaries else 1.0
    )
    boundary_positive_weight = float(np.clip(boundary_positive_weight, 1.0, 30.0))
    statistics = {
        "surface_triangle_counts": dict(zip(SURFACE_CLASSES, class_counts.tolist(), strict=True)),
        "surface_class_weights": dict(zip(SURFACE_CLASSES, class_weights.tolist(), strict=True)),
        "boundary_positive_count": positive_boundaries,
        "boundary_negative_count": negative_boundaries,
        "boundary_positive_weight": boundary_positive_weight,
        "balancing_samples_scanned": len(scanned_files),
    }
    return class_weights.astype(np.float32), boundary_positive_weight, statistics


def _evaluate(
    model: Any,
    files: list[Path],
    *,
    torch: Any,
    device: Any,
    use_amp: bool,
    max_core_faces: int | None,
    halo_hops: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    if not files:
        return {}
    model.eval()
    confusion = torch.zeros(
        (len(SURFACE_CLASSES), len(SURFACE_CLASSES)), dtype=torch.int64, device=device
    )
    true_positive = false_positive = false_negative = 0
    with torch.no_grad():
        for file_index, path in enumerate(files, start=1):
            for partition in _sample_partitions(
                path, max_core_faces=max_core_faces, halo_hops=halo_hops
            ):
                (
                    features,
                    adjacency,
                    surface_labels,
                    boundary_labels,
                    core_node_mask,
                    edge_loss_mask,
                ) = _tensor_partition(partition, torch, device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    surface_logits, boundary_logits = model(features, adjacency)
                prediction = surface_logits[core_node_mask].argmax(dim=1)
                truth = surface_labels[core_node_mask]
                flat = truth * len(SURFACE_CLASSES) + prediction
                confusion += torch.bincount(flat, minlength=len(SURFACE_CLASSES) ** 2).reshape(
                    len(SURFACE_CLASSES), len(SURFACE_CLASSES)
                )
                boundary_prediction = boundary_logits[edge_loss_mask].sigmoid() >= 0.5
                boundary_truth = boundary_labels[edge_loss_mask] >= 0.5
                true_positive += int((boundary_prediction & boundary_truth).sum().cpu())
                false_positive += int((boundary_prediction & ~boundary_truth).sum().cpu())
                false_negative += int((~boundary_prediction & boundary_truth).sum().cpu())
            if progress_callback is not None and (
                file_index == len(files) or file_index % 25 == 0
            ):
                progress_callback(file_index, len(files))

    matrix = confusion.cpu().numpy()
    true_by_class = np.diag(matrix)
    truth_count = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    union = truth_count + predicted_count - true_by_class
    class_iou = np.divide(
        true_by_class,
        union,
        out=np.zeros_like(true_by_class, dtype=np.float64),
        where=union > 0,
    )
    class_recall = np.divide(
        true_by_class,
        truth_count,
        out=np.zeros_like(true_by_class, dtype=np.float64),
        where=truth_count > 0,
    )
    present = truth_count > 0
    supported_indices = np.asarray(
        [SURFACE_CLASSES.index("plane"), SURFACE_CLASSES.index("cylinder")],
        dtype=np.int64,
    )
    supported_present = present[supported_indices]
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    boundary_f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "surface_accuracy": float(true_by_class.sum() / max(1, matrix.sum())),
        "surface_macro_iou": float(class_iou[present].mean()) if np.any(present) else 0.0,
        "supported_surface_macro_iou": (
            float(class_iou[supported_indices][supported_present].mean())
            if np.any(supported_present)
            else 0.0
        ),
        "supported_surface_accuracy": float(
            true_by_class[supported_indices].sum()
            / max(1, truth_count[supported_indices].sum())
        ),
        "surface_iou": dict(zip(SURFACE_CLASSES, class_iou.tolist(), strict=True)),
        "surface_recall": dict(zip(SURFACE_CLASSES, class_recall.tolist(), strict=True)),
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": boundary_f1,
    }


def _selection_score(metrics: dict[str, Any], metric: str) -> float:
    if metric == "analytic":
        surface_score = metrics.get("supported_surface_macro_iou")
    elif metric == "all-surfaces":
        surface_score = metrics.get("surface_macro_iou")
    else:
        raise ValueError(f"unknown model-selection metric: {metric}")
    if surface_score is None or metrics.get("boundary_f1") is None:
        raise ValueError(f"validation metrics are incomplete for {metric!r} selection")
    return float(surface_score) + float(metrics["boundary_f1"])


def train(
    dataset: str | Path | Sequence[str | Path],
    output: str | Path,
    *,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    hidden_size: int = 96,
    layers: int = 4,
    device_name: str = "auto",
    seed: int = 17,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    amp: bool = True,
    max_core_faces: int | None = 50_000,
    max_batch_faces: int = 160_000,
    halo_hops: int = 2,
    split_manifest: str | Path | None = None,
    max_samples_per_source: int | None = None,
    balance_samples: int | None = 4_096,
    rotation_augmentation: bool = True,
    prefetch_batches: int = 2,
    sample_workers: int = 2,
    sample_cache_gib: float = 0.0,
    selection_metric: str = "analytic",
    epoch_snapshots: str | Path | None = None,
    resume: bool = False,
    checkpoint_every_batches: int = 25,
    telemetry_path: str | Path | None = None,
    stop_event: Event | None = None,
) -> dict[str, Any]:
    torch, functional = _torch()
    from mesh2cad.learning.model import TriangleGraphNet

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    files, dataset_roots = _dataset_files(
        dataset,
        seed=seed,
        max_samples_per_source=max_samples_per_source,
    )
    if split_manifest is None:
        random.shuffle(files)
        training_files, validation_files, test_files = _split_files(
            files,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
    else:
        training_files, validation_files, test_files = _split_files_from_manifest(
            files, split_manifest
        )
    device = _device(device_name, torch)
    use_amp = amp and device.type == "cuda"
    telemetry_started = time.time()
    telemetry_destination = None if telemetry_path is None else Path(telemetry_path)
    recent_losses: deque[dict[str, float | int]] = deque(maxlen=240)
    publish_telemetry(
        telemetry_destination,
        {
            "status": "running",
            "phase": "setup",
            "device": str(device),
            "amp": use_amp,
            "epochs": epochs,
            "training_samples": len(training_files),
            "validation_samples": len(validation_files),
            "test_samples": len(test_files),
            "rotation_augmentation": rotation_augmentation,
            "checkpoint": str(Path(output).resolve()),
        },
    )
    class_weights_array, boundary_weight_value, dataset_statistics = _balancing_weights(
        training_files, max_samples=balance_samples, seed=seed
    )
    class_weights = torch.as_tensor(class_weights_array, device=device)
    boundary_positive_weight = torch.tensor(boundary_weight_value, device=device)
    model = TriangleGraphNet(hidden_size=hidden_size, layers=layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    history: list[dict[str, Any]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_score = -1.0
    best_epoch = 0
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    if checkpoint_every_batches < 0:
        raise ValueError("checkpoint_every_batches cannot be negative")
    if prefetch_batches < 0:
        raise ValueError("prefetch_batches cannot be negative")
    if sample_workers < 1:
        raise ValueError("sample_workers must be positive")
    if sample_cache_gib < 0.0:
        raise ValueError("sample_cache_gib cannot be negative")
    if selection_metric not in {"analytic", "all-surfaces"}:
        raise ValueError("selection_metric must be 'analytic' or 'all-surfaces'")
    sample_cache = SamplePartitionCache(int(sample_cache_gib * 1024**3))
    snapshot_directory = Path(epoch_snapshots) if epoch_snapshots is not None else None

    if resume:
        if not destination.is_file():
            raise RuntimeError(f"Cannot resume; checkpoint does not exist: {destination}")
        checkpoint = torch.load(destination, map_location=device, weights_only=False)
        expected_model = {"hidden_size": hidden_size, "layers": layers}
        if checkpoint.get("model") != expected_model:
            raise ValueError(
                "Resume model mismatch: "
                f"checkpoint={checkpoint.get('model')}, requested={expected_model}"
            )
        if tuple(checkpoint.get("surface_classes", ())) != tuple(SURFACE_CLASSES):
            raise ValueError("Resume surface classes do not match this mesh2cad version")
        model.load_state_dict(checkpoint.get("current_model_state", checkpoint["model_state"]))
        best_state = checkpoint["model_state"]
        history = list(checkpoint.get("history", []))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        checkpoint_selection_metric = checkpoint.get("selection_metric")
        if (
            checkpoint_selection_metric is not None
            and checkpoint_selection_metric != selection_metric
        ):
            raise ValueError(
                "Resume model-selection metric mismatch: "
                f"checkpoint={checkpoint_selection_metric!r}, requested={selection_metric!r}"
            )
        if best_epoch > 0 and best_epoch <= len(history):
            best_score = _selection_score(
                history[best_epoch - 1]["validation"], selection_metric
            )
        else:
            best_score = -1.0
        start_epoch = len(history)
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        if "training_files" in checkpoint:
            training_files = [Path(path) for path in checkpoint["training_files"]]
            validation_files = [Path(path) for path in checkpoint["validation_files"]]
            test_files = [Path(path) for path in checkpoint["test_files"]]
        if "python_random_state" in checkpoint:
            random.setstate(checkpoint["python_random_state"])
        if "numpy_random_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_random_state"])
        if "torch_random_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_random_state"].cpu())
        if device.type == "cuda" and "torch_cuda_random_state" in checkpoint:
            torch.cuda.set_rng_state_all(
                _cpu_cuda_rng_states(checkpoint["torch_cuda_random_state"])
            )

    global_batch = sum(int(record.get("training_partitions", 0)) for record in history)
    if resume:
        global_batch = int(checkpoint.get("global_batch", global_batch))
        recent_losses.extend(checkpoint.get("recent_losses", ()))

    def publish_training_state(
        *,
        status: str,
        phase: str,
        epoch: int | None = None,
        batch: int | None = None,
        current_loss: float | None = None,
        mean_loss: float | None = None,
        validation: dict[str, Any] | None = None,
        test: dict[str, Any] | None = None,
        evaluation_progress: dict[str, int] | None = None,
    ) -> None:
        latest_validation = validation
        if latest_validation is None and history:
            latest_validation = history[-1].get("validation")
        best_validation = (
            history[best_epoch - 1].get("validation")
            if best_epoch > 0 and best_epoch <= len(history)
            else latest_validation
        )
        publish_telemetry(
            telemetry_destination,
            {
                "status": status,
                "phase": phase,
                "device": str(device),
                "amp": use_amp,
                "epoch": epoch,
                "epochs": epochs,
                "batch": batch,
                "global_batch": global_batch,
                "current_loss": current_loss,
                "mean_loss": mean_loss,
                "rotation_augmentation": rotation_augmentation,
                "recent_losses": list(recent_losses),
                "training_samples": len(training_files),
                "validation_samples": len(validation_files),
                "test_samples": len(test_files),
                "best_epoch": best_epoch,
                "best_score": best_score,
                "selection_metric": selection_metric,
                "validation": latest_validation,
                "best_validation": best_validation,
                "test": test,
                "evaluation_progress": evaluation_progress,
                "batches_per_epoch_estimate": (
                    int(history[-1]["training_partitions"]) if history else None
                ),
                "checkpoint": str(destination.resolve()),
                "partition": {
                    "max_core_faces": max_core_faces,
                    "max_batch_faces": max_batch_faces,
                    "halo_hops": halo_hops,
                    "prefetch_batches": prefetch_batches,
                    "sample_workers": sample_workers,
                    "sample_cache_gib": sample_cache_gib,
                },
                "sample_cache": sample_cache.stats(),
                "elapsed_seconds": time.time() - telemetry_started,
            },
        )

    def save_checkpoint(
        test_metrics: dict[str, Any],
        *,
        training_complete: bool,
        progress: dict[str, Any],
    ) -> None:
        temporary = destination.with_name(f".{destination.name}.tmp")
        backup = destination.with_name(f"{destination.name}.bak")
        backup_temporary = destination.with_name(f".{destination.name}.bak.tmp")
        torch.save(
            {
                "model_state": best_state,
                "current_model_state": model.state_dict(),
                "model": {"hidden_size": hidden_size, "layers": layers},
                "surface_classes": SURFACE_CLASSES,
                "feature_names": FEATURE_NAMES,
                "history": history,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "selection_metric": selection_metric,
                "global_batch": global_batch,
                "recent_losses": list(recent_losses),
                "test": test_metrics,
                "dataset_statistics": dataset_statistics,
                "device_used": str(device),
                "amp": use_amp,
                "partition": {
                    "max_core_faces": max_core_faces,
                    "max_batch_faces": max_batch_faces,
                    "halo_hops": halo_hops,
                    "prefetch_batches": prefetch_batches,
                    "sample_workers": sample_workers,
                    "sample_cache_gib": sample_cache_gib,
                },
                "seed": seed,
                "dataset_roots": dataset_roots,
                "max_samples_per_source": max_samples_per_source,
                "augmentation": {"random_rotation": rotation_augmentation},
                "split_manifest": (
                    str(Path(split_manifest).resolve()) if split_manifest else None
                ),
                "training_complete": training_complete,
                "progress": progress,
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "training_files": [str(path) for path in training_files],
                "validation_files": [str(path) for path in validation_files],
                "test_files": [str(path) for path in test_files],
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.get_rng_state(),
                "torch_cuda_random_state": (
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                ),
            },
            temporary,
        )
        # Preserve the previous known-good checkpoint before publishing the new one. Both
        # promotions are same-directory atomic replacements, so a power loss cannot expose a
        # half-written checkpoint at the public path.
        if destination.is_file():
            shutil.copy2(destination, backup_temporary)
            backup_temporary.replace(backup)
        temporary.replace(destination)

    def save_epoch_snapshot(record: dict[str, Any]) -> None:
        if snapshot_directory is None:
            return
        snapshot_directory.mkdir(parents=True, exist_ok=True)
        epoch_number = int(record["epoch"])
        snapshot = snapshot_directory / f"epoch-{epoch_number:03d}.pt"
        temporary = snapshot.with_suffix(snapshot.suffix + ".part")
        torch.save(
            {
                "checkpoint_kind": "mesh2cad-training-candidate",
                "schema_version": 1,
                "model_state": model.state_dict(),
                "model": {"hidden_size": hidden_size, "layers": layers},
                "surface_classes": SURFACE_CLASSES,
                "feature_names": FEATURE_NAMES,
                "epoch": epoch_number,
                "train_loss": record["train_loss"],
                "validation": record["validation"],
                "selection_metric": selection_metric,
                "selection_score": record["selection_score"],
                "training_samples": len(training_files),
                "validation_samples": len(validation_files),
                "test_samples": len(test_files),
            },
            temporary,
        )
        temporary.replace(snapshot)

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        partition_count = 0
        random.shuffle(training_files)
        batch_iterator = _training_batches(
            training_files,
            max_core_faces=max_core_faces,
            halo_hops=halo_hops,
            max_batch_faces=max_batch_faces,
            rotation_augmentation=rotation_augmentation,
            sample_workers=sample_workers,
            sample_loader=sample_cache.load,
        )
        for partition in _prefetch_batches(batch_iterator, depth=prefetch_batches):
            (
                features,
                adjacency,
                surface_labels,
                boundary_labels,
                core_node_mask,
                edge_loss_mask,
            ) = _tensor_partition(partition, torch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                surface_logits, boundary_logits = model(features, adjacency)
                semantic_loss = functional.cross_entropy(
                    surface_logits[core_node_mask],
                    surface_labels[core_node_mask],
                    weight=class_weights,
                )
                if torch.any(edge_loss_mask):
                    boundary_loss = functional.binary_cross_entropy_with_logits(
                        boundary_logits[edge_loss_mask],
                        boundary_labels[edge_loss_mask],
                        pos_weight=boundary_positive_weight,
                    )
                else:
                    boundary_loss = semantic_loss.new_zeros(())
                loss = semantic_loss + boundary_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach().cpu())
            partition_count += 1
            global_batch += 1
            current_loss = float(loss.detach().cpu())
            recent_losses.append(
                {
                    "step": global_batch,
                    "epoch": epoch + 1,
                    "batch": partition_count,
                    "loss": current_loss,
                }
            )
            publish_training_state(
                status="running",
                phase="training",
                epoch=epoch + 1,
                batch=partition_count,
                current_loss=current_loss,
                mean_loss=epoch_loss / partition_count,
            )
            if (
                checkpoint_every_batches
                and partition_count % checkpoint_every_batches == 0
            ):
                save_checkpoint(
                    {},
                    training_complete=False,
                    progress={
                        "phase": "training",
                        "epoch": epoch + 1,
                        "completed_batches_in_epoch": partition_count,
                    },
                )
            if stop_event is not None and stop_event.is_set():
                save_checkpoint(
                    {},
                    training_complete=False,
                    progress={
                        "phase": "interrupted",
                        "epoch": epoch + 1,
                        "completed_batches_in_epoch": partition_count,
                    },
                )
                publish_training_state(
                    status="paused",
                    phase="interrupted",
                    epoch=epoch + 1,
                    batch=partition_count,
                    current_loss=current_loss,
                    mean_loss=epoch_loss / partition_count,
                )
                return {
                    "checkpoint": str(destination.resolve()),
                    "device": str(device),
                    "interrupted": True,
                    "epoch": epoch + 1,
                    "completed_batches_in_epoch": partition_count,
                    "global_batch": global_batch,
                }

        validation_targets = validation_files or training_files
        publish_training_state(
            status="running",
            phase="validation",
            epoch=epoch + 1,
            batch=partition_count,
            mean_loss=epoch_loss / partition_count,
            evaluation_progress={"completed": 0, "total": len(validation_targets)},
        )
        validation_metrics = _evaluate(
            model,
            validation_targets,
            torch=torch,
            device=device,
            use_amp=use_amp,
            max_core_faces=max_core_faces,
            halo_hops=halo_hops,
            progress_callback=lambda completed,
            total,
            current_epoch=epoch + 1,
            current_batch=partition_count,
            current_mean=epoch_loss / partition_count: publish_training_state(
                status="running",
                phase="validation",
                epoch=current_epoch,
                batch=current_batch,
                mean_loss=current_mean,
                evaluation_progress={"completed": completed, "total": total},
            ),
        )
        score = _selection_score(validation_metrics, selection_metric)
        record = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss / partition_count,
            "training_partitions": partition_count,
            "validation": validation_metrics,
            "selection_metric": selection_metric,
            "selection_score": score,
        }
        history.append(record)
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        save_epoch_snapshot(record)
        publish_training_state(
            status="running",
            phase="validation",
            epoch=epoch + 1,
            batch=partition_count,
            mean_loss=record["train_loss"],
            validation=validation_metrics,
        )
        print(json.dumps(record), flush=True)
        # Keep a usable best-so-far checkpoint after every epoch.
        save_checkpoint(
            {},
            training_complete=False,
            progress={"phase": "validation", "completed_epochs": len(history)},
        )

    model.load_state_dict(best_state)
    publish_training_state(
        status="running",
        phase="test",
        epoch=len(history),
        evaluation_progress={"completed": 0, "total": len(test_files)},
    )
    test_metrics = _evaluate(
        model,
        test_files,
        torch=torch,
        device=device,
        use_amp=use_amp,
        max_core_faces=max_core_faces,
        halo_hops=halo_hops,
        progress_callback=lambda completed, total: publish_training_state(
            status="running",
            phase="test",
            epoch=len(history),
            evaluation_progress={"completed": completed, "total": total},
        ),
    )
    save_checkpoint(
        test_metrics,
        training_complete=True,
        progress={"phase": "complete", "completed_epochs": len(history)},
    )
    publish_training_state(
        status="complete",
        phase="complete",
        epoch=len(history),
        test=test_metrics,
        validation=history[best_epoch - 1]["validation"],
    )
    return {
        "checkpoint": str(destination.resolve()),
        "device": str(device),
        "amp": use_amp,
        "partition": {
            "max_core_faces": max_core_faces,
            "max_batch_faces": max_batch_faces,
            "halo_hops": halo_hops,
            "prefetch_batches": prefetch_batches,
            "sample_workers": sample_workers,
            "sample_cache_gib": sample_cache_gib,
        },
        "training_samples": len(training_files),
        "validation_samples": len(validation_files),
        "test_samples": len(test_files),
        "dataset_roots": dataset_roots,
        "max_samples_per_source": max_samples_per_source,
        "selection_metric": selection_metric,
        "epoch_snapshots": (
            str(snapshot_directory.resolve()) if snapshot_directory is not None else None
        ),
        "augmentation": {"random_rotation": rotation_augmentation},
        "resumed_from_epoch": start_epoch,
        "checkpoint_every_batches": checkpoint_every_batches,
        "telemetry_path": (
            str(telemetry_destination.resolve()) if telemetry_destination is not None else None
        ),
        "split_manifest": str(Path(split_manifest).resolve()) if split_manifest else None,
        "best_epoch": best_epoch,
        "best_validation": history[best_epoch - 1]["validation"],
        "test": test_metrics,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the mesh2cad triangle-graph baseline")
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="+",
        help="one or more prepared NPZ dataset directories",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="mixed precision on CUDA (default: enabled)",
    )
    parser.add_argument(
        "--max-core-faces",
        type=int,
        default=50_000,
        help="GPU core triangles per partition; the complete mesh is still used",
    )
    parser.add_argument(
        "--max-batch-faces",
        type=int,
        default=160_000,
        help="packed GPU triangles per optimizer step; meshes are never truncated",
    )
    parser.add_argument("--halo-hops", type=int, default=2)
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=2,
        help="CPU batches prepared ahead of the accelerator (default: 2; 0 disables)",
    )
    parser.add_argument(
        "--sample-workers",
        type=int,
        default=2,
        help="bounded concurrent NPZ decode/partition workers (default: 2)",
    )
    parser.add_argument(
        "--sample-cache-gib",
        type=float,
        default=0.0,
        help="process-local decoded sample cache across epochs (default: disabled)",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("analytic", "all-surfaces"),
        default="analytic",
        help=(
            "best-checkpoint objective: analytic plane/cylinder IoU or all-surface IoU, "
            "plus boundary F1"
        ),
    )
    parser.add_argument(
        "--epoch-snapshots",
        type=Path,
        help="optional directory for path-free per-epoch candidate weights",
    )
    parser.add_argument(
        "--max-samples-per-source",
        type=int,
        help="deterministic source-balanced subset for staged training",
    )
    parser.add_argument(
        "--balance-samples",
        type=int,
        default=4_096,
        help="samples used to estimate class weights (default: 4096)",
    )
    parser.add_argument(
        "--rotation-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="randomly rotate centroid/normal features during training (default: enabled)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume OUTPUT from its latest recovery checkpoint",
    )
    parser.add_argument(
        "--checkpoint-every-batches",
        type=int,
        default=25,
        help="write an atomic recovery checkpoint every N optimizer batches (0 disables)",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help="explicit train/validation/test JSON; prevents random cross-source leakage",
    )
    parser.add_argument(
        "--telemetry-path",
        type=Path,
        help="optional atomic JSON status file for a live operations dashboard",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    stop_event = Event()
    handled_signals = tuple(
        candidate
        for candidate in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None))
        if candidate is not None
    )
    previous_handlers: dict[int, Any] = {}

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    for candidate in handled_signals:
        previous_handlers[int(candidate)] = signal.signal(candidate, request_stop)
    try:
        summary = train(
            args.dataset,
            args.output,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            layers=args.layers,
            device_name=args.device,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            amp=args.amp,
            max_core_faces=args.max_core_faces,
            max_batch_faces=args.max_batch_faces,
            halo_hops=args.halo_hops,
            prefetch_batches=args.prefetch_batches,
            sample_workers=args.sample_workers,
            sample_cache_gib=args.sample_cache_gib,
            selection_metric=args.selection_metric,
            epoch_snapshots=args.epoch_snapshots,
            split_manifest=args.split_manifest,
            max_samples_per_source=args.max_samples_per_source,
            balance_samples=args.balance_samples,
            rotation_augmentation=args.rotation_augmentation,
            resume=args.resume,
            checkpoint_every_batches=args.checkpoint_every_batches,
            telemetry_path=args.telemetry_path,
            stop_event=stop_event,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        publish_telemetry(
            getattr(args, "telemetry_path", None),
            {"status": "failed", "phase": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        print(f"mesh2cad-train: error: {exc}", file=sys.stderr)
        return 2
    finally:
        for candidate in handled_signals:
            signal.signal(candidate, previous_handlers[int(candidate)])
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
