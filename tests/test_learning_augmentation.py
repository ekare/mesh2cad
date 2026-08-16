from __future__ import annotations

from pathlib import Path

import numpy as np

from mesh2cad.learning.partition import GraphPartition
from mesh2cad.learning.train import (
    SamplePartitionCache,
    _ordered_sample_partitions,
    _prefetch_batches,
    _random_rotation_matrix,
    _rotate_partition_features,
)


def _partition() -> GraphPartition:
    features = np.asarray(
        [
            [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [-2.0, 1.0, 0.5, 0.0, 1.0, 0.0, -7.0, -8.0, -9.0, 1.0, 2.0, 3.0],
        ],
        dtype=np.float32,
    )
    return GraphPartition(
        features=features,
        adjacency=np.asarray([[0, 1]], dtype=np.int64),
        surface_labels=np.asarray([0, 1], dtype=np.int64),
        boundary_labels=np.asarray([1.0], dtype=np.float32),
        core_node_mask=np.asarray([True, True]),
        edge_loss_mask=np.asarray([True]),
        global_node_ids=np.asarray([0, 1], dtype=np.int64),
        global_edge_ids=np.asarray([0], dtype=np.int64),
    )


def test_random_rotation_is_seeded_right_handed_and_orthonormal() -> None:
    np.random.seed(23)
    first = _random_rotation_matrix()
    np.random.seed(23)
    second = _random_rotation_matrix()

    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first @ first.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(first), 1.0, atol=1e-12)


def test_rotation_changes_only_centroid_and_normal_features() -> None:
    partition = _partition()
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )

    rotated = _rotate_partition_features(partition, rotation)

    np.testing.assert_allclose(
        np.linalg.norm(rotated.features[:, 0:3], axis=1),
        np.linalg.norm(partition.features[:, 0:3], axis=1),
    )
    np.testing.assert_allclose(
        np.linalg.norm(rotated.features[:, 3:6], axis=1),
        np.linalg.norm(partition.features[:, 3:6], axis=1),
    )
    np.testing.assert_array_equal(rotated.features[:, 6:], partition.features[:, 6:])
    np.testing.assert_array_equal(rotated.adjacency, partition.adjacency)
    np.testing.assert_array_equal(rotated.surface_labels, partition.surface_labels)
    np.testing.assert_array_equal(partition.features[:, :6], _partition().features[:, :6])


def test_batch_prefetch_preserves_order() -> None:
    partitions = [_partition(), _partition(), _partition()]

    prefetched = list(_prefetch_batches(iter(partitions), depth=2))

    assert len(prefetched) == len(partitions)
    assert all(actual is expected for actual, expected in zip(prefetched, partitions, strict=True))


def test_batch_prefetch_propagates_producer_errors() -> None:
    def batches():
        yield _partition()
        raise ValueError("broken prepared sample")

    iterator = _prefetch_batches(batches(), depth=1)

    assert next(iterator).features.shape == (2, 12)
    try:
        next(iterator)
    except ValueError as exc:
        assert str(exc) == "broken prepared sample"
    else:
        raise AssertionError("producer error was not propagated")


def test_concurrent_sample_decode_preserves_file_order(monkeypatch) -> None:
    import mesh2cad.learning.train as training

    def load(path, **_kwargs):
        partition = _partition()
        partition.features[0, 0] = float(path.stem)
        return (partition,)

    monkeypatch.setattr(training, "_sample_partitions", load)
    files = [Path(str(index)) for index in range(6)]

    loaded = list(
        _ordered_sample_partitions(files, max_core_faces=10, halo_hops=1, workers=2)
    )

    assert [int(parts[0].features[0, 0]) for parts in loaded] == list(range(6))


def test_concurrent_sample_decode_propagates_worker_error(monkeypatch) -> None:
    import mesh2cad.learning.train as training

    def load(path, **_kwargs):
        if path.stem == "1":
            raise ValueError("broken sample")
        return (_partition(),)

    monkeypatch.setattr(training, "_sample_partitions", load)

    with np.testing.assert_raises_regex(ValueError, "broken sample"):
        list(
            _ordered_sample_partitions(
                [Path("0"), Path("1")],
                max_core_faces=10,
                halo_hops=1,
                workers=2,
            )
        )


def test_sample_partition_cache_reuses_and_evicts_decoded_samples(monkeypatch) -> None:
    import mesh2cad.learning.train as training

    partition = _partition()
    item_bytes = SamplePartitionCache._partition_bytes((partition,))
    cache = SamplePartitionCache(item_bytes)
    calls = []

    def load(path, **_kwargs):
        calls.append(path)
        return (_partition(),)

    monkeypatch.setattr(training, "_sample_partitions", load)
    first = Path("first.npz")
    second = Path("second.npz")

    cache.load(first, max_core_faces=10, halo_hops=1)
    cache.load(first, max_core_faces=10, halo_hops=1)
    cache.load(second, max_core_faces=10, halo_hops=1)
    cache.load(first, max_core_faces=10, halo_hops=1)

    assert calls == [first, second, first]
    assert cache.stats() == {
        "max_bytes": item_bytes,
        "current_bytes": item_bytes,
        "items": 1,
        "hits": 1,
        "misses": 3,
    }
