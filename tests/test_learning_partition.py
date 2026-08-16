import numpy as np

from mesh2cad.learning.features import TriangleGraphSample
from mesh2cad.learning.partition import partition_sample
from mesh2cad.learning.train import _merge_partitions


def test_partition_preserves_every_node_and_edge_target_once():
    face_count = 11
    adjacency = np.asarray([[index, index + 1] for index in range(face_count - 1)])
    sample = TriangleGraphSample(
        vertices=np.empty((0, 3), dtype=np.float32),
        faces=np.empty((0, 3), dtype=np.int64),
        features=np.zeros((face_count, 12), dtype=np.float32),
        adjacency=adjacency,
        surface_labels=np.arange(face_count, dtype=np.int64) % 2,
        face_instance_labels=np.arange(face_count, dtype=np.int64),
        boundary_labels=np.ones(len(adjacency), dtype=np.float32),
    )

    partitions = partition_sample(sample, max_core_faces=4, halo_hops=2)
    core_visits = np.zeros(face_count, dtype=np.int64)
    edge_visits = np.zeros(len(adjacency), dtype=np.int64)
    for partition in partitions:
        core_visits[partition.global_node_ids[partition.core_node_mask]] += 1
        edge_visits[partition.global_edge_ids[partition.edge_loss_mask]] += 1
        assert partition.adjacency.min() >= 0
        assert partition.adjacency.max() < len(partition.features)

    np.testing.assert_array_equal(core_visits, np.ones(face_count, dtype=np.int64))
    np.testing.assert_array_equal(edge_visits, np.ones(len(adjacency), dtype=np.int64))


def test_merge_partitions_offsets_edges_without_connecting_graphs():
    face_count = 4
    adjacency = np.asarray([[0, 1], [1, 2], [2, 3]])
    sample = TriangleGraphSample(
        vertices=np.empty((0, 3), dtype=np.float32),
        faces=np.empty((0, 3), dtype=np.int64),
        features=np.zeros((face_count, 12), dtype=np.float32),
        adjacency=adjacency,
        surface_labels=np.arange(face_count, dtype=np.int64),
        face_instance_labels=np.arange(face_count, dtype=np.int64),
        boundary_labels=np.ones(len(adjacency), dtype=np.float32),
    )
    partition = partition_sample(sample, max_core_faces=None)[0]

    merged = _merge_partitions([partition, partition])

    assert len(merged.features) == 8
    assert len(merged.adjacency) == 6
    np.testing.assert_array_equal(merged.adjacency[:3], adjacency)
    np.testing.assert_array_equal(merged.adjacency[3:], adjacency + 4)
    assert not np.any((merged.adjacency[:, 0] < 4) & (merged.adjacency[:, 1] >= 4))
