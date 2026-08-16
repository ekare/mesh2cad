import numpy as np

from mesh2cad.learning.infer import connected_components


def test_connected_components_respects_semantics_and_boundaries():
    adjacency = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4]])
    boundaries = np.asarray([0.1, 0.9, 0.1, 0.1])
    semantics = np.asarray([0, 0, 0, 1, 1])

    instances = connected_components(5, adjacency, boundaries, semantics)

    assert instances[0] == instances[1]
    assert instances[1] != instances[2]
    assert instances[2] != instances[3]
    assert instances[3] == instances[4]


def test_confident_boundary_head_can_repair_one_semantic_disagreement():
    adjacency = np.asarray([[0, 1], [1, 2]])
    boundaries = np.asarray([0.01, 0.8])
    semantics = np.asarray([0, 1, 1])

    instances = connected_components(3, adjacency, boundaries, semantics)

    assert instances[0] == instances[1]
    assert instances[1] != instances[2]
