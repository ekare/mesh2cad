"""Lossless overlapping partitions for very large triangle graphs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from mesh2cad.learning.features import TriangleGraphSample


@dataclass(frozen=True)
class GraphPartition:
    features: npt.NDArray[np.float32]
    adjacency: npt.NDArray[np.int64]
    surface_labels: npt.NDArray[np.int64]
    boundary_labels: npt.NDArray[np.float32]
    core_node_mask: npt.NDArray[np.bool_]
    edge_loss_mask: npt.NDArray[np.bool_]
    global_node_ids: npt.NDArray[np.int64]
    global_edge_ids: npt.NDArray[np.int64]


def _face_neighbours(face_count: int, adjacency: np.ndarray) -> list[list[int]]:
    neighbours: list[list[int]] = [[] for _ in range(face_count)]
    for left_value, right_value in adjacency:
        left, right = int(left_value), int(right_value)
        neighbours[left].append(right)
        neighbours[right].append(left)
    return neighbours


def _core_groups(face_count: int, adjacency: np.ndarray, max_core_faces: int) -> list[np.ndarray]:
    neighbours = _face_neighbours(face_count, adjacency)
    unassigned = np.ones(face_count, dtype=bool)
    groups: list[np.ndarray] = []
    next_seed = 0
    while np.any(unassigned):
        while next_seed < face_count and not unassigned[next_seed]:
            next_seed += 1
        if next_seed == face_count:
            break
        queue: deque[int] = deque([next_seed])
        unassigned[next_seed] = False
        members: list[int] = []
        while queue and len(members) < max_core_faces:
            current = queue.popleft()
            members.append(current)
            for neighbour in neighbours[current]:
                if unassigned[neighbour]:
                    unassigned[neighbour] = False
                    queue.append(neighbour)
                    if len(members) + len(queue) >= max_core_faces:
                        break
        # Queued nodes beyond the cap were reserved but not consumed; release them.
        for queued in queue:
            unassigned[queued] = True
        groups.append(np.asarray(members, dtype=np.int64))
    return groups


def partition_sample(
    sample: TriangleGraphSample,
    *,
    max_core_faces: int | None = 50_000,
    halo_hops: int = 2,
) -> tuple[GraphPartition, ...]:
    """Partition nodes without dropping any node or edge loss target.

    Core nodes are disjoint. Halo nodes provide message-passing context. Every global edge is owned
    exactly once by the partition containing its left endpoint as a core node.
    """
    face_count = len(sample.features)
    edge_count = len(sample.adjacency)
    if max_core_faces is None or face_count <= max_core_faces:
        return (
            GraphPartition(
                features=sample.features,
                adjacency=sample.adjacency,
                surface_labels=sample.surface_labels,
                boundary_labels=sample.boundary_labels,
                core_node_mask=np.ones(face_count, dtype=bool),
                edge_loss_mask=np.ones(edge_count, dtype=bool),
                global_node_ids=np.arange(face_count, dtype=np.int64),
                global_edge_ids=np.arange(edge_count, dtype=np.int64),
            ),
        )
    if max_core_faces < 1:
        raise ValueError("max_core_faces must be positive or None")
    if halo_hops < 1:
        raise ValueError("halo_hops must be at least one so every owned edge is present")

    adjacency = sample.adjacency
    left = adjacency[:, 0]
    right = adjacency[:, 1]
    groups = _core_groups(face_count, adjacency, max_core_faces)
    partitions: list[GraphPartition] = []
    for core_ids in groups:
        core_global = np.zeros(face_count, dtype=bool)
        core_global[core_ids] = True
        included = core_global.copy()
        frontier = core_global.copy()
        for _ in range(halo_hops):
            touching = frontier[left] | frontier[right]
            expanded_ids = np.unique(adjacency[touching].reshape(-1))
            next_frontier = np.zeros(face_count, dtype=bool)
            next_frontier[expanded_ids] = True
            next_frontier &= ~included
            included |= next_frontier
            frontier = next_frontier
            if not np.any(frontier):
                break

        global_node_ids = np.flatnonzero(included).astype(np.int64)
        global_edge_mask = included[left] & included[right]
        global_edge_ids = np.flatnonzero(global_edge_mask).astype(np.int64)
        global_to_local = np.full(face_count, -1, dtype=np.int64)
        global_to_local[global_node_ids] = np.arange(len(global_node_ids), dtype=np.int64)
        local_adjacency = global_to_local[adjacency[global_edge_ids]]
        local_core = core_global[global_node_ids]
        edge_loss_mask = core_global[left[global_edge_ids]]
        partitions.append(
            GraphPartition(
                features=sample.features[global_node_ids],
                adjacency=local_adjacency,
                surface_labels=sample.surface_labels[global_node_ids],
                boundary_labels=sample.boundary_labels[global_edge_ids],
                core_node_mask=local_core,
                edge_loss_mask=edge_loss_mask,
                global_node_ids=global_node_ids,
                global_edge_ids=global_edge_ids,
            )
        )
    return tuple(partitions)
