import numpy as np

from octreelib.octree import OctreeNode, Octree, OctreeConfig


__all__ = ["test_octree", "test_octree_node"]


def test_octree_node():
    cached_leaves = []
    node = OctreeNode(np.array([0, 0, 0]), np.float_(10), cached_leaves)

    point_cloud = np.array(
        [
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 3],
            [9, 9, 8],
            [9, 9, 9],
        ],
        dtype=float,
    )

    node.insert_points(point_cloud)

    node.subdivide([lambda points: len(points) > 2])
    assert node.n_leaves == 3
    assert node.n_points == 5
    node.filter([lambda points: len(points) >= 2])
    assert node.n_points == 4
    assert len(cached_leaves) == 15


def test_octree():
    octree_config = OctreeConfig()
    octree = Octree(
        octree_config,
        np.array([0, 0, 0]),
        np.float_(10),
    )

    point_cloud = np.array(
        [
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 3],
            [9, 9, 8],
            [9, 9, 9],
        ],
        dtype=float,
    )

    octree.insert_points(point_cloud)

    received_points = octree.get_points()

    assert (point_cloud == received_points).all()

    octree.subdivide([lambda points: len(points) > 2])
    assert octree.n_leaves == 3
    assert octree.n_points == 5
    octree.filter([lambda points: len(points) >= 2])
    assert octree.n_points == 4
