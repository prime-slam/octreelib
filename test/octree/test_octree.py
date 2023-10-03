import numpy as np

from octreelib.octree.octree import OctreeNode, Octree, OctreeConfig


__all__ = ["test_octree", "test_octree_node"]


def test_octree_node():
    node = OctreeNode(
        np.array([0, 0, 0]),
        np.float_(10),
    )

    point_cloud = [
        np.array([0, 0, 1]),
        np.array([0, 0, 2]),
        np.array([0, 0, 3]),
        np.array([9, 9, 8]),
        np.array([9, 9, 9]),
    ]

    node.insert_points(point_cloud)

    node.subdivide([lambda points: len(points) > 2])
    assert node.n_leafs == 15
    assert node.n_points == 5
    node.filter(lambda points: len(points) >= 2)
    assert node.n_points == 4


def test_octree():
    octree_config = OctreeConfig()
    octree = Octree(
        octree_config,
        np.array([0, 0, 0]),
        np.float_(10),
    )

    point_cloud = [
        np.array([0, 0, 1]),
        np.array([0, 0, 2]),
        np.array([0, 0, 3]),
        np.array([9, 9, 8]),
        np.array([9, 9, 9]),
    ]

    octree.insert_points(point_cloud)

    received_points = octree.get_points()

    assert point_cloud == received_points

    octree.subdivide([lambda points: len(points) > 2])
    assert octree.n_leafs == 15
    assert octree.n_points == 5
    octree.filter(lambda points: len(points) >= 2)
    assert octree.n_points == 4
