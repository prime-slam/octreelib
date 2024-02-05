import numpy as np
import pytest

from octreelib.octree import OctreeNode, Octree, OctreeConfig
from octreelib.ransac.cuda_ransac import CudaRansac


__all__ = ["test_octree", "test_octree_node"]


@pytest.fixture()
def generated_multi_pose_large():
    def generate_planar_cloud(N, A, B, C, D, voxel_corner, edge_length, sigma):
        voxel_points = (
            np.random.rand(N, 3) * np.array([edge_length - 6 * sigma] * 3)
            + voxel_corner
            + 3 * sigma
        )
        noise = np.random.normal(0, sigma, (N,))
        plane_points_z = (-A * voxel_points[:, 0] - B * voxel_points[:, 1] - D) / C
        noisy_plane_points_z = plane_points_z + noise
        return np.column_stack((voxel_points[:, :2], noisy_plane_points_z))

    N = 100
    A, B, C, D = 1, 2, 3, 0.5
    corner = np.array([0, 0, 0])
    edge_length = 5
    sigma = 0.5

    octree = Octree(OctreeConfig(), corner, edge_length * 2)
    octree.insert_points(
        generate_planar_cloud(N, A, B, C, D, corner, edge_length, sigma)
    )
    octree.insert_points(
        generate_planar_cloud(N, -A, B, C, D, corner + edge_length, edge_length, sigma)
    )

    octree.subdivide([lambda points: len(points) > 150])

    return octree


def test_octree_node():
    node = OctreeNode(np.array([0, 0, 0]), np.float_(10))

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


def test_cuda_ransac(generated_multi_pose_large):
    octree = generated_multi_pose_large
    ransac = CudaRansac()

    octree.map_leaf_points_cuda(ransac, 8)
