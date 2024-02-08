import numpy as np
import pytest

from octreelib.internal import PointCloud, Voxel
from octreelib.octree import OctreeConfig, Octree
from octreelib.octree_manager import OctreeManager
from octreelib.ransac.cuda_ransac import CudaRansac


def points_are_same(points_first: PointCloud, points_second: PointCloud):
    return set(map(str, points_first.tolist())) == set(map(str, points_second.tolist()))


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

    octree_manager = OctreeManager(Octree, OctreeConfig(), corner, edge_length * 2)
    octree_manager.insert_points(
        0, generate_planar_cloud(N, A, B, C, D, corner, edge_length, sigma)
    )
    octree_manager.insert_points(
        0,
        generate_planar_cloud(N, -A, B, C, D, corner + edge_length, edge_length, sigma),
    )
    octree_manager.insert_points(
        1, generate_planar_cloud(N, A, B, C, D, corner, edge_length, sigma)
    )
    octree_manager.insert_points(
        1,
        generate_planar_cloud(N, -A, B, C, D, corner + edge_length, edge_length, sigma),
    )

    octree_manager.subdivide([lambda points: len(points) > 150], [0])

    return octree_manager


@pytest.fixture()
def generated_multi_pose():
    multi_pose = OctreeManager(Octree, OctreeConfig(), np.array([0, 0, 0]), 5)

    points_0 = np.array(
        [
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 3],
        ],
        dtype=float,
    )
    multi_pose.insert_points(0, points_0)
    points_1 = np.array(
        [
            [1, 0, 1],
            [4, 0, 2],
            [0, 2, 3],
        ],
        dtype=float,
    )
    multi_pose.insert_points(1, points_1)

    return multi_pose, {0: points_0, 1: points_1}


def test_insert_and_get_points(generated_multi_pose):
    multi_pose, clouds = generated_multi_pose
    assert points_are_same(multi_pose.get_points(0), clouds[0])
    assert points_are_same(multi_pose.get_points(1), clouds[1])


@pytest.mark.parametrize(
    "subdivision_criteria, subdivision_pose_numbers, nodes_expected, leaves_expected",
    [
        ([lambda points: len(points) > 2], [0], [9, 9], [2, 3]),
        ([lambda points: len(points) > 1], None, [33, 33], [3, 3]),
    ],
)
def test_subdivide(
    generated_multi_pose,
    subdivision_criteria,
    subdivision_pose_numbers,
    nodes_expected,
    leaves_expected,
):
    multi_pose, clouds = generated_multi_pose
    assert multi_pose.n_nodes(0) == 1
    assert multi_pose.n_nodes(1) == 1
    assert multi_pose.n_leaves(0) == 1
    assert multi_pose.n_leaves(1) == 1
    multi_pose.subdivide(subdivision_criteria, subdivision_pose_numbers)
    assert multi_pose.n_nodes(0) == nodes_expected[0]
    assert multi_pose.n_nodes(1) == nodes_expected[1]
    assert multi_pose.n_leaves(0) == leaves_expected[0]
    assert multi_pose.n_leaves(1) == leaves_expected[1]


def test_map_leaf_points(generated_multi_pose):
    multi_pose, clouds = generated_multi_pose
    multi_pose.map_leaf_points(lambda points: points[0].reshape((1, 3)), [0])
    assert multi_pose.n_points(0) == 1
    assert multi_pose.n_points(1) == 3


def test_filter(generated_multi_pose):
    multi_pose, clouds = generated_multi_pose
    multi_pose.subdivide([lambda points: len(points) > 2], [0])
    multi_pose.filter([lambda points: False], [0])
    multi_pose.filter([lambda points: True], [1])
    assert multi_pose.n_points(0) == 0
    assert multi_pose.n_points(1) == 3


@pytest.mark.parametrize(
    "subdivision_criteria, subdivision_pose_numbers, leaves_expected",
    [
        (
            [lambda points: len(points) > 2],
            [0],
            [
                [
                    Voxel(
                        np.array([0, 0, 0]),
                        2.5,
                        points=np.array([[0, 0, 1], [0, 0, 2]]),
                    ),
                    Voxel(
                        np.array([0, 0, 2.5]),
                        2.5,
                        points=np.array([[0, 0, 3]]),
                    ),
                ],
                [
                    Voxel(
                        np.array([0, 0, 0]),
                        2.5,
                        points=np.array([[1, 0, 1]]),
                    ),
                    Voxel(
                        np.array([0, 0, 2.5]),
                        2.5,
                        points=np.array([[0, 2, 3]]),
                    ),
                    Voxel(
                        np.array([2.5, 0, 0]),
                        2.5,
                        points=np.array([[4, 0, 2]]),
                    ),
                ],
            ],
        ),
        (
            [lambda points: len(points) > 1],
            None,
            [
                [
                    Voxel(
                        np.array([0, 0, 0.625]),
                        0.625,
                        points=np.array([[0, 0, 1]]),
                    ),
                    Voxel(
                        np.array([0, 0, 1.25]),
                        1.25,
                        points=np.array([[0, 0, 2]]),
                    ),
                    Voxel(
                        np.array([0, 0, 2.5]),
                        1.25,
                        points=np.array([[0, 0, 3]]),
                    ),
                ],
                [
                    Voxel(
                        np.array([0.625, 0, 0.625]),
                        0.625,
                        points=np.array([[1, 0, 1]]),
                    ),
                    Voxel(
                        np.array([0, 1.25, 2.5]),
                        1.25,
                        points=np.array([[0, 2, 3]]),
                    ),
                    Voxel(
                        np.array([2.5, 0, 0]),
                        2.5,
                        points=np.array([[4, 0, 2]]),
                    ),
                ],
            ],
        ),
    ],
)
def test_get_leaf_points(
    generated_multi_pose,
    subdivision_criteria,
    subdivision_pose_numbers,
    leaves_expected,
):
    multi_pose, clouds = generated_multi_pose
    multi_pose.subdivide(subdivision_criteria, subdivision_pose_numbers)
    leaves_0 = multi_pose.get_leaf_points(0)
    leaves_1 = multi_pose.get_leaf_points(1)
    assert [voxel.id for voxel in leaves_0] == [
        voxel.id for voxel in leaves_expected[0]
    ]
    assert [voxel.id for voxel in leaves_1] == [
        voxel.id for voxel in leaves_expected[1]
    ]


def test_n_points(generated_multi_pose):
    multi_pose, clouds = generated_multi_pose
    assert multi_pose.n_points(0) == 3
    assert multi_pose.n_points(1) == 3


def test_map_leaf_points_cuda(generated_multi_pose_large):
    octree_manager = generated_multi_pose_large
    octree_manager.map_leaf_points_cuda(CudaRansac(), 8**2, 16)
