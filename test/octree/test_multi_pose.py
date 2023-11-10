import numpy as np
import pytest

from octreelib.internal import RawPointCloud, Voxel
from octreelib.octree import OctreeConfig
from octreelib.octree_manager import OctreeManager


def points_are_same(points_first: RawPointCloud, points_second: RawPointCloud):
    return set(map(str, points_first.tolist())) == set(map(str, points_second.tolist()))


@pytest.fixture()
def generated_multi_pose():
    multi_pose = OctreeManager(OctreeConfig(), np.array([0, 0, 0]), 5)

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
