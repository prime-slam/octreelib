import numpy as np
import pytest

from octreelib.grid import GridWithPoints, GridWithPointsConfig
from octreelib.octree import MultiPoseOctree, MultiPoseOctreeConfig


def points_are_same(points_a, points_b):
    return set(map(str, list(points_a))) == set(map(str, list(points_b)))


@pytest.fixture()
def generated_grid():
    grid = GridWithPoints(
        GridWithPointsConfig(
            octree_type=MultiPoseOctree,
            octree_config=MultiPoseOctreeConfig(),
            min_voxel_size=5,
        )
    )
    points_0 = [
        np.array([0, 0, 1]),  # voxel 0,0,0
        np.array([0, 0, 2]),  # voxel 0,0,0
        np.array([0, 0, 3]),  # voxel 0,0,0
        np.array([9, 9, 8]),  # voxel 5,5,5
        np.array([9, 9, 9]),  # voxel 5,5,5
    ]
    grid.insert_points(0, points_0)
    points_1 = [
        np.array([1, 0, 1]),  # voxel 0,0,0
        np.array([4, 0, 2]),  # voxel 0,0,0
        np.array([0, 2, 3]),  # voxel 0,0,0
        np.array([5, 9, 9]),  # voxel 5,5,5
        np.array([9, 3, 8]),  # voxel 5,0,5
    ]
    grid.insert_points(1, points_1)

    return grid, [points_0, points_1]


def test_n_leafs(generated_grid):
    grid, pose_points = generated_grid

    assert 2 == grid.n_leafs(0)
    assert 3 == grid.n_leafs(1)
    grid.subdivide([lambda points: len(points) > 2])
    assert 4 == grid.n_leafs(0)
    assert 5 == grid.n_leafs(1)


def test_n_points(generated_grid):
    grid, pose_points = generated_grid

    assert 5 == grid.n_points(0)
    assert 5 == grid.n_points(1)
    grid.subdivide([lambda points: len(points) > 2])
    assert 5 == grid.n_points(0)
    assert 5 == grid.n_points(1)


def test_n_nodes(generated_grid):
    grid, pose_points = generated_grid

    assert 2 == grid.n_nodes(0)
    assert 3 == grid.n_nodes(1)
    grid.subdivide([lambda points: len(points) > 2])
    assert 7 == grid.n_nodes(0)
    assert 8 == grid.n_nodes(1)


def test_get_points(generated_grid):
    grid, pose_points = generated_grid
    assert set(map(str, grid.get_points(0))) == set(map(str, pose_points[0]))
    assert set(map(str, grid.get_points(1))) == set(map(str, pose_points[1]))
    grid.subdivide([lambda points: len(points) > 2])
    assert set(map(str, grid.get_points(0))) == set(map(str, pose_points[0]))
    assert set(map(str, grid.get_points(1))) == set(map(str, pose_points[1]))


@pytest.mark.parametrize(
    "subdivision_criteria, nodes_expected, leafs_expected",
    [
        ([lambda points: len(points) > 2], [7, 8], [4, 5]),
        ([lambda points: len(points) > 3], [4, 6], [3, 5]),
    ],
)
def test_subdivide(
    generated_grid, subdivision_criteria, nodes_expected, leafs_expected
):
    grid, pose_points = generated_grid

    grid.subdivide(subdivision_criteria)
    assert nodes_expected == [grid.n_nodes(0), grid.n_nodes(1)]
    assert leafs_expected == [grid.n_leafs(0), grid.n_leafs(1)]


def test_map_leaf_points(generated_grid):
    grid, pose_points = generated_grid

    assert grid.n_points(0) > grid.n_leafs(0)
    assert grid.n_points(1) > grid.n_leafs(1)
    grid.map_leaf_points(lambda cloud: [cloud[0]])
    assert grid.n_points(0) == grid.n_leafs(0)
    assert grid.n_points(1) == grid.n_leafs(1)


def test_get_leaf_points(generated_grid):
    grid, pose_points = generated_grid

    leaf_points_pos_0 = grid.get_leaf_points(0)
    leaf_points_pos_1 = grid.get_leaf_points(1)

    assert (
        len(
            {
                leaf_points_pos_0[0].id,
                leaf_points_pos_0[1].id,
                leaf_points_pos_1[0].id,
                leaf_points_pos_1[1].id,
                leaf_points_pos_1[2].id,
            }
        )
        == 5
    )

    assert set(map(str, leaf_points_pos_0[0].get_points())) == set(
        map(str, pose_points[0][:3])
    )
    assert set(map(str, leaf_points_pos_0[1].get_points())) == set(
        map(str, pose_points[0][3:])
    )
    assert set(map(str, leaf_points_pos_1[0].get_points())) == set(
        map(str, pose_points[1][:3])
    )
    assert set(map(str, leaf_points_pos_1[1].get_points())) == set(
        map(str, pose_points[1][3:4])
    )
    assert set(map(str, leaf_points_pos_1[2].get_points())) == set(
        map(str, pose_points[1][4:])
    )
