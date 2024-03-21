import numpy as np
import pytest

from octreelib.internal import PointCloud
from octreelib.grid import Grid, GridConfig
from octreelib.octree import OctreeConfig, Octree
from octreelib.octree_manager import OctreeManager


def points_are_same(points_first: PointCloud, points_second: PointCloud):
    return set(map(str, points_first.tolist())) == set(map(str, points_second.tolist()))


@pytest.fixture()
def generated_grid():
    grid = Grid(GridConfig(voxel_edge_length=5))
    points_0 = np.array(
        [
            [0, 0, 1],  # voxel 0,0,0
            [0, 0, 2],  # voxel 0,0,0
            [0, 0, 3],  # voxel 0,0,0
            [9, 9, 8],  # voxel 5,5,5
            [9, 9, 9],  # voxel 5,5,5
        ],
        dtype=float,
    )
    grid.insert_points(0, points_0)
    points_1 = np.array(
        [
            [1, 0, 1],  # voxel 0,0,0
            [4, 0, 2],  # voxel 0,0,0
            [0, 2, 3],  # voxel 0,0,0
            [5, 9, 9],  # voxel 5,5,5
            [9, 3, 8],  # voxel 5,0,5
        ],
        dtype=float,
    )
    grid.insert_points(1, points_1)

    return grid, [points_0, points_1]


def test_n_leaves(generated_grid):
    grid, pose_points = generated_grid

    assert grid.n_leaves(0) == 2
    assert grid.n_leaves(1) == 3
    grid.subdivide([lambda points: len(points) > 2])
    assert grid.n_leaves(0) == 4
    assert grid.n_leaves(1) == 5


def test_n_points(generated_grid):
    grid, pose_points = generated_grid

    assert grid.n_points(0) == 5
    assert grid.n_points(1) == 5
    grid.subdivide([lambda points: len(points) > 2])
    assert grid.n_points(0) == 5
    assert grid.n_points(1) == 5


def test_n_nodes(generated_grid):
    grid, pose_points = generated_grid

    assert grid.n_nodes(0) == 2
    assert grid.n_nodes(1) == 3
    grid.subdivide([lambda points: len(points) > 2])
    assert grid.n_nodes(0) == 26
    assert grid.n_nodes(1) == 27


def test_get_points(generated_grid):
    grid, pose_points = generated_grid
    assert set(map(str, grid.get_points(0))) == set(map(str, pose_points[0]))
    assert set(map(str, grid.get_points(1))) == set(map(str, pose_points[1]))
    grid.subdivide([lambda points: len(points) > 2])
    assert set(map(str, grid.get_points(0))) == set(map(str, pose_points[0]))
    assert set(map(str, grid.get_points(1))) == set(map(str, pose_points[1]))


@pytest.mark.parametrize(
    "subdivision_criteria, leaves_expected",
    [
        ([lambda points: len(points) > 2], [4, 5]),
        ([lambda points: len(points) > 3], [3, 5]),
    ],
)
def test_subdivide(generated_grid, subdivision_criteria, leaves_expected):
    grid, pose_points = generated_grid

    grid.subdivide(subdivision_criteria)
    assert leaves_expected == [grid.n_leaves(0), grid.n_leaves(1)]


def test_map_leaf_points(generated_grid):
    grid, pose_points = generated_grid

    assert grid.n_points(0) > grid.n_leaves(0)
    assert grid.n_points(1) > grid.n_leaves(1)
    grid.map_leaf_points(lambda cloud: [cloud[0]])
    assert grid.n_points(0) == grid.n_leaves(0)
    assert grid.n_points(1) == grid.n_leaves(1)


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
        == 3
    )

    assert {
        leaf_points_pos_0_voxel.id for leaf_points_pos_0_voxel in leaf_points_pos_0
    }.issubset(
        {leaf_points_pos_1_voxel.id for leaf_points_pos_1_voxel in leaf_points_pos_1}
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
        map(str, pose_points[1][4:])
    )
    assert set(map(str, leaf_points_pos_1[2].get_points())) == set(
        map(str, pose_points[1][3:4])
    )


def test_invalid_octree_type():
    try:
        Grid(
            GridConfig(
                octree_manager_type=type(None),
                octree_config=OctreeConfig(),
                voxel_edge_length=5,
            )
        )
    except TypeError as e:
        assert str(e) == (
            "Cannot use the provided octree manager type NoneType. "
            "It has to be a subclass of octree_manager.OctreeManager."
        )
    else:
        raise AssertionError(
            "This type of octree manager should have caused an exception"
        )

    try:
        Grid(
            GridConfig(
                octree_manager_type=OctreeManager,
                octree_type=type(None),
                octree_config=OctreeConfig(),
                voxel_edge_length=5,
            )
        )
    except TypeError as e:
        assert str(e) == (
            "Cannot use the provided octree type NoneType. "
            "It has to be a subclass of octree.OctreeBase."
        )
    else:
        raise AssertionError("This type of octree should have caused an exception")
