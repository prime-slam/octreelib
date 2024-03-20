import numpy as np
import pytest

from octreelib.grid import Grid, GridConfig


@pytest.fixture()
def generated_grid_with_planar_clouds():
    def generate_planar_cloud(
        n_points, plane_coefficients, voxel_corner, edge_length, sigma
    ):
        voxel_points = (
            np.random.rand(n_points, 3) * np.array([edge_length - 6 * sigma] * 3)
            + voxel_corner
            + 3 * sigma
        )
        noise = np.random.normal(0, sigma, (n_points,))
        plane_points_z = (
            -plane_coefficients[0] * voxel_points[:, 0]
            - plane_coefficients[1] * voxel_points[:, 1]
            - plane_coefficients[3]
        ) / plane_coefficients[2]
        noisy_plane_points_z = plane_points_z + noise
        return np.column_stack((voxel_points[:, :2], noisy_plane_points_z))

    n_points = 10
    corner = np.array([0, 0, 0])
    edge_length = 5
    sigma = 0.1

    grid = Grid(GridConfig(voxel_edge_length=edge_length))

    grid.insert_points(
        0,
        generate_planar_cloud(
            n_points=n_points,
            plane_coefficients=(1, 2, 3, 0.5),
            voxel_corner=corner,
            edge_length=edge_length,
            sigma=sigma,
        ),
    )
    grid.insert_points(
        1,
        generate_planar_cloud(
            n_points=n_points,
            plane_coefficients=(-1, 2, 3, 0.5),
            voxel_corner=corner,
            edge_length=edge_length,
            sigma=sigma,
        ),
    )

    return grid


def test_map_leaf_points_cuda_ransac(generated_grid_with_planar_clouds):
    grid = generated_grid_with_planar_clouds
    grid.map_leaf_points_cuda_ransac()
