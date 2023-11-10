from __future__ import annotations

import itertools
from typing import Annotated, Literal, List

import numpy as np
import numpy.typing as npt


__all__ = ["RawPoint", "RawPointCloud", "CloudManager"]

"""
RawPoint and RawPointCloud are intended to be used in the methods
which interact with the user or the methods which facilitate those.
These are meant to be the main types for Points and Point Clouds to
be used by user.
"""

RawPoint = Annotated[npt.NDArray[np.float_], Literal[3]]
RawPointCloud = Annotated[npt.NDArray[np.float_], Literal["N", 3]]


class CloudManager:
    """
    This class implements methods for point cloud manipulation.
    """
    def __init__(self):
        raise TypeError("This class is not intended to be instantiated")

    @classmethod
    def hash_point(cls, point: RawPoint):
        """
        Hash point.
        :param point: Point to hash.
        :return: Hash of the point.
        """
        return hash((point[0], point[1], point[2]))

    @classmethod
    def empty(cls):
        """
        :return: Empty point cloud.
        """
        return np.empty((0, 3), dtype=float)

    @classmethod
    def add(cls, points_a: RawPoint, points_b: RawPoint):
        """
        Add two point clouds.
        :param points_a: First point cloud.
        :param points_b: Second point cloud.
        :return: Combined point cloud.
        """
        return np.vstack([points_a, points_b])

    @classmethod
    def distribute_grid(
        cls, points: RawPointCloud, voxel_size: float, grid_start: RawPoint
    ):
        """
        Distribute points into voxels.
        :param points: Points to distribute.
        :param voxel_size: Edge length of the voxel.
        :param grid_start: Start of the grid.
        :return: Dictionary of voxel coordinates and points in the voxel.
        """
        voxel_indices = (((points - grid_start) // voxel_size) * voxel_size).astype(int)
        voxel_dict = {}
        unique_indices = np.unique(voxel_indices, axis=0)

        for unique_id in unique_indices:
            mask = np.where((voxel_indices == unique_id).all(axis=1))
            voxel_dict[tuple(unique_id)] = points[mask]

        return voxel_dict

    @classmethod
    def distribute(
        cls, points: RawPointCloud, corner_min: RawPoint, edge_length: float
    ) -> List[RawPointCloud]:
        """
        Distribute points into 8 octants.
        :param points: Points to distribute.
        :param corner_min: Corner of the octree node.
        :param edge_length: Edge length of the octree node.
        :return: List of point clouds in the octants.
        """
        clouds = []

        for offset in itertools.product([0, edge_length / 2], repeat=3):
            child_corner_min = corner_min + np.array(offset)
            child_corner_max = child_corner_min + edge_length / 2
            mask = np.all(
                (points >= child_corner_min) & (points < child_corner_max), axis=1
            )
            child_points = points[mask]
            clouds.append(child_points)

        return clouds
