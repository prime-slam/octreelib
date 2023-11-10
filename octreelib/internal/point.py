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
    def __init__(self):
        raise TypeError("This class is not intended to be instantiated")

    @classmethod
    def hash_point(cls, point: RawPoint):
        return hash((point[0], point[1], point[2]))

    @classmethod
    def empty(cls):
        return np.empty((0, 3), dtype=float)

    @classmethod
    def add(cls, points_a, points_b):
        return np.vstack([points_a, points_b])

    @classmethod
    def distribute_grid(cls, points, voxel_size, grid_start):
        # Calculate voxel indices for all points using integer division
        voxel_indices = (((points - grid_start) // voxel_size) * voxel_size).astype(int)

        # Create a unique identifier for each voxel based on its indices
        voxel_ids = tuple(voxel_indices.T)

        # Create a dictionary to store points in each voxel
        voxel_dict = {}

        # Use np.unique to get unique voxel_ids and their corresponding indices
        unique_ids, unique_indices = np.unique(voxel_ids, axis=1, return_inverse=True)

        # Iterate over unique voxel_ids and assign points to corresponding voxels
        for unique_id in unique_ids.T:
            indices_to_select = np.where((voxel_indices == unique_id).all(axis=1))
            voxel_dict[tuple(unique_id)] = points[indices_to_select]

        return voxel_dict

    @classmethod
    def distribute(
        cls, points: RawPointCloud, corner_min, edge_length
    ) -> List[RawPointCloud]:
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
