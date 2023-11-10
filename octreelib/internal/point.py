from __future__ import annotations

import itertools
from typing import Annotated, Literal, List

import numpy as np
import numpy.typing as npt


__all__ = [
    "RawPoint",
    "RawPointCloud",
    "CloudManager"
]

"""
RawPoint and RawPointCloud are intended to be used in the methods
which interact with the user or the methods which facilitate those.
These are meant to be the main types for Points and Point Clouds to
be used by user.
"""

RawPoint = Annotated[npt.NDArray[np.float_], Literal[3]]
RawPointCloud = Annotated[npt.NDArray[np.float_], Literal["N", 3]]


class CloudManager:
    class Voxel:
        def __init__(self, min_corner, edge_length):
            self.corner_min = min_corner
            self.corner_max = min_corner + edge_length
            self.edge_length = edge_length

        def is_point_geometrically_inside(self, point: RawPoint) -> bool:
            return bool((point >= self.corner_min).all()) and bool(
                (point <= self.corner_max).all()
            )

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
        # TODO: implement this smarter 🙄
        def __generate_children():
            child_edge_length = edge_length / np.float_(2)
            children_corners_offsets = itertools.product(
                [0, child_edge_length], repeat=3
            )
            return [
                CloudManager.Voxel(
                    corner_min + offset,
                    child_edge_length,
                )
                for internal_position, offset in enumerate(children_corners_offsets)
            ]

        clouds = []

        for child in __generate_children():
            new_cloud = cls.empty()
            for point in points:
                if child.is_point_geometrically_inside(point):
                    new_cloud = cls.add(new_cloud, point.reshape((1, 3)))
            clouds.append(new_cloud)

        return clouds

