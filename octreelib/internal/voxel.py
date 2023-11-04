import itertools
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from octreelib.internal.box import Box
from octreelib.internal.interfaces import WithID
from octreelib.internal.point import (
    RawPoint,
    RawPointCloud,
    PointCloud,
)

__all__ = ["Voxel", "VoxelBase"]


class VoxelBase(WithID):
    """
    Represents a Voxel with ID
    :param corner_min: corner point with all minimal coordinates
    :param edge_length: edge_length of the voxel
    """

    _static_voxel_id_map = {}

    def __init__(
        self,
        corner_min: RawPoint,
        edge_length: float,
    ):
        self._corner_min = corner_min
        self._edge_length = edge_length

        voxel_position_hash = hash(
            PointCloud(np.vstack([corner_min, corner_min + edge_length]))
        )

        if voxel_position_hash not in self._static_voxel_id_map:
            self._static_voxel_id_map[voxel_position_hash] = len(
                self._static_voxel_id_map
            )

        WithID.__init__(self, self._static_voxel_id_map[voxel_position_hash])

    def is_point_geometrically_inside(self, point: RawPoint) -> bool:
        """
        This method checks if the point is inside the voxel geometrically.
        :param point: Point to check.
        :return: True if point is inside the bounding box of a voxel, False if outside.
        """
        return bool((self.corner_min <= point).all()) and bool(
            (point <= self.corner_max).all()
        )

    @property
    def corner_min(self):
        return self._corner_min

    @property
    def edge_length(self):
        return self._edge_length

    @property
    def corner_max(self):
        return self.corner_min + self.edge_length

    @property
    def bounding_box(self):
        """
        :return: bounding box
        """
        return Box(self._corner_min, self._corner_min + self._edge_length)

    @property
    def all_corners(self):
        """
        :return: 8 points, which represent the corners of the voxel
        """
        return [
            self._corner_min + offset
            for offset in itertools.product([0, self._edge_length], repeat=3)
        ]


class Voxel(VoxelBase):
    """
    Voxel with a mutable point cloud.
    :param corner_min: corner point with all minimal coordinates
    :param edge_length: edge_length of the voxel
    """

    def __init__(
        self,
        corner_min: RawPoint,
        edge_length: float,
        points: Optional[RawPointCloud] = None,
    ):
        super().__init__(corner_min, edge_length)

        self._points: PointCloud = (
            points if points is not None else PointCloud.empty()
        )

    def get_points(self) -> RawPointCloud:
        """
        :return: Points, which are stored inside the voxel.
        """
        return self._points.copy()

    def insert_points(self, points: RawPointCloud):
        """
        :param points: Points to insert
        """
        self._points = self._points.extend(points)
