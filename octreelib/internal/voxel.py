import itertools
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from octreelib.internal.box import Box
from octreelib.internal.interfaces import WithID
from octreelib.internal.point import (
    RawPoint,
    RawPointCloud,
    Point,
    get_hashable_from_point,
    PointCloud,
)

__all__ = ["DynamicVoxel", "Voxel"]


class Voxel(WithID):
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
        points: Optional[RawPointCloud] = None,
    ):
        self._corner_min = corner_min
        self._edge_length = edge_length

        voxel_position_hash = hash(PointCloud(np.vstack([corner_min, corner_min + edge_length])))

        if voxel_position_hash not in self._static_voxel_id_map:
            self._static_voxel_id_map[voxel_position_hash] = len(
                self._static_voxel_id_map
            )

        WithID.__init__(
            self, self._static_voxel_id_map[voxel_position_hash]
        )

        self._points: RawPointCloud = (
            points if points is not None else PointCloud.empty()
        )

    def get_points(self) -> RawPointCloud:
        """
        :return: Points, which are stored inside the voxel.
        """
        return self._points.copy()

    @property
    def corner_min(self):
        return self._corner_min

    @property
    def edge_length(self):
        return self._edge_length

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


class DynamicVoxel(Voxel, ABC):
    """
    Voxel with a mutable point cloud.
    :param corner_min: corner point with all minimal coordinates
    :param edge_length: edge_length of the voxel
    """

    def __init__(self, corner_min: RawPoint, edge_length: float):
        Voxel.__init__(self, corner_min, edge_length)

    @abstractmethod
    def insert_points(self, points: RawPointCloud):
        """
        :param points: Points to insert
        """
        pass
