import itertools
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from octreelib.internal.box import Box
from octreelib.internal.interfaces import WithID, WithPoints
from octreelib.internal.point import RawPoint, RawPointCloud, get_hashable_from_point

__all__ = ["StaticStoringVoxel", "StoringVoxel", "Voxel"]



class Voxel(WithID):
    _static_voxel_id_map = {}

    def __init__(self, corner: RawPoint, edge_length: float):
        corner_min_hashable = get_hashable_from_point(corner)
        corner_max_hashable = get_hashable_from_point(corner + edge_length)

        if (corner_min_hashable, corner_max_hashable) not in self._static_voxel_id_map:
            self._static_voxel_id_map[(corner_min_hashable, corner_max_hashable)] = len(
                self._static_voxel_id_map
            )

        WithID.__init__(
            self, self._static_voxel_id_map[(corner_min_hashable, corner_max_hashable)]
        )
        self._corner = corner
        self._edge_length = edge_length

    @property
    def corner(self):
        return self._corner

    @property
    def edge_length(self):
        return self._edge_length

    @property
    def bounding_box(self):
        """
        :return: bounding box
        """
        return Box(self._corner, self._corner + self._edge_length)

    @property
    def corners(self):
        """
        :return: 8 points, which represent the corners of the voxel
        """
        return [
            self._corner + offset
            for offset in itertools.product([0, self._edge_length], repeat=3)
        ]


class StaticStoringVoxel(Voxel, WithPoints):
    """
    Voxel with an immutable point cloud.
    """

    def __init__(
        self,
        corner: RawPoint,
        edge_length: np.float_,
        points: Optional[RawPointCloud] = None,
    ):
        Voxel.__init__(self, corner, edge_length)
        WithPoints.__init__(self, points)

    def get_points(self) -> RawPointCloud:
        """
        :return: Points, which are stored inside the voxel.
        """
        return self._points.copy()


class StoringVoxel(Voxel, WithPoints, ABC):
    """
    Voxel with a mutable point cloud.
    """

    def __init__(self, corner: RawPoint, edge_length: np.float_):
        Voxel.__init__(self, corner, edge_length)
        WithPoints.__init__(self)

    @abstractmethod
    def insert_points(self, points: RawPointCloud):
        """
        :param points: Points to insert
        """
        pass

    @abstractmethod
    def get_points(self) -> RawPointCloud:
        """
        :return: Points, which are stored inside the voxel.
        """
        pass
