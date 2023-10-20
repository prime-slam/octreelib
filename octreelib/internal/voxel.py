import itertools
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import numpy as np

from octreelib.internal.typing import Point, PointCloud
from octreelib.internal.interfaces import WithID, WithPoints

__all__ = ["StaticStoringVoxel", "StoringVoxel", "Voxel"]


class Voxel(WithID):
    def __init__(self, corner: Point, edge_length: np.float_):
        WithID.__init__(self)
        self.corner = corner
        self.edge_length = edge_length

    @property
    def corners(self):
        return [
            self.corner + offset
            for offset in itertools.product([0, self.edge_length], repeat=3)
        ]


class StaticStoringVoxel(Voxel, WithPoints):
    def __init__(
        self, corner: Point, edge_length: np.float_, points: Optional[PointCloud] = None
    ):
        Voxel.__init__(self, corner, edge_length)
        WithPoints.__init__(self, points)

    def get_points(self) -> PointCloud:
        return self.points.copy()


class StoringVoxel(Voxel, WithPoints, ABC):
    def __init__(self, corner: Point, edge_length: np.float_):
        Voxel.__init__(self, corner, edge_length)
        WithPoints.__init__(self)

    @abstractmethod
    def subdivide(self, subdivision_criteria: List[Callable[[PointCloud], bool]]):
        pass

    @abstractmethod
    def insert_points(self, points: PointCloud):
        pass

    @abstractmethod
    def get_points(self) -> PointCloud:
        pass
