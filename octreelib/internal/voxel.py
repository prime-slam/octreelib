import itertools
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import numpy as np

from octreelib.internal.typing import Point, PointCloud
from octreelib.internal.interfaces import WithID

__all__ = ["Voxel", "StoringVoxel"]


class Voxel(WithID, ABC):
    def __init__(self, corner: Point, edge_length: np.float_):
        super().__init__()
        self.corner = corner
        self.edge_length = edge_length

    @property
    def corners(self):
        return [
            self.corner + offset
            for offset in itertools.product([0, self.edge_length], repeat=3)
        ]

    @abstractmethod
    def subdivide(self, subdivision_criteria: List[Callable[[PointCloud], bool]]):
        pass

    @abstractmethod
    def insert_points(self, points: PointCloud):
        pass

    @abstractmethod
    def get_points(self) -> PointCloud:
        pass


class StoringVoxel(Voxel, ABC):
    def __init__(
        self,
        corner: Point,
        edge_length: np.float_,
        points: Optional[List[Point]] = None,
    ):
        super().__init__(corner, edge_length)
        self.points: List[Point] = points or []
