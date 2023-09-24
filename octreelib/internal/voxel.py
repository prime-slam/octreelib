from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable,List

import numpy as np

from octreelib.internal.typing import Point, PointCloud

__all__ = ["Voxel"]


@dataclass
class Voxel(ABC):
    corner: Point
    edge_length: np.float_

    @abstractmethod
    def insert_points(self, points: PointCloud):
        pass

    @abstractmethod
    def get_points(self) -> PointCloud:
        pass

    @abstractmethod
    def subdivide(self, subdivision_criteria: List[Callable[[PointCloud], bool]]):
        pass
