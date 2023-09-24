import abc
import dataclasses
from typing import Callable
from typing import List

import numpy as np

from octreelib.internal.typing import Point, PointCloud


@dataclasses.dataclass
class Voxel(abc.ABC):
    corner: Point
    edge_length: np.float_

    @abc.abstractmethod
    def insert_points(self, points: PointCloud):
        pass

    @abc.abstractmethod
    def get_points(self) -> PointCloud:
        pass

    @abc.abstractmethod
    def subdivide(self, subdivision_criteria: List[Callable[[PointCloud], bool]]):
        pass
