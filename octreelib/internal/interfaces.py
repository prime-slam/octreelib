from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from octreelib.internal.typing import Point, PointCloud

__all__ = ["WithID", "WithPoints"]


class WithID(ABC):
    _id_static_counter = 0

    @property
    def id(self):
        return self._id

    def __init__(self, _id: Optional[int] = None):
        if _id is not None:
            self._id = _id
        else:
            self._id = WithID._id_static_counter
            WithID._id_static_counter += 1


class WithPoints(ABC):
    @property
    def _empty_point_cloud(self):
        return np.empty((0, 3), dtype=float)

    def __init__(self, points: Optional[PointCloud] = None):
        self.points: List[Point] = points or []
