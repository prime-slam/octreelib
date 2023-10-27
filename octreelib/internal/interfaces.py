from abc import ABC
from typing import Optional

import numpy as np

from octreelib.internal.point import RawPointCloud

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

    def __init__(self, points: Optional[RawPointCloud] = None):
        self.points: RawPointCloud = (
            points if points is not None else self._empty_point_cloud
        )
