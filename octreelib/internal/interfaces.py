from abc import ABC, abstractmethod
from typing import List

from octreelib.internal.typing import Point, PointCloud

__all__ = ["WithID"]


class WithID(ABC):
    _id_static_counter = 0

    def __init__(self):
        self.id = WithID._id_static_counter
        WithID._id_static_counter += 1
