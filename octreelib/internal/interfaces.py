"""
This file contains interface classes which represent
certain features of the objects.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np

from octreelib.internal.point import RawPointCloud, PointCloud, PosePointCloud

__all__ = ["WithID"]


class WithID(ABC):
    """
    This class represents the fact that the object has an id.
    When initialized, the class is assigned an id, which can
    be retrieved later using .id property.

    :param _id: Optional[int] -- if specified, this id will be assigned.
    """

    _id_static_counter = 0

    def __init__(self, _id: Optional[int] = None):
        if _id is not None:
            self._id = _id
        else:
            self._id = WithID._id_static_counter
            WithID._id_static_counter += 1

    @property
    def id(self):
        return self._id
