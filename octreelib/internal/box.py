from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from octreelib.internal.point import RawPoint


__all__ = ["Box"]


@dataclass
class Box:
    """
    Represents a box (or cuboid) in 3D space as a geometric object.
    Unlike Voxel, this class is designed for purely geometric purposes
    e.g. retrieving bounding boxes, determining whether a point is inside some space.

    A box is defined by two opposite points:
    :param corner_min: point with all minimal coordinates
    :param corner_min: point with all  maximal coordinates
    """

    corner_min: RawPoint
    corner_max: RawPoint

    def __post_init__(self):
        # check that the coordinates are valid for a box
        assert (self.corner_min <= self.corner_max).all()

    def intersect(self, other: Box) -> Optional[Box]:
        """
        :param other: other box to intersect
        :return: Box, which represents intersection of two boxes
        """
        new_corner_min = np.maximum(self.corner_min, other.corner_min)
        new_corner_max = np.minimum(self.corner_max, other.corner_max)
        if np.all(new_corner_min < new_corner_max):
            return Box(new_corner_min, new_corner_max)

    def is_point_inside(self, point: RawPoint):
        """
        :param point: Point to check.
        :return: True if point is inside the box, False if outside.
        """
        return bool((self.corner_min <= point).all()) and bool(
            (point <= self.corner_max).all()
        )
