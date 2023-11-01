from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from octreelib.internal.point import RawPoint


__all__ = ["Box"]


@dataclass
class Box:
    """
    This class represents a box object.
    The box is defined by two opposite points:
    point_a has all minimal coordinates and
    point_b has all maximal coordinates
    """
    corner_min: RawPoint
    corner_max: RawPoint

    def __post_init__(self):
        # check that the coordinates are valid for a box
        assert (self.corner_min <= self.corner_max).all()

    def intersect(self, other: Box) -> Box:
        """
        :param other: other box to intersect
        :return: Box, which represents intersection of two boxes
        """
        min_point = np.maximum(self.corner_min, other.corner_min)
        max_point = np.minimum(self.corner_max, other.corner_max)
        if np.all(min_point < max_point):
            return Box(min_point, max_point)

    def is_point_inside(self, point: RawPoint):
        """
        :param point: Point to check.
        :return: True if point is inside the box, False if outside.
        """
        return bool((self.corner_min <= point).all()) and bool(
            (point <= self.corner_max).all()
        )
