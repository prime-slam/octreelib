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
    corner_a: RawPoint
    corner_b: RawPoint

    def __post_init__(self):
        # check that the coordinates are valid for a box
        assert (self.corner_a <= self.corner_b).all()

    def intersect(self, other: Box) -> Box:
        """
        :param other: other box to intersect
        :return: Box, which represents intersection of two boxes
        """
        min_point = np.maximum(self.corner_a, other.corner_a)
        max_point = np.minimum(self.corner_b, other.corner_b)
        if np.all(min_point < max_point):
            return Box(min_point, max_point)

    def is_point_inside(self, point: RawPoint):
        """
        :param point: Point to check.
        :return: True if point is inside the box, False if outside.
        """
        return bool((self.corner_a <= point).all()) and bool(
            (point <= self.corner_b).all()
        )
