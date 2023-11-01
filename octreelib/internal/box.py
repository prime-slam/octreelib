from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from octreelib.internal.point import RawPoint


__all__ = ["Box"]


@dataclass
class Box:
    corner_a: RawPoint
    corner_b: RawPoint

    def __post_init__(self):
        assert (self.corner_a <= self.corner_b).all()

    def intersect(self, other: Box) -> Box:
        min_point = np.maximum(self.corner_a, other.corner_a)
        max_point = np.minimum(self.corner_b, other.corner_b)
        if np.all(min_point < max_point):
            return Box(min_point, max_point)

    def is_point_inside(self, point: RawPoint):
        return bool((self.corner_a <= point).all()) and bool(
            (point <= self.corner_b).all()
        )
