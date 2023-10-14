from typing import Optional

import numpy as np

from octreelib.internal import Point, Box


def point_is_inside_box(point: Point, box: Box) -> bool:
    return bool((box[0] <= point).all()) and bool((point <= box[1]).all())


def boxes_intersection(box_1: Box, box_2: Box) -> Optional[Box]:
    min_point = np.maximum(box_1[0], box_2[0])
    max_point = np.minimum(box_1[1], box_2[1])
    if np.all(min_point < max_point):
        return min_point, max_point
