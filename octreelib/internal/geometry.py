from typing import Optional

from internal import Point, Box


def point_is_inside_box(point: Point, box: Box) -> bool:
    return bool((box[0] <= point).all()) and bool((point <= box[1]).all())


def boxes_intersection(box_1: Box, box_2: Box) -> Optional[Box]:
    raise NotImplementedError
