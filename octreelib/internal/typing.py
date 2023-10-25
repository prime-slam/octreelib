from typing import TypeVar, Tuple

from octreelib.internal.point import Point

__all__ = ["Box", "T"]

Box = Tuple[Point, Point]
T = TypeVar("T")
