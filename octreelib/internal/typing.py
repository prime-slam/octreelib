from typing import TypeVar, Tuple

from octreelib.internal.point import RawPoint

__all__ = ["Box", "T"]

Box = Tuple[RawPoint, RawPoint]
T = TypeVar("T")
