from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt


__all__ = [
    "RawPoint",
    "RawPointCloud",
    "hash_point",
]

"""
RawPoint and RawPointCloud are intended to be used in the methods
which interact with the user or the methods which facilitate those.
These are meant to be the main types for Points and Point Clouds to
be used by user.
"""

RawPoint = Annotated[npt.NDArray[np.float_], Literal[3]]
RawPointCloud = Annotated[npt.NDArray[np.float_], Literal["N", 3]]


def hash_point(point: RawPoint):
    return hash((point[0], point[1], point[2]))
