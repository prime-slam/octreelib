from typing import List, TypeVar, Tuple

import numpy as np
import numpy.typing as npt

__all__ = ["Box", "Point", "PointCloud", "T"]

Point = npt.NDArray[np.float_]  # 3x1
PointCloud = npt.NDArray[np.float_]  # 3xN
Box = Tuple[Point, Point]
T = TypeVar("T")
