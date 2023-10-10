from typing import List, TypeVar, Tuple

import numpy as np
import numpy.typing as npt

__all__ = ["Box", "Point", "PointCloud", "T"]

Point = npt.NDArray[np.float_]  # typedef for a numpy array of float
PointCloud = List[Point]
Box = Tuple[Point, Point]
T = TypeVar("T")
