from typing import List, TypeVar

import numpy as np
import numpy.typing as npt

Point = npt.NDArray[np.float_]  # typedef for a numpy array of float
PointCloud = List[Point]
T = TypeVar("T")
