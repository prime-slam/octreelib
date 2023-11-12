from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt


__all__ = ["Point", "PointCloud"]

"""
Point and PointCloud are intended to be used in the methods
which interact with the user or the methods which facilitate those.
These are meant to be the main types for Points and Point Clouds to
be used by user.
"""

Point = Annotated[npt.NDArray[np.float_], Literal[3]]
PointCloud = Annotated[npt.NDArray[np.float_], Literal["N", 3]]
