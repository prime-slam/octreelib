import abc
import dataclasses

import numpy as np
import numpy.typing as npt


@dataclasses.dataclass
class Voxel(abc.ABC):
    center: npt.NDArray[np.float_]
    edge_size: np.float_
