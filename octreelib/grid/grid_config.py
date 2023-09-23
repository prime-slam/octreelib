import dataclasses

import numpy as np

from octreelib.internal.typing import Point


@dataclasses.dataclass
class GridConfig:
    """
    Positioning within world coordinates.

    ----

    **corner:** the left bottom rear corner of the grid

    **voxel_edge_length:** length of voxel edge

    **n_voxels_x:** number of voxels along X axis

    **n_voxels_y:** number of voxels along Y axis

    **n_voxels_z:** number of voxels along Z axis

    **phi:** horizontal rotation of the grid

    **theta:** vertical rotation of the grid
    """
    corner: Point
    voxel_edge_length: np.float_
    n_voxels_x: int
    n_voxels_y: int
    n_voxels_z: int
    phi: np.float_ = np.double(0)
    theta: np.float_ = np.double(0)
