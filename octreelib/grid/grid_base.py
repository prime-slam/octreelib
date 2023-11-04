from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Generic, List, Type

import numpy as np

from octreelib.internal.point import RawPoint, RawPointCloud
from octreelib.internal.voxel import Voxel
from octreelib.internal.typing import T
from octreelib.octree.octree_base import OctreeConfigBase

__all__ = ["GridVisualizationType", "GridConfigBase", "GridBase"]


class GridVisualizationType(Enum):
    """
    Represents types of Grid visualization:
    1. POSE - renders in different colors point clouds belonging to different poses
    2. VOXEL - renders in same colors voxels belonging to different poses
    """
    POSE = "pose"
    VOXEL = "voxel"


@dataclass
class VisualizationConfig:
    """
    Represents configuration for Grid visualization

    visualization_type: Represents type of visualization. For more information check VisualizationType definition
    point_size: Represents size of points in Grid
    line_width_size: Represents size of voxels lines in Grid
    line_color: Represents color of voxels lines in Grid
    visualization_filepath: Path to produced `.html` file
    visualization_seed: Represents random seed for generating colors
    """
    type: GridVisualizationType = GridVisualizationType.VOXEL
    point_size: float = 0.1
    line_width_size: float = 0.01
    line_color: int = 0xFF0000
    filepath: str = "visualization.html"
    seed: int = 0


@dataclass
class GridConfigBase(ABC):
    """
    Config for Grid

    octree_type: type of Octree used
    octree_config: config to be forwarded to the octrees
    debug: debug mode
    grid_voxel_edge_length: initial size of voxels
    corner: corner of a grid
    """

    _compatible_octree_types = []

    octree_type: Type[T]
    octree_config: OctreeConfigBase
    debug: bool = False
    grid_voxel_edge_length: int = 1
    corner: RawPoint = np.array(([0.0, 0.0, 0.0]))

    @property
    def compatible_octree_types(self):
        """
        :return: Types of Octrees which are compatible with this
        """
        return self._compatible_octree_types

    def __post_init__(self):
        """
        Check that
        :raises TypeError: if given octree_type is not compatible with this type of grid.
        """
        if self.octree_type not in self.compatible_octree_types:
            raise TypeError(
                f"Cannot use the provided octree type {self.octree_type.__name__}. "
                f"The compatible octree types are [{', '.join(cls.__name__ for cls in self.compatible_octree_types)}]."
            )


class GridBase(ABC, Generic[T]):
    """
    This class stores octrees for different nodes
    Generic[T] is used for specifying the class of Octree used.

    :param grid_config: GridConfig
    """

    def __init__(
        self,
        grid_config: GridConfigBase,
    ):
        """
        :param grid_config: config
        """
        self._grid_config = grid_config

    @abstractmethod
    def insert_points(self, pose_number: int, points: List[RawPoint]) -> None:
        """
        Insert points to the according octree.
        If an octree for this pos does not exist, a new octree is created
        :param pose_number: pos to which the cloud is inserted
        :param points: point cloud
        """
        pass

    @abstractmethod
    def get_points(self, pose_number: int) -> List[RawPoint]:
        """
        Returns points for a specific pos
        :param pose_number: the desired pos
        :return: point cloud
        """
        pass

    @abstractmethod
    def subdivide(self, subdivision_criteria: List[Callable[[RawPointCloud], bool]]):
        """
        Subdivides all octrees
        :param subdivision_criteria: criteria for subdivision.
        If any of the criteria returns **true**, the octree node is subdivided
        """
        pass

    @abstractmethod
    def filter(self, filtering_criteria: List[Callable[[RawPointCloud], bool]]):
        """
        Filters nodes of each octree with points by criterion
        :param filtering_criteria:
        """
        pass

    @abstractmethod
    def map_leaf_points(self, function: Callable[[RawPointCloud], RawPointCloud]):
        """
        Transform point cloud in each node of each octree using the function
        :param function: transformation function PointCloud -> PointCloud
        """
        pass

    @abstractmethod
    def n_nodes(self, pose_number: int):
        """
        :param pose_number: desired pose number.
        :return: number of nodes of an octree for given pose number
        """
        pass

    @abstractmethod
    def n_points(self, pose_number: int):
        """
        :param pose_number: desired pose number.
        :return: number of points of an octree for given pose number
        """
        pass

    @abstractmethod
    def n_leaves(self, pose_number: int):
        """
        :param pose_number: the desired pose number.
        :return: number of leaf nodes in the octree for given pose number
        """
        pass

    @abstractmethod
    def get_leaf_points(self, pose_number: int) -> List[Voxel]:
        """
        :param pose_number: the desired pose number
        :return: List of voxels. Each voxel is a representation of a leaf node.
        Each voxel has the same corner, edge_length and points as one of the leaf nodes.
        """
        pass

    @abstractmethod
    def visualize(self, config: VisualizationConfig) -> None:
        """
        Represents method for visualizing Grid. It produces `.html` file using
        [k3d](https://github.com/K3D-tools/K3D-jupyter) library
        """
        pass
