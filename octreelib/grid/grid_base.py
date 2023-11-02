from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Generic, List, Type

import numpy as np

from octreelib.internal.point import RawPoint, RawPointCloud
from octreelib.internal.voxel import StoringVoxel
from octreelib.internal.typing import T
from octreelib.octree.octree_base import OctreeConfigBase

__all__ = ["GridVisualisationType", "GridConfigBase", "GridBase"]


class GridVisualisationType(Enum):
    """
    Represents types of Grid visualisation:
    1. POSE - colors in different colors point clouds belonging to different poses
    2. VOXEL - colors in same colors voxels belonging to different poses
    """
    POSE = "pose"
    VOXEL = "voxel"


@dataclass
class GridConfigBase(ABC):
    """
    Config for Grid

    octree_type: type of Octree used
    octree_config: config to be forwarded to the octrees
    debug: debug mode
    grid_voxel_edge_length: initial size of voxels
    corner: corner of a grid
    visualisation_type: Represents type of visualisation. For more information check VisualisationType definition
    visualisation_filepath: Path to produced `.html` file
    visualisation_seed: Represents random seed for generating colors
    """

    octree_type: Type[T]
    octree_config: OctreeConfigBase
    debug: bool = False
    grid_voxel_edge_length: int = 1
    corner: RawPoint = np.array(([0.0, 0.0, 0.0]))
    visualisation_type: GridVisualisationType = GridVisualisationType.POSE
    visualisation_filepath: str = ""
    visualisation_seed: int = 0

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
    def get_leaf_points(self, pose_number: int) -> List[StoringVoxel]:
        """
        :param pose_number: the desired pose number
        :return: List of voxels. Each voxel is a representation of a leaf node.
        Each voxel has the same corner, edge_length and points as one of the leaf nodes.
        """
        pass

    @abstractmethod
    def visualise(self) -> None:
        """
        Represents method for visualising Grid. It produces `.html` file using
        [k3d](https://github.com/K3D-tools/K3D-jupyter) library
        """
        pass
