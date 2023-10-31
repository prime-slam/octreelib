from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Generic, List, Type

import numpy as np

from octreelib.internal.point import RawPoint, RawPointCloud
from octreelib.internal.typing import T
from octreelib.internal.voxel import StoringVoxel
from octreelib.octree.octree_base import OctreeConfigBase

__all__ = ["GridConfigBase", "GridBase"]


@dataclass
class GridConfigBase(ABC):
    """
    Config for Grid

    octree_type: type of Octree used
    octree_config: config to be forwarded to the octrees
    grid_voxel_edge_length: size of a minimal possible voxel, the octree will be able to subdivide to
    corner: corner of a grid
    debug: debug mode
    """

    octree_type: Type[T]
    octree_config: OctreeConfigBase
    debug: bool = False
    grid_voxel_edge_length: int = 1
    corner: RawPoint = np.array(([0.0, 0.0, 0.0]))


class GridBase(ABC, Generic[T]):
    """
    This class stores octrees for different nodes
    Generic[T] is used for specifying the class of Octree used.
    """

    def __init__(
        self,
        grid_config: GridConfigBase,
    ):
        """
        :param grid_config: config
        """
        self.grid_config = grid_config

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
        :return: List of
        """
        pass
