from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Type

import numpy as np

from octreelib.internal.typing import Point, PointCloud, T
from octreelib.octree import OctreeConfigBase

__all__ = ["GridConfigBase", "GridBase"]


@dataclass
class GridConfigBase(ABC):
    """
    Config for Grid

    min_voxel_size: size of a minimal possible voxel, the octree will be able to subdivide to
    corner: corner of a grid
    octree_type: type of Octree used
    octree_config: config to be forwarded to the octrees
    debug: debug mode
    """

    min_voxel_size = 1
    corner = np.array(([0.0, 0.0, 0.0]))
    octree_type: Type[T]
    octree_config: OctreeConfigBase
    debug: bool = False


class GridBase(ABC, Generic[T]):
    """
    This class stores octrees for different nodes
    Generic[T] is used for specifying the class of Octree used.
    """

    octrees: Dict[int, T] = {}  # {pos: octree}

    def __init__(
        self,
        grid_config: GridConfigBase,
    ):
        """
        :param grid_config: config
        """
        self.grid_config = grid_config

    @abstractmethod
    def insert_points(self, pose_number: int, points: List[Point]) -> None:
        """
        Insert points to the according octree.
        If an octree for this pos does not exist, a new octree is created
        :param pose_number: pos to which the cloud is inserted
        :param points: point cloud
        """
        pass

    @abstractmethod
    def get_points(self, pose_number: int) -> List[Point]:
        """
        Returns points for a specific pos
        :param pose_number: the desired pos
        :return: point cloud
        """
        pass

    @abstractmethod
    def merge(self, merger: Any):
        pass

    @abstractmethod
    def subdivide(self, subdivision_criteria: List[Callable[[PointCloud], bool]]):
        """
        Subdivides all octrees
        :param subdivision_criteria: criteria for subdivision.
        If any of the criteria returns **true**, the octree node is subdivided
        """
        pass

    @abstractmethod
    def filter(self, filtering_criteria: List[Callable[[PointCloud], bool]]):
        pass
