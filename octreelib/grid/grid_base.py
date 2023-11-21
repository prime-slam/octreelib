from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Generic, List, Type, Optional

import numpy as np

from octreelib.internal.point import Point, PointCloud
from octreelib.internal.voxel import Voxel
from octreelib.internal.typing import T
from octreelib.octree import OctreeConfigBase, OctreeBase, Octree, OctreeConfig
from octreelib.octree_manager import OctreeManager

__all__ = ["GridVisualizationType", "VisualizationConfig", "GridConfigBase", "GridBase"]


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
    unused_voxels: Represents list of voxels which were unused on optimisation stage
    """

    type: GridVisualizationType = GridVisualizationType.VOXEL
    point_size: float = 0.1
    line_width_size: float = 0.01
    line_color: int = 0xFF0000
    filepath: str = "visualization.html"
    seed: int = 0
    unused_voxels: List[int] = field(default_factory=list)


@dataclass
class GridConfigBase(ABC):
    """
    Config for Grid

    octree_manager_type: type of OctreeManager used.
        OctreeManager is responsible for managing octrees for different poses.
    octree_type: type of Octree used
        Octree is the data structure used for storing points.
    octree_config: This config will be forwarded to the octrees.
    debug: True enables debug mode.
    voxel_edge_length: Initial size of voxels.
    corner: Corner of a grid.
    """

    octree_manager_type: Type[OctreeManager] = OctreeManager
    octree_type: Type[OctreeBase] = Octree
    octree_config: OctreeConfigBase = field(default_factory=OctreeConfig)
    debug: bool = False
    voxel_edge_length: float = 1
    corner: Point = field(default_factory=lambda: np.array(([0.0, 0.0, 0.0])))

    def __post_init__(self):
        """
        :raises TypeError: if given octree_type is not compatible with this type of grid.
        :raises TypeError: if given octree_manager_type is not compatible with this type of grid.
        """
        if not issubclass(self.octree_manager_type, OctreeManager):
            raise TypeError(
                f"Cannot use the provided octree manager type {self.octree_manager_type.__name__}. "
                "It has to be a subclass of octree_manager.OctreeManager."
            )
        if not issubclass(self.octree_type, OctreeBase):
            raise TypeError(
                f"Cannot use the provided octree type {self.octree_type.__name__}. "
                "It has to be a subclass of octree.OctreeBase."
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
    def insert_points(self, pose_number: int, points: List[Point]) -> None:
        """
        Insert points to the according octree.
        If an octree for this pos does not exist, a new octree is created
        :param pose_number: Pose number to which the cloud is inserted.
        :param points: Point cloud to be inserted.
        """
        pass

    @abstractmethod
    def get_points(self, pose_number: int) -> List[Point]:
        """
        Returns points for a specific pose number.
        :param pose_number: The desired pose number.
        :return: Points belonging to the pose.
        """
        pass

    @abstractmethod
    def subdivide(
        self,
        subdivision_criteria: List[Callable[[PointCloud], bool]],
        pose_numbers: Optional[List[int]] = None,
    ):
        """
        Subdivides all octrees based on all points and given subdivision criteria.
        :param pose_numbers: List of pose numbers to subdivide.
        :param subdivision_criteria: List of bool functions which represent criteria for subdivision.
        If any of the criteria returns **true**, the octree node is subdivided.
        """
        pass

    @abstractmethod
    def filter(self, filtering_criteria: List[Callable[[PointCloud], bool]]):
        """
        Filters nodes of each octree with points by criterion
        :param filtering_criteria: List of bool functions which represent criteria for filtering.
            If any of the criteria returns **false**, the point cloud in octree leaf is removed.
        """
        pass

    @abstractmethod
    def map_leaf_points(self, function: Callable[[PointCloud], PointCloud]):
        """
        Transform point cloud in each node of each octree using the function
        :param function: Transformation function PointCloud -> PointCloud. It is applied to each leaf node.
        """
        pass

    @abstractmethod
    def get_leaf_points(self, pose_number: int) -> List[Voxel]:
        """
        :param pose_number: The desired pose number.
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

    @abstractmethod
    def n_nodes(self, pose_number: int):
        """
        :param pose_number: The desired pose number.
        :return: Number of nodes of an octree for given pose number.
        """
        pass

    @abstractmethod
    def n_points(self, pose_number: int):
        """
        :param pose_number: The desired pose number.
        :return: Number of points of an octree for given pose number.
        """
        pass

    @abstractmethod
    def n_leaves(self, pose_number: int):
        """
        :param pose_number: The desired pose number.
        :return: Number of leaf nodes in the octree for given pose number.
        """
        pass
