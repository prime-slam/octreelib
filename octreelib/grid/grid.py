from typing import Callable
from typing import Dict
from typing import Generic
from typing import List
from typing import Optional
from typing import TypeVar

from octreelib.internal.typing import Point
from octreelib.internal.typing import PointCloud
from octreelib.octree import OctreeConfig

T = TypeVar("T")


class Grid(Generic[T]):
    """
    This class stores octrees for different nodes
    Generic[T] is used for specifying the class of Octree used.
    """

    octrees: Dict[int, T] = {}  # {pos: octree}

    def __init__(
        self,
        octree_config: OctreeConfig,
    ):
        """
        :param octree_config: config which will be forwarded to constructed octrees
        """
        self.octree_config = octree_config
        raise NotImplementedError

    def insert_points(self, pos: int, points: List[Point]) -> None:
        """
        Insert points to the according octree.
        If an octree for this pos does not exist, a new octree is created
        :param pos: pos to which the cloud is inserted
        :param points: point cloud
        """
        raise NotImplementedError

    def get_points(self, pos: int) -> List[Point]:
        """
        Returns points for a specific pos
        :param pos: the desired pos
        :return: point cloud
        """
        raise NotImplementedError

    def merge_trees_for_poses(
        self,
        pos1: int,
        pos2: int,
        merger: Callable[[PointCloud, PointCloud], PointCloud],
        new_pos: Optional[int],
    ):
        """
        Merge octrees for two poses.
        :param pos1: pos 1
        :param pos2: pos 2
        :param merger: the function which merges two point clouds into one.
        :param new_pos: if specified, the new octree is associated with this pos, defaults to pos1
        """
        raise NotImplementedError

    def subdivide_pos(
        self, pos: int, subdivision_criteria: List[Callable[[PointCloud], bool]]
    ):
        """
        Subdivides all octree nodes which store points for a specific pos
        :param subdivision_criteria: criteria for subdivision.
        If any of the criteria returns **true**, the octree node is subdivided
        :param pos:
        """
        raise NotImplementedError

    def subdivide_all(self, subdivision_criteria: List[Callable[[PointCloud], bool]]):
        """
        Subdivides all octrees
        :param subdivision_criteria: criteria for subdivision.
        If any of the criteria returns **true**, the octree node is subdivided
        """
        raise NotImplementedError
