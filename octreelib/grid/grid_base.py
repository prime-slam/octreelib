import abc

from typing import Any, Callable, Dict, Generic, List, TypeVar

from octreelib.grid.grid_config import GridConfig
from octreelib.internal.typing import Point, PointCloud

T = TypeVar("T")


class GridBase(abc.ABC, Generic[T]):
    """
    This class stores octrees for different nodes
    Generic[T] is used for specifying the class of Octree used.
    """

    octrees: Dict[int, T] = {}  # {pos: octree}

    @abc.abstractmethod
    def __init__(
        self,
        grid_config: GridConfig,
    ):
        """
        :param grid_config: config
        """
        self.grid_config = grid_config

    @abc.abstractmethod
    def insert_points(self, pos: int, points: List[Point]) -> None:
        """
        Insert points to the according octree.
        If an octree for this pos does not exist, a new octree is created
        :param pos: pos to which the cloud is inserted
        :param points: point cloud
        """
        pass

    @abc.abstractmethod
    def get_points(self, pos: int) -> List[Point]:
        """
        Returns points for a specific pos
        :param pos: the desired pos
        :return: point cloud
        """
        pass

    @abc.abstractmethod
    def merge(self, merger: Any):
        pass

    @abc.abstractmethod
    def __subdivide_pos(
        self, pos: int, subdivision_criteria: List[Callable[[PointCloud], bool]]
    ):
        """
        Subdivides all octree nodes which store points for a specific pos
        :param subdivision_criteria: criteria for subdivision.
        If any of the criteria returns **true**, the octree node is subdivided
        :param pos:
        """
        pass

    @abc.abstractmethod
    def subdivide(self, subdivision_criteria: List[Callable[[PointCloud], bool]]):
        """
        Subdivides all octrees
        :param subdivision_criteria: criteria for subdivision.
        If any of the criteria returns **true**, the octree node is subdivided
        """
        pass

    @abc.abstractmethod
    def filter(self, finter_criterion: Callable[[PointCloud], bool]):
        pass
