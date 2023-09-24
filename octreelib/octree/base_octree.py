import abc

from typing import Generic, Callable, List, Optional, TypeVar

from octreelib.internal import Voxel
from octreelib.internal.typing import PointCloud


class OctreeNodeBase(Voxel, abc.ABC):
    """
    points: stores points of a node

    children: stores children of a node

    When subdivided, all points are **transferred** to children
    and are not stored in the parent node.
    """

    points: Optional[PointCloud]
    children: Optional[List["OctreeNodeBase"]]
    has_children: bool

    @property
    @abc.abstractmethod
    def n_nodes(self):
        """
        :return: number of nodes
        """
        pass

    @property
    @abc.abstractmethod
    def n_leafs(self):
        """
        :return: number of leafs a.k.a. number of nodes which have points
        """
        pass

    @property
    @abc.abstractmethod
    def n_points(self):
        """
        :return: number of points in the octree
        """
        return

    @abc.abstractmethod
    def filter(self, filtering_criterion: Callable[[PointCloud], bool]):
        pass


T = TypeVar("T")


class OctreeBase(Voxel, abc.ABC, Generic[T]):
    """
    Octree stores points of a **single** pos.
    Generic[T] is used for specifying the class of OctreeNode used.

    root: root node of an octree
    """

    root: T

    @property
    @abc.abstractmethod
    def n_nodes(self):
        """
        :return: number of nodes
        """
        pass

    @property
    @abc.abstractmethod
    def n_leafs(self):
        """
        :return: number of leafs a.k.a. number of nodes which have points
        """
        pass

    @property
    @abc.abstractmethod
    def n_points(self):
        """
        :return: number of points in the octree
        """
        pass

    @abc.abstractmethod
    def filter(self, filtering_criterion: Callable[[PointCloud], bool]):
        pass
