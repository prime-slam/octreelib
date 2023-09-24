from abc import ABC, abstractmethod
from typing import Generic, Callable, List, Optional

from octreelib.internal import Voxel
from octreelib.internal.typing import PointCloud, T


class OctreeNodeBase(Voxel, ABC):
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
    @abstractmethod
    def n_nodes(self):
        """
        :return: number of nodes
        """
        pass

    @property
    @abstractmethod
    def n_leafs(self):
        """
        :return: number of leafs a.k.a. number of nodes which have points
        """
        pass

    @property
    @abstractmethod
    def n_points(self):
        """
        :return: number of points in the octree
        """
        return

    @abstractmethod
    def filter(self, filtering_criterion: Callable[[PointCloud], bool]):
        pass


class OctreeBase(Voxel, ABC, Generic[T]):
    """
    Octree stores points of a **single** pos.
    Generic[T] is used for specifying the class of OctreeNode used.

    root: root node of an octree
    """

    root: T

    @property
    @abstractmethod
    def n_nodes(self):
        """
        :return: number of nodes
        """
        pass

    @property
    @abstractmethod
    def n_leafs(self):
        """
        :return: number of leafs a.k.a. number of nodes which have points
        """
        pass

    @property
    @abstractmethod
    def n_points(self):
        """
        :return: number of points in the octree
        """
        pass

    @abstractmethod
    def filter(self, filtering_criterion: Callable[[PointCloud], bool]):
        pass
