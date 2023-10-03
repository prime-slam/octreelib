from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from octreelib.internal import Voxel
from octreelib.internal.typing import PointCloud, Point

__all__ = ["OctreeConfigBase", "OctreeBase", "OctreeNodeBase"]


@dataclass
class OctreeConfigBase(ABC):
    """
    Config for OcTree
    """

    debug: bool = True


class OctreeNodeBase(Voxel, ABC):
    """
    points: stores points of a node

    children: stores children of a node

    has_children: node stores children instead of points

    When subdivided, all points are **transferred** to children
    and are not stored in the parent node.
    """

    points: Optional[PointCloud]
    children: Optional[List["OctreeNodeBase"]]
    has_children: bool

    def __init__(self, corner: Point, edge_length: np.float_):
        super().__init__(corner, edge_length)
        self.points = []
        self.children = []
        self.has_children = False

    @property
    @abstractmethod
    def n_nodes(self):
        """
        :return: number of nodes
        """
        pass

    @abstractmethod
    def _point_is_inside(self, point: Point) -> bool:
        """
        :param point: point
        :return: the given point is inside the
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
        """
        filter nodes with points by criterion
        :param filtering_criterion:
        """
        pass


class OctreeBase(Voxel, ABC):
    """
    Octree stores points of a **single** pos.

    root: root node of an octree
    """

    _node_type = OctreeNodeBase

    def __init__(
        self,
        octree_config: OctreeConfigBase,
        corner: Point,
        edge_length: np.float_,
    ):
        super().__init__(corner, edge_length)
        self.config = octree_config
        self.root = self._node_type(self.corner, self.edge_length)

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
        """
        filter nodes with points by criterion
        :param filtering_criterion:
        """
        pass
