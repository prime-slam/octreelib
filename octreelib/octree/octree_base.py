from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Callable, List, Optional, Type

import numpy as np

from octreelib.internal import Voxel
from octreelib.internal.typing import PointCloud, T, Point

__all__ = ["OctreeBase", "OctreeNodeBase"]


@dataclass
class OctreeConfigBase:
    """
    Config for OcTree
    """


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

    node_type = OctreeNodeBase

    def __init__(
        self,
        octree_config: OctreeConfigBase,
        corner: Point,
        edge_length: np.float_,
    ):
        super().__init__(corner, edge_length)
        self.config = octree_config
        self.root = self.node_type(self.corner, self.edge_length)

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
