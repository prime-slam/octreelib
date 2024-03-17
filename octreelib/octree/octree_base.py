from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from octreelib.internal.point import Point, PointCloud
from octreelib.internal.voxel import Voxel

__all__ = ["OctreeConfigBase", "OctreeBase", "OctreeNodeBase"]


@dataclass
class OctreeConfigBase(ABC):
    """
    Config for Octree

    debug: debug mode is enabled
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

    def __init__(
        self,
        corner_min: Point,
        edge_length: float,
        octree_cached_leaves: List["OctreeNodeBase"],
    ):
        super().__init__(corner_min, edge_length)
        self._points: np.empty((0, 3), dtype=float)
        self._children: Optional[List["OctreeNodeBase"]] = []
        self._has_children: bool = False
        self._cached_leaves = octree_cached_leaves
        self._cached_leaves.append(self)

    @property
    @abstractmethod
    def n_nodes(self):
        """
        :return: number of nodes
        """
        pass

    @property
    @abstractmethod
    def n_leaves(self):
        """
        :return: number of leaves a.k.a. number of nodes which have points
        """
        pass

    @property
    @abstractmethod
    def n_points(self):
        """
        :return: number of points in the octree node
        """
        return

    @abstractmethod
    def filter(self, filtering_criteria: List[Callable[[PointCloud], bool]]):
        """
        Filter nodes with points by filtering criteria
        :param filtering_criteria: List of bool functions which represent criteria for filtering.
            If any of the criteria returns **false**, the point cloud in octree leaf is removed.
        """
        pass

    @abstractmethod
    def map_leaf_points(self, function: Callable[[PointCloud], PointCloud]):
        """
        transform point cloud in the node using the function
        :param function: transformation function PointCloud -> PointCloud
        """
        pass

    @abstractmethod
    def get_leaf_points(self) -> List[Voxel]:
        """
        :return: List of voxels where each voxel represents a leaf node with points.
        """
        pass

    @abstractmethod
    def subdivide(self, subdivision_criteria: List[Callable[[PointCloud], bool]]):
        """
        Subdivide node based on the subdivision criteria.
        :param subdivision_criteria: List of bool functions which represent criteria for subdivision.
        If any of the criteria returns **true**, the octree node is subdivided.
        """
        pass

    @abstractmethod
    def subdivide_as(self, other: "OctreeNodeBase"):
        """
        Subdivide octree node using the subdivision scheme of a different octree node.
        :param other: Octree node to copy subdivision scheme from.
        """
        pass

    @abstractmethod
    def get_points(self) -> PointCloud:
        """
        :return: Points, which are stored inside the node.
        """
        pass

    @abstractmethod
    def apply_mask(self, mask: np.ndarray):
        """
        Apply mask to the point cloud in the octree node
        :param mask: Mask to apply
        """
        self._points = self._points[mask]


class OctreeBase(Voxel, ABC):
    """
    Stores points in the form of an octree.

    :param octree_config: Configuration for the octree.
    :param corner_min: Min corner of the octree.
    :param edge_length: Edge length of the octree.
    """

    _node_type = OctreeNodeBase

    def __init__(
        self,
        octree_config: OctreeConfigBase,
        corner_min: Point,
        edge_length: float,
    ):
        super().__init__(corner_min, edge_length)
        self._config = octree_config
        self._cached_leaves = []
        self._root = self._node_type(
            self.corner_min, self.edge_length, self._cached_leaves
        )

    @property
    @abstractmethod
    def n_nodes(self):
        """
        :return: number of nodes
        """
        pass

    @property
    @abstractmethod
    def n_leaves(self):
        """
        :return: number of leaves a.k.a. number of nodes which have points
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
    def filter(self, filtering_criteria: List[Callable[[PointCloud], bool]]):
        """
        filter nodes with points by criterion
        :param filtering_criteria: List of bool functions which represent criteria for filtering.
            If any of the criteria returns **false**, the point cloud in octree leaf is removed.
        """
        pass

    @abstractmethod
    def map_leaf_points(self, function: Callable[[PointCloud], PointCloud]):
        """
        transform point cloud in each node using the function
        :param function: transformation function PointCloud -> PointCloud
        """
        pass

    @abstractmethod
    def get_leaf_points(self, non_empty: bool) -> List[Voxel]:
        """
        :param non_empty: If True, only non-empty leaf nodes are returned.
        :return: List of PointClouds where each PointCloud
        represents points in a separate leaf node
        """
        pass

    @abstractmethod
    def subdivide(self, subdivision_criteria: List[Callable[[PointCloud], bool]]):
        """
        Subdivide node based on the subdivision criteria.
        :param subdivision_criteria: list of criteria for subdivision
        """
        pass

    @abstractmethod
    def subdivide_as(self, other: "OctreeBase"):
        """
        Subdivide octree using the subdivision scheme of a different octree.
        :param other: Octree to copy subdivision scheme from.
        """
        pass

    @abstractmethod
    def get_points(self) -> PointCloud:
        """
        :return: Points, which are stored inside the octree.
        """
        pass

    @abstractmethod
    def insert_points(self, points: PointCloud):
        pass

    @abstractmethod
    def apply_mask(self, mask: np.ndarray):
        """
        Apply mask to the point cloud in the octree
        :param mask: Mask to apply
        """
