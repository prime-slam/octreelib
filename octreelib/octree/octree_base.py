from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from octreelib.internal.box import Box
from octreelib.internal.point import RawPoint, RawPointCloud, PointCloud
from octreelib.internal.voxel import DynamicVoxel

__all__ = ["OctreeConfigBase", "OctreeBase", "OctreeNodeBase"]


@dataclass
class OctreeConfigBase(ABC):
    """
    Config for Octree

    debug: debug mode is enabled
    """

    debug: bool = True


class OctreeNodeBase(DynamicVoxel, ABC):
    """
    points: stores points of a node

    children: stores children of a node

    has_children: node stores children instead of points

    When subdivided, all points are **transferred** to children
    and are not stored in the parent node.
    """

    _point_cloud_type = PointCloud

    def __init__(self, corner_min: RawPoint, edge_length: np.float_):
        super().__init__(corner_min, edge_length)
        self._points: OctreeNodeBase._point_cloud_type = self._point_cloud_type.empty()
        self._children: Optional[List["OctreeNodeBase"]] = []
        self._has_children: bool = False

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
    def filter(self, filtering_criteria: List[Callable[[RawPointCloud], bool]]):
        """
        Filter nodes with points by filtering criteria
        :param filtering_criteria: List of filtering criteria functions
        """
        pass

    @abstractmethod
    def map_leaf_points(self, function: Callable[[RawPointCloud], RawPointCloud]):
        """
        transform point cloud in the node using the function
        :param function: transformation function RawPointCloud -> RawPointCloud
        """
        pass

    @abstractmethod
    def get_points_inside_box(self, box: Box) -> RawPointCloud:
        """
        Returns points that occupy the given box
        :param box: tuple of two points representing min and max points of the box
        :return: points which are inside the box.
        """

    @abstractmethod
    def get_leaf_points(self) -> List[DynamicVoxel]:
        """
        :return: List of voxels where each voxel represents a leaf node with points.
        """
        pass

    @abstractmethod
    def subdivide(self, subdivision_criteria: List[Callable[[RawPointCloud], bool]]):
        """
        Subdivide node based on the subdivision criteria.
        :param subdivision_criteria: list of criteria for subdivision
        """
        pass

    @abstractmethod
    def get_points(self) -> RawPointCloud:
        """
        :return: Points, which are stored inside the node.
        """
        pass


class OctreeBase(DynamicVoxel, ABC):
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
        corner_min: RawPoint,
        edge_length: np.float_,
    ):
        super().__init__(corner_min, edge_length)
        self._config = octree_config
        self._root = self._node_type(self.corner_min, self.edge_length)

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
    def filter(self, filtering_criteria: List[Callable[[RawPointCloud], bool]]):
        """
        filter nodes with points by criterion
        :param filtering_criteria:
        """
        pass

    @abstractmethod
    def map_leaf_points(self, function: Callable[[RawPointCloud], RawPointCloud]):
        """
        transform point cloud in each node using the function
        :param function: transformation function PointCloud -> PointCloud
        """
        pass

    @abstractmethod
    def get_points_in_box(self, box: Box) -> RawPointCloud:
        """
        Returns points that occupy the given box
        :param box: tuple of two points representing min and max points of the box
        :return: PointCloud
        """
        pass

    @abstractmethod
    def get_leaf_points(self) -> List[DynamicVoxel]:
        """
        :return: List of PointClouds where each PointCloud
        represents points in a separate leaf node
        """
        pass

    @abstractmethod
    def subdivide(self, subdivision_criteria: List[Callable[[RawPointCloud], bool]]):
        """
        Subdivide node based on the subdivision criteria.
        :param subdivision_criteria: list of criteria for subdivision
        """
        pass

    @abstractmethod
    def get_points(self) -> RawPointCloud:
        """
        :return: Points, which are stored inside the octree.
        """
        pass
