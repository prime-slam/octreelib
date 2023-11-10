from __future__ import annotations

import itertools

from dataclasses import dataclass
from typing import Callable, List, Generic

import numpy as np

from octreelib.internal import PointCloud, T, Voxel, CloudManager
from octreelib.octree.octree_base import OctreeBase, OctreeNodeBase, OctreeConfigBase

__all__ = ["OctreeNode", "Octree", "OctreeConfig"]


@dataclass
class OctreeConfig(OctreeConfigBase):
    pass


class OctreeNode(OctreeNodeBase):
    def __generate_children(self):
        child_edge_length = self.edge_length / np.float_(2)
        children_corners_offsets = itertools.product([0, child_edge_length], repeat=3)
        return [
            OctreeNode(
                self.corner_min + offset,
                child_edge_length,
                8 * self._position + internal_position,
            )
            for internal_position, offset in enumerate(children_corners_offsets)
        ]

    def __unify(self):
        self._points = self.get_points()
        self._has_children = False
        self._children = []

    def subdivide(self, subdivision_criteria: List[Callable[[PointCloud], bool]]):
        """
        Subdivide node based on the subdivision criteria.
        :param subdivision_criteria: list of criteria for subdivision
        """
        if any([criterion(self._points) for criterion in subdivision_criteria]):
            self._children = self.__generate_children()
            self._has_children = True
            self.insert_points(self._points.copy())
            self._points = np.empty((0, 3), dtype=float)
            for child in self._children:
                child.subdivide(subdivision_criteria)

    def subdivide_as(self, other: OctreeNode):
        if other._has_children and not self._has_children:
            self._children = self.__generate_children()
            self._has_children = True
            self.insert_points(self._points.copy())
            self._points = np.empty((0, 3), dtype=float)

        if other._has_children:
            for self_child, other_child in zip(self._children, other._children):
                self_child.subdivide_as(other_child)
        elif self._has_children:
            self.__unify()

    def get_points(self) -> PointCloud:
        """
        :return: Points inside the octree node.
        """
        if not self._has_children:
            return self._points.copy()

        points = np.empty((0, 3), dtype=float)
        for child in self._children:
            points = np.vstack((points, child.get_points()))
        return points

    def insert_points(self, points: PointCloud):
        """
        :param points: Points to insert.
        """
        if self._has_children:
            clouds = CloudManager.distribute(points, self.corner_min, self.edge_length)
            for child, cloud in zip(self._children, clouds):
                child.insert_points(cloud)
        else:
            self._points = np.vstack([self._points, points])

    def filter(self, filtering_criteria: List[Callable[[PointCloud], bool]]):
        """
        Filter nodes with points by filtering criteria
        :param filtering_criteria: List of filtering criteria functions.
        """
        if self._has_children:
            for child in self._children:
                child.filter(filtering_criteria)
            if all([child.n_points == 0 for child in self._children]):
                self._children = []
                self._has_children = False
        elif not all([criterion(self._points) for criterion in filtering_criteria]):
            self._points = np.empty((0, 3), dtype=float)

    def map_leaf_points(self, function: Callable[[PointCloud], PointCloud]):
        """
        Transform point cloud in the node using the function.
        :param function: Transformation function PointCloud -> PointCloud.
        """
        if self._has_children:
            for child in self._children:
                child.map_leaf_points(function)
        elif len(self._points):
            self._points = function(self._points.copy())

    def get_leaf_points(self) -> List[Voxel]:
        """
        :return: List of voxels where each voxel represents a leaf node with points.
        """
        if self._has_children:
            return sum([child.get_leaf_points() for child in self._children], [])
        return (
            [Voxel(self.corner_min, self.edge_length, self._points)]
            if len(self._points)
            else []
        )

    @property
    def n_leaves(self):
        """
        :return: number of leaves a.k.a. number of nodes which have points
        """
        return (
            sum([child.n_leaves for child in self._children])
            if self._has_children
            else 1
            if len(self._points) != 0
            else 0
        )

    @property
    def n_nodes(self):
        """
        :return: number of nodes
        """
        return (
            sum([child.n_nodes for child in self._children]) + 1
            if self._has_children
            else 1
        )

    @property
    def n_points(self):
        """
        :return: number of points in the octree node
        """
        return (
            sum([child.n_points for child in self._children])
            if self._has_children
            else len(self._points)
        )


class Octree(OctreeBase, Generic[T]):
    """
    Stores points from a **single pose** in the form of an octree.

    :param octree_config: Configuration for the octree.
    :param corner: Min corner of the octree.
    :param edge_length: Edge length of the octree.
    """

    _node_type = OctreeNode

    def subdivide(self, subdivision_criteria: List[Callable[[PointCloud], bool]]):
        """
        Subdivide node based on the subdivision criteria.
        :param subdivision_criteria: list of criteria for subdivision
        """
        self._root.subdivide(subdivision_criteria)

    def subdivide_as(self, other: Octree):
        self._root.subdivide_as(other._root)

    def get_points(self) -> PointCloud:
        """
        :return: Points, which are stored inside the Octree.
        """
        return self._root.get_points()

    def insert_points(self, points: PointCloud):
        """
        :param points: Points to insert
        """
        self._root.insert_points(points)

    def filter(self, filtering_criteria: List[Callable[[PointCloud], bool]]):
        """
        Filter nodes with points by filtering criteria
        :param filtering_criteria: List of filtering criteria functions
        """
        self._root.filter(filtering_criteria)

    def map_leaf_points(self, function: Callable[[PointCloud], PointCloud]):
        """
        transform point cloud in the node using the function
        :param function: transformation function PointCloud -> PointCloud
        """
        self._root.map_leaf_points(function)

    def get_leaf_points(self) -> List[Voxel]:
        """
        :return: List of voxels where each voxel represents a leaf node with points.
        """
        return self._root.get_leaf_points()

    @property
    def n_points(self):
        """
        :return: number of points in the octree
        """
        return self._root.n_points

    @property
    def n_leaves(self):
        """
        :return: number of leaves a.k.a. number of nodes which have points
        """
        return self._root.n_leaves

    @property
    def n_nodes(self):
        """
        :return: number of nodes
        """
        return self._root.n_nodes
