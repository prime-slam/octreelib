import itertools

from dataclasses import dataclass
from typing import Callable, List, Generic

import numpy as np

from octreelib.internal import Box
from octreelib.internal import RawPointCloud, T, StoringVoxel
from octreelib.octree.octree_base import OctreeBase, OctreeNodeBase, OctreeConfigBase

__all__ = ["OctreeNode", "Octree", "OctreeConfig"]


@dataclass
class OctreeConfig(OctreeConfigBase):
    pass


class OctreeNode(OctreeNodeBase):
    def get_points_inside_box(self, box: Box) -> RawPointCloud:
        """
        Returns points that occupy the given box
        :param box: tuple of two points representing min and max points of the box
        :return: points which are inside the box.
        """
        if self._has_children:
            return sum(
                [child.get_points_inside_box(box) for child in self._children], []
            )
        points_inside = np.empty((0, 3), dtype=float)
        for point in self._points:
            if box.is_point_inside(point):
                points_inside = np.vstack((points_inside, point))
        return points_inside

    def subdivide(self, subdivision_criteria: List[Callable[[RawPointCloud], bool]]):
        """
        Subdivide node based on the subdivision criteria.
        :param subdivision_criteria: list of criteria for subdivision
        """
        if any([criterion(self._points) for criterion in subdivision_criteria]):
            child_edge_length = self.edge_length / np.float_(2)
            children_corners_offsets = itertools.product(
                [0, child_edge_length], repeat=3
            )
            self._children = [
                OctreeNode(self.corner + offset, child_edge_length)
                for offset in children_corners_offsets
            ]
            self._has_children = True
            self.insert_points(self._points.copy())
            self._points = self._empty_point_cloud
            for child in self._children:
                child.subdivide(subdivision_criteria)

    def get_points(self) -> RawPointCloud:
        """
        :return: Points inside the octree node.
        """
        if not self._has_children:
            return self._points.copy()

        points = self._empty_point_cloud
        for child in self._children:
            points = np.vstack((points, child.get_points()))
        return points

    def insert_points(self, points: RawPointCloud):
        """
        :param points: Points to insert.
        """
        if self._has_children:
            for point in points:
                for child in self._children:
                    if child.bounding_box.is_point_inside(point):
                        child.insert_points(point)
        else:
            self._points = np.vstack((self._points, points))

    def filter(self, filtering_criteria: List[Callable[[RawPointCloud], bool]]):
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
            self._points = self._empty_point_cloud

    def map_leaf_points(self, function: Callable[[RawPointCloud], RawPointCloud]):
        """
        Transform point cloud in the node using the function.
        :param function: Transformation function RawPointCloud -> RawPointCloud.
        """
        if self._has_children:
            for child in self._children:
                child.map_leaf_points(function)
        elif self._points:
            self._points = function(self._points.copy())

    def get_leaf_points(self) -> List[StoringVoxel]:
        """
        :return: List of voxels where each voxel represents a leaf node with points.
        """
        if self._has_children:
            return sum([child.get_leaf_points() for child in self._children], [])
        return [self] if len(self._points) else []

    @property
    def n_leaves(self):
        """
        :return: number of leaves a.k.a. number of nodes which have points
        """
        return (
            sum([child.n_leaves for child in self._children]) if self._has_children else 1
        )

    @property
    def n_nodes(self):
        """
        :return: number of nodes
        """
        return (
            len(self._children) + sum([child.n_nodes for child in self._children])
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
    _node_type = OctreeNode

    def get_points_in_box(self, box: Box) -> RawPointCloud:
        """
        Returns points that occupy the given box
        :param box: tuple of two points representing min and max points of the box
        :return: PointCloud
        """
        return self._root.get_points_inside_box(box)

    def subdivide(self, subdivision_criteria: List[Callable[[RawPointCloud], bool]]):
        """
        Subdivide node based on the subdivision criteria.
        :param subdivision_criteria: list of criteria for subdivision
        """
        self._root.subdivide(subdivision_criteria)

    def get_points(self) -> RawPointCloud:
        """
        :return: Points, which are stored inside the Octree.
        """
        return self._root.get_points()

    def insert_points(self, points: RawPointCloud):
        """
        :param points: Points to insert
        """
        self._root.insert_points(points)

    def filter(self, filtering_criteria: List[Callable[[RawPointCloud], bool]]):
        """
        Filter nodes with points by filtering criteria
        :param filtering_criteria: List of filtering criteria functions
        """
        self._root.filter(filtering_criteria)

    def map_leaf_points(self, function: Callable[[RawPointCloud], RawPointCloud]):
        """
        transform point cloud in the node using the function
        :param function: transformation function RawPointCloud -> RawPointCloud
        """
        self._root.map_leaf_points(function)

    def get_leaf_points(self) -> List[StoringVoxel]:
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
