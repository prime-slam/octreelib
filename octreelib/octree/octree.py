import itertools

from dataclasses import dataclass
from typing import Callable, List, Generic

import numpy as np

from octreelib.internal import PointCloud, T, Voxel
from octreelib.octree.octree_base import OctreeBase, OctreeNodeBase, OctreeConfigBase

__all__ = ["OctreeNode", "Octree", "OctreeConfig"]


@dataclass
class OctreeConfig(OctreeConfigBase):
    pass


class OctreeNode(OctreeNodeBase):
    def subdivide(self, subdivision_criteria: List[Callable[[PointCloud], bool]]):
        """
        Subdivide node based on the subdivision criteria.
        :param subdivision_criteria: List of bool functions which represent criteria for subdivision.
        If any of the criteria returns **true**, the octree node is subdivided.
        """
        if any([criterion(self._points) for criterion in subdivision_criteria]):
            self._children = self._generate_children()
            self._has_children = True
            self.insert_points(self._points.copy())
            self._points = np.empty((0, 3), dtype=float)
            for child in self._children:
                child.subdivide(subdivision_criteria)

    def subdivide_as(self, other: "OctreeNode"):
        """
        Subdivide octree node using the subdivision scheme of a different octree node.
        :param other: Octree node to copy subdivision scheme from.
        """
        if other._has_children and not self._has_children:
            self._children = self._generate_children()
            self._has_children = True
            self.insert_points(self._points.copy())
            self._points = np.empty((0, 3), dtype=float)

        if other._has_children:
            for self_child, other_child in zip(self._children, other._children):
                self_child.subdivide_as(other_child)
        elif self._has_children:
            self._points = self.get_points()
            self._has_children = False
            for child in self._children:
                child._remove_from_cache()
            self._children = []

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
            # For all points calculate the voxel to insert in
            voxel_indices = (
                (points - self.corner_min) // (self.edge_length / 2)
            ).astype(int)

            # Create a unique identifier for each voxel based on its indices
            unique_voxel_indices, point_inverse_indices = np.unique(
                voxel_indices, axis=0, return_inverse=True
            )

            # Points are reordered based on the `point_inverse_indices`, so that they can be split
            # into groups of points, where each group is inserted into the corresponding voxel.
            # The indices for splitting are calculated using `np.cumsum()` based on the number
            # of points which would be distributed into each voxel.
            grouped_points = np.split(
                points[point_inverse_indices.argsort()],
                np.cumsum(np.bincount(point_inverse_indices))[:-1],
            )
            for unique_voxel_index, child_points in zip(
                unique_voxel_indices, grouped_points
            ):
                # Calculate the internal child id from its binary representation
                child_id = sum(
                    2**i * exists_offset
                    for i, exists_offset in enumerate(unique_voxel_index[::-1])
                )
                self._children[child_id].insert_points(child_points)
        else:
            self._points = np.vstack([self._points, points])

    def filter(self, filtering_criteria: List[Callable[[PointCloud], bool]]):
        """
        Filter nodes with points by filtering criteria
        :param filtering_criteria: List of bool functions which represent criteria for filtering.
            If any of the criteria returns **false**, the point cloud in octree leaf is removed.
        """
        if self._has_children:
            for child in self._children:
                child.filter(filtering_criteria)
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

    def apply_mask(self, mask: np.ndarray):
        """
        Apply mask to the point cloud in the octree node
        :param mask: Mask to apply
        """
        self._points = self._points[mask]

    @property
    def n_leaves(self):
        """
        :return: number of leaves a.k.a. number of nodes which have points
        """
        return (
            sum([child.n_leaves for child in self._children])
            if self._has_children
            else 1 if len(self._points) != 0 else 0
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

    def _generate_children(self):
        """
        Generate children of the node.
        """
        child_edge_length = self.edge_length / np.float_(2)
        children_corners_offsets = itertools.product([0, child_edge_length], repeat=3)
        self._cached_leaves.remove(self)
        return [
            OctreeNode(
                self.corner_min + offset,
                child_edge_length,
                self._cached_leaves,
            )
            for internal_position, offset in enumerate(children_corners_offsets)
        ]

    def _remove_from_cache(self):
        """
        Remove the node and its children from the cached leaves.
        """
        self._cached_leaves.remove(self)
        if self._has_children:
            for child in self._children:
                child._remove_from_cache()


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
        :param subdivision_criteria: List of bool functions which represent criteria for subdivision.
        If any of the criteria returns **true**, the octree node is subdivided.
        """
        self._root.subdivide(subdivision_criteria)

    def subdivide_as(self, other_octree: "Octree"):
        """
        Subdivide octree using the subdivision scheme of a different octree.
        :param other_octree: Octree to copy subdivision scheme from.
        """
        self._root.subdivide_as(other_octree._root)

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
        :param filtering_criteria: List of bool functions which represent criteria for filtering.
            If any of the criteria returns **false**, the point cloud in octree leaf is removed.
        """
        self._root.filter(filtering_criteria)

    def map_leaf_points(self, function: Callable[[PointCloud], PointCloud]):
        """
        transform point cloud in the node using the function
        :param function: transformation function PointCloud -> PointCloud
        """
        self._root.map_leaf_points(function)

    def get_leaf_points(self, non_empty: bool = True) -> List[Voxel]:
        """
        :param non_empty: If True, only non-empty leaf nodes are returned.
        :return: List of voxels where each voxel represents a leaf node with points.
        """
        if non_empty:
            return list(filter(lambda v: v.n_points != 0, self._cached_leaves))
        return self._cached_leaves

    def apply_mask(self, mask: np.ndarray):
        """
        Apply mask to the point cloud in the octree
        :param mask: Mask to apply
        """
        start_index = 0
        for leaf in filter(lambda v: v.n_points != 0, self._cached_leaves):
            n_points = leaf.n_points
            leaf.apply_mask(mask[start_index : start_index + n_points])
            start_index += n_points

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
