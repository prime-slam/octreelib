import itertools

from dataclasses import dataclass
from typing import Callable, List, Generic

import numpy as np

from octreelib.internal.geometry import point_is_inside_box
from octreelib.internal import RawPointCloud, RawPoint, T, StoringVoxel
from octreelib.internal.typing import Box
from octreelib.octree.octree_base import OctreeBase, OctreeNodeBase, OctreeConfigBase

__all__ = ["OctreeNode", "Octree", "OctreeConfig"]


@dataclass
class OctreeConfig(OctreeConfigBase):
    pass


class OctreeNode(OctreeNodeBase):
    def get_points_inside_box(self, box: Box) -> RawPointCloud:
        if self.has_children:
            return sum(
                [child.get_points_inside_box(box) for child in self.children], []
            )
        points_inside = np.empty((0, 3), dtype=float)
        for point in self.points:
            if point_is_inside_box(point, box):
                points_inside = np.vstack((points_inside, point))
        return points_inside

    def subdivide(self, subdivision_criteria: List[Callable[[RawPointCloud], bool]]):
        if any([criterion(self.points) for criterion in subdivision_criteria]):
            child_edge_length = self.edge_length / np.float_(2)
            children_corners_offsets = itertools.product(
                [0, child_edge_length], repeat=3
            )
            self.children = [
                OctreeNode(self.corner + offset, child_edge_length)
                for offset in children_corners_offsets
            ]
            self.has_children = True
            self.insert_points(self.points.copy())
            self.points = self._empty_point_cloud
            for child in self.children:
                child.subdivide(subdivision_criteria)

    def get_points(self) -> RawPointCloud:
        if not self.has_children:
            return self.points.copy()

        points = self._empty_point_cloud
        for child in self.children:
            points = np.vstack((points, child.get_points()))
        return points

    def insert_points(self, points: RawPointCloud):
        if self.has_children:
            for point in points:
                for child in self.children:
                    if point_is_inside_box(point, child.bounding_box):
                        child.insert_points(point)
        else:
            self.points = np.vstack((self.points, points))

    def filter(self, filtering_criteria: List[Callable[[RawPointCloud], bool]]):
        if self.has_children:
            for child in self.children:
                child.filter(filtering_criteria)
            if all([child.n_points == 0 for child in self.children]):
                self.children = []
                self.has_children = False
        elif not all([criterion(self.points) for criterion in filtering_criteria]):
            self.points = self._empty_point_cloud

    def map_leaf_points(self, function: Callable[[RawPointCloud], RawPointCloud]):
        if self.has_children:
            for child in self.children:
                child.map_leaf_points(function)
        elif self.points:
            self.points = function(self.points.copy())

    def get_leaf_points(self) -> List[StoringVoxel]:
        if self.has_children:
            return sum([child.get_leaf_points() for child in self.children], [])
        return [self] if len(self.points) else []

    @property
    def bounding_box(self):
        return self.corner, self.corner + np.ones(3) * self.edge_length

    @property
    def n_leaves(self):
        return (
            sum([child.n_leaves for child in self.children]) if self.has_children else 1
        )

    @property
    def n_nodes(self):
        return (
            len(self.children) + sum([child.n_nodes for child in self.children])
            if self.has_children
            else 1
        )

    @property
    def n_points(self):
        return (
            sum([child.n_points for child in self.children])
            if self.has_children
            else len(self.points)
        )


class Octree(OctreeBase, Generic[T]):
    _node_type = OctreeNode

    def get_points_in_box(self, box: Box) -> RawPointCloud:
        return self.root.get_points_inside_box(box)

    def subdivide(self, subdivision_criteria: List[Callable[[RawPointCloud], bool]]):
        self.root.subdivide(subdivision_criteria)

    def get_points(self) -> RawPointCloud:
        return self.root.get_points()

    def insert_points(self, points: RawPointCloud):
        self.root.insert_points(points)

    def filter(self, filtering_criteria: List[Callable[[RawPointCloud], bool]]):
        self.root.filter(filtering_criteria)

    def map_leaf_points(self, function: Callable[[RawPointCloud], RawPointCloud]):
        self.root.map_leaf_points(function)

    def get_leaf_points(self) -> List[StoringVoxel]:
        return self.root.get_leaf_points()

    @property
    def n_points(self):
        return self.root.n_points

    @property
    def n_leaves(self):
        return self.root.n_leaves

    @property
    def n_nodes(self):
        return self.root.n_nodes
