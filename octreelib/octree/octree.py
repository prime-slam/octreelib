import itertools

from dataclasses import dataclass
from typing import Callable, List, Generic

import numpy as np

from octreelib.internal import PointCloud, Point, T
from octreelib.octree.octree_base import OctreeBase, OctreeNodeBase, OctreeConfigBase

__all__ = ["OctreeNode", "Octree", "OctreeConfig"]


@dataclass
class OctreeConfig(OctreeConfigBase):
    pass


class OctreeNode(OctreeNodeBase):
    def _point_is_inside(self, point: Point) -> bool:
        return bool((self.corner <= point).all()) and bool(
            (point <= (self.corner + self.edge_length)).all()
        )

    def subdivide(self, subdivision_criteria: List[Callable[[PointCloud], bool]]):
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
            self.points = []
            for child in self.children:
                child.subdivide(subdivision_criteria)

    def get_points(self) -> PointCloud:
        return (
            sum([child.get_points() for child in self.children], [])
            if self.has_children
            else self.points
        )

    def insert_points(self, points: PointCloud):
        if self.has_children:
            for point in points:
                for child in self.children:
                    if child._point_is_inside(point):
                        child.insert_points([point])
        else:
            self.points.extend(points)

    def filter(self, filtering_criteria: List[Callable[[PointCloud], bool]]):
        if self.has_children:
            for child in self.children:
                child.filter(filtering_criteria)
            if all([child.n_points == 0 for child in self.children]):
                self.children = []
                self.has_children = False
        elif not all([criterion(self.points) for criterion in filtering_criteria]):
            self.points = []

    def transform_point_cloud(self, function: Callable[[PointCloud], PointCloud]):
        if self.has_children:
            for child in self.children:
                child.transform_point_cloud(function)
        elif self.points:
            self.points = function(self.points.copy())

    @property
    def n_leafs(self):
        return (
            sum([child.n_leafs for child in self.children]) if self.has_children else 1
        )

    @property
    def n_nodes(self):
        return len(self.children) if self.has_children else 0

    @property
    def n_points(self):
        return (
            sum([child.n_points for child in self.children])
            if self.has_children
            else len(self.points)
        )


class Octree(OctreeBase, Generic[T]):
    _node_type = OctreeNode

    def subdivide(self, subdivision_criteria: List[Callable[[PointCloud], bool]]):
        self.root.subdivide(subdivision_criteria)

    def get_points(self) -> PointCloud:
        return self.root.get_points()

    def insert_points(self, points: PointCloud):
        self.root.insert_points(points)

    def filter(self, filtering_criteria: List[Callable[[PointCloud], bool]]):
        self.root.filter(filtering_criteria)

    def transform_point_clouds(self, function: Callable[[PointCloud], PointCloud]):
        self.root.transform_point_clouds(function)

    @property
    def n_points(self):
        return self.root.n_points

    @property
    def n_leafs(self):
        return self.root.n_leafs

    @property
    def n_nodes(self):
        return self.root.n_node
