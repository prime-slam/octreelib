import itertools

from functools import reduce
from typing import List, Callable

import numpy as np

from octreelib.internal.geometry import point_is_inside_box
from octreelib.internal.voxel import StaticStoringVoxel
from octreelib.octree.octree import Octree, OctreeNode, OctreeConfig
from octreelib.internal.point import (
    CPosePointCloud,
    CPointCloud,
    CPosePoint,
    PointCloud,
)

__all__ = ["MultiPoseOctreeNode", "MultiPoseOctree", "MultiPoseOctreeConfig"]


class MultiPoseOctreeConfig(OctreeConfig):
    pass


def _filter_by_pose_number(
    pose_number: int, points: CPosePointCloud
) -> CPosePointCloud:
    return CPosePointCloud(points[points[:, 3] == pose_number])


class MultiPoseOctreeNode(OctreeNode):
    def get_leaf_points_for_pose(self, pose_number: int) -> List[StaticStoringVoxel]:
        # if node has children, return sum of children leaf voxels
        # else return voxel with points for this node
        if self.has_children:
            return sum(
                [
                    child.get_leaf_points_for_pose(pose_number)
                    for child in self.children
                ],
                [],
            )
        filtered_points = _filter_by_pose_number(pose_number, self.points)
        if len(filtered_points):
            return [
                StaticStoringVoxel(
                    self.corner,
                    self.edge_length,
                    filtered_points.without_poses(),
                )
            ]
        return []

    def get_points_for_pose(self, pose_number: int) -> PointCloud:
        # if node has children, return sum of points in children
        # else return self.points
        return (
            reduce(
                lambda points_a, points_b: points_a.extend(points_b),
                [child.get_points_for_pose(pose_number) for child in self.children],
            )
            if self.has_children
            else _filter_by_pose_number(pose_number, self.points).without_poses()
        )

    def n_points_for_pose(self, pose_number: int) -> int:
        # if node has children return sum of n_points_for_pose in children
        # else return number of points for this pose in self
        return (
            sum([child.n_points_for_pose(pose_number) for child in self.children])
            if self.has_children
            else len(_filter_by_pose_number(pose_number, self.points))
        )

    def n_leafs_for_pose(self, pose_number: int) -> int:
        # if node has children return sum of n_leafs_for_pose in children
        # else return 1 if this leaf has points for this pose else 0
        return (
            sum([child.n_leafs_for_pose(pose_number) for child in self.children])
            if self.has_children
            else len(_filter_by_pose_number(pose_number, self.points)) != 0
        )

    def n_nodes_for_pose(self, pose_number: int) -> int:
        # if node has children and any of the children has points for this pose
        #     return sum of n_nodes_for_pose for children + 1 (because this voxel also counts)
        # else return 1 if this leaf has points for this pose else 0
        n_nodes = (
            sum([child.n_nodes_for_pose(pose_number) for child in self.children])
            if self.has_children
            else None
        )
        if n_nodes:
            n_nodes += 1

        return (
            n_nodes
            if n_nodes is not None
            else len(_filter_by_pose_number(pose_number, self.points)) != 0
        )

    def map_leaf_points(self, function: Callable[[PointCloud], PointCloud]):
        if self.has_children:
            for child in self.children:
                child.map_leaf_points(function)
        elif len(self.points):
            new_points = self._empty_point_cloud
            pose_numbers = set(self.points.poses())
            for pose_number in pose_numbers:
                points = _filter_by_pose_number(pose_number, self.points)
                if len(points):
                    points = CPointCloud(function(points.without_poses())).with_pose(
                        pose_number
                    )
                    new_points = new_points.extend(points)

            self.points = new_points

    def subdivide(self, subdivision_criteria: List[Callable[[PointCloud], bool]]):
        if any(
            [
                criterion(self.points.without_poses())
                for criterion in subdivision_criteria
            ]
        ):
            # calculate child edge length and offsets for each child node
            child_edge_length = self.edge_length / np.float_(2)
            children_corners_offsets = itertools.product(
                [0, child_edge_length], repeat=3
            )

            # initialize children
            self.children = [
                MultiPoseOctreeNode(self.corner + offset, child_edge_length)
                for offset in children_corners_offsets
            ]
            self.has_children = True

            # reinsert points so that they are inserted into the child nodes
            self.insert_points(self.points.copy())
            self.points = self._empty_point_cloud

            # subdivide children
            for child in self.children:
                child.subdivide(subdivision_criteria)

    def insert_point(self, point: CPosePoint):
        if self.has_children:
            for child in self.children:
                if point_is_inside_box(point.without_pose(), child.bounding_box):
                    child.insert_points(point)
        else:
            self.points = self.points.extend(point)

    def insert_points(self, points: CPosePointCloud):
        if self.has_children:
            for point in points:
                self.insert_point(point)
        else:
            self.points = self.points.extend(points)

    @property
    def _empty_point_cloud(self) -> CPosePointCloud:
        return CPosePointCloud(np.empty((0, 4), dtype=float))


class MultiPoseOctree(Octree):
    _node_type = MultiPoseOctreeNode

    def get_leaf_points_for_pose(self, pose_number: int) -> List[StaticStoringVoxel]:
        return self.root.get_leaf_points_for_pose(pose_number)

    def get_points_for_pose(self, pose_number: int) -> PointCloud:
        return self.root.get_points_for_pose(pose_number)

    def n_points_for_pose(self, pose_number: int) -> int:
        return self.root.n_points_for_pose(pose_number)

    def n_leafs_for_pose(self, pose_number: int) -> int:
        return self.root.n_leafs_for_pose(pose_number)

    def n_nodes_for_pose(self, pose_number: int) -> int:
        return self.root.n_nodes_for_pose(pose_number)
