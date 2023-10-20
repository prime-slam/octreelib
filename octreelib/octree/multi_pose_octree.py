import itertools
from typing import List, Callable

import numpy as np

from octreelib.internal.voxel import StaticStoringVoxel
from octreelib.octree.octree import Octree, OctreeNode, OctreeConfig
from octreelib.internal.typing import PointCloud
from octreelib.internal.point import PointWithPose

__all__ = ["MultiPoseOctreeNode", "MultiPoseOctree", "MultiPoseOctreeConfig"]


class MultiPoseOctreeConfig(OctreeConfig):
    pass


def _filter_by_pose_number(pose_number: int, points: List[PointWithPose]) -> PointCloud:
    return list(filter(lambda point: point.pose_number == pose_number, points))


class MultiPoseOctreeNode(OctreeNode):
    def get_leaf_points_for_pose(self, pose_number: int) -> List[StaticStoringVoxel]:
        return (
            sum(
                [
                    child.get_leaf_points_for_pose(pose_number)
                    for child in self.children
                ],
                [],
            )
            if self.has_children
            else [
                StaticStoringVoxel(
                    self.corner,
                    self.edge_length,
                    _filter_by_pose_number(pose_number, self.points),
                )
            ]
        )

    def get_points_for_pose(self, pose_number: int) -> PointCloud:
        return (
            sum(
                [child.get_points_for_pose(pose_number) for child in self.children],
                [],
            )
            if self.has_children
            else _filter_by_pose_number(pose_number, self.points)
        )

    def n_points_for_pose(self, pose_number: int) -> int:
        return (
            sum([child.n_points_for_pose(pose_number) for child in self.children])
            if self.has_children
            else len(_filter_by_pose_number(pose_number, self.points))
        )

    def n_leafs_for_pose(self, pose_number: int) -> int:
        return (
            sum([child.n_leafs_for_pose(pose_number) for child in self.children])
            if self.has_children
            else len(_filter_by_pose_number(pose_number, self.points)) != 0
        )

    def n_nodes_for_pose(self, pose_number: int) -> int:
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

    def subdivide(self, subdivision_criteria: List[Callable[[PointCloud], bool]]):
        if any([criterion(self.points) for criterion in subdivision_criteria]):
            child_edge_length = self.edge_length / np.float_(2)
            children_corners_offsets = itertools.product(
                [0, child_edge_length], repeat=3
            )
            self.children = [
                MultiPoseOctreeNode(self.corner + offset, child_edge_length)
                for offset in children_corners_offsets
            ]
            self.has_children = True
            self.insert_points(self.points.copy())
            self.points = []
            for child in self.children:
                child.subdivide(subdivision_criteria)


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
