from typing import List, Callable, Optional, Dict

import numpy as np

from octreelib.internal.voxel import Voxel, VoxelBase
from octreelib.internal.point import PointCloud, Point
from octreelib.octree.octree import Octree, OctreeConfig

__all__ = ["OctreeManager"]


class OctreeManager(VoxelBase):
    def __init__(
        self, octree_config: OctreeConfig, corner_min: Point, edge_length: float
    ):
        super().__init__(corner_min, edge_length)
        self._octree_config = octree_config
        self._octrees: Dict[int, Octree] = {}
        self._empty_octree = Octree(octree_config, corner_min, edge_length)

    def subdivide(
        self,
        subdivision_criteria: List[Callable[[PointCloud], bool]],
        pose_numbers: Optional[List[int]],
    ):
        if pose_numbers is None:
            pose_numbers = self._octrees.keys()

        scheme_octree = Octree(self._octree_config, self._corner_min, self._edge_length)
        scheme_octree.insert_points(
            np.vstack(
                [
                    self._octrees[pose_number].get_points()
                    for pose_number in pose_numbers
                ]
            )
        )
        scheme_octree.subdivide(subdivision_criteria)

        for pose_number in self._octrees:
            self._octrees[pose_number].subdivide_as(scheme_octree)

    def map_leaf_points(
        self,
        function: Callable[[PointCloud], PointCloud],
        pose_numbers: Optional[List[int]] = None,
    ):
        if pose_numbers is None:
            pose_numbers = self._octrees.keys()

        for pose_number in pose_numbers:
            self._octrees[pose_number].map_leaf_points(function)

    def filter(
        self,
        filtering_criteria: List[Callable[[PointCloud], bool]],
        pose_numbers: Optional[List[int]],
    ):
        if pose_numbers is None:
            pose_numbers = self._octrees.keys()

        for pose_number in pose_numbers:
            self._octrees[pose_number].filter(filtering_criteria)

    def get_leaf_points(self, pose_number: Optional[int] = None) -> List[Voxel]:
        """
        :param pose_number: Desired pose number.
        :return: List of leaf voxels with points for this pose.
        """
        if pose_number is None:
            return sum(
                [octree.get_leaf_points() for octree in self._octrees.values()], []
            )
        return self._octrees.get(pose_number, self._empty_octree).get_leaf_points()

    def get_points(self, pose_number: Optional[int] = None) -> PointCloud:
        """
        :param pose_number: Desired pose number.
        :return: Points for this pose which are stored inside the octree.
        """
        if pose_number is None:
            return np.vstack([octree.get_points() for octree in self._octrees.values()])
        return self._octrees.get(pose_number, self._empty_octree).get_points()

    def n_points(self, pose_number: Optional[int] = None) -> int:
        """
        :param pose_number: Desired pose number.
        :return: Number of points for this pose inside this node.
        """
        if pose_number is None:
            return sum(octree.n_points() for octree in self._octrees.values())
        return self._octrees.get(pose_number, self._empty_octree).n_points

    def n_leaves(self, pose_number: Optional[int] = None) -> int:
        """
        :param pose_number: Desired pose number.
        :return: Number of leaves which store points for this pose.
        """
        if pose_number is None:
            raise NotImplementedError
            # return sum(octree.n_leaves for octree in self._octrees.values())
        return self._octrees.get(pose_number, self._empty_octree).n_leaves

    def n_nodes(self, pose_number: Optional[int] = None) -> int:
        """
        :param pose_number: Desired pose number.
        :return: Number of nodes (both leaves and not) which store points for this pose.
        """
        if pose_number is None:
            raise NotImplementedError
            # return sum(octree.n_nodes for octree in self._octrees.values())
        return self._octrees.get(pose_number, self._empty_octree).n_nodes

    def insert_points(self, pose_number: int, points: PointCloud):
        """
        :param pose_number:
        :param points: Points to insert
        """
        if pose_number not in self._octrees:
            self._octrees[pose_number] = Octree(
                self._octree_config, self._corner_min, self._edge_length
            )
        self._octrees[pose_number].insert_points(points)
