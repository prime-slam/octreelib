from typing import List, Callable, Optional, Dict

import numpy as np

from octreelib.internal.point import PointCloud, Point
from octreelib.internal.typing import T
from octreelib.internal.voxel import Voxel, VoxelBase
from octreelib.octree.octree import Octree, OctreeConfig

__all__ = ["OctreeManager"]


class OctreeManager(VoxelBase):
    """
    Octree manager which stores octrees for different poses.

    :param octree_config: Octree configuration.
    :param corner_min: Minimum corner of the node.
    :param edge_length: Edge length of the node.
    """

    def __init__(
        self,
        octree_type: T,
        octree_config: OctreeConfig,
        corner_min: Point,
        edge_length: float,
    ):
        super().__init__(corner_min, edge_length)
        self._octree_type = octree_type
        self._octree_config = octree_config
        self._octrees: Dict[int, octree_type] = {}
        # Empty octree for poses which are not present in the manager
        self.__empty_octree = self._octree_type(octree_config, corner_min, edge_length)

    def subdivide(
        self,
        subdivision_criteria: List[Callable[[PointCloud], bool]],
        pose_numbers: Optional[List[int]],
    ):
        """
        Subdivide all nodes based on the subdivision criteria.
        :param subdivision_criteria: list of criteria for subdivision
        :param pose_numbers: List of pose numbers to subdivide
        """
        if pose_numbers is None:
            pose_numbers = self._octrees.keys()

        scheme_octree = self._octree_type(
            self._octree_config, self._corner_min, self._edge_length
        )
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
        """
        Transform point cloud in the node using the function
        :param function: Function PointCloud -> PointCloud
        :param pose_numbers: List of pose numbers to transform
        """
        if pose_numbers is None:
            pose_numbers = self._octrees.keys()

        for pose_number in pose_numbers:
            self._octrees[pose_number].map_leaf_points(function)

    def filter(
        self,
        filtering_criteria: List[Callable[[PointCloud], bool]],
        pose_numbers: Optional[List[int]],
    ):
        """
        Filter nodes with points by filtering criteria
        :param filtering_criteria: List of filtering criteria functions
        :param pose_numbers: List of pose numbers to filter
        """
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
        return self._octrees.get(pose_number, self.__empty_octree).get_leaf_points()

    def get_points(self, pose_number: Optional[int] = None) -> PointCloud:
        """
        :param pose_number: Desired pose number.
        :return: Points for this pose which are stored inside the octree.
        """
        if pose_number is None:
            return np.vstack([octree.get_points() for octree in self._octrees.values()])
        return self._octrees.get(pose_number, self.__empty_octree).get_points()

    def n_points(self, pose_number: Optional[int] = None) -> int:
        """
        :param pose_number: Desired pose number.
        :return: Number of points for this pose inside this node.
        """
        if pose_number is None:
            return sum(octree.n_points() for octree in self._octrees.values())
        return self._octrees.get(pose_number, self.__empty_octree).n_points

    def n_leaves(self, pose_number: int) -> int:
        """
        :param pose_number: Desired pose number.
        :return: Number of leaves which store points for this pose.
        """
        return self._octrees.get(pose_number, self.__empty_octree).n_leaves

    def n_nodes(self, pose_number: int) -> int:
        """
        :param pose_number: Desired pose number.
        :return: Number of nodes (both leaves and not) which store points for this pose.
        """
        return self._octrees.get(pose_number, self.__empty_octree).n_nodes

    def insert_points(self, pose_number: int, points: PointCloud):
        """
        :param pose_number:
        :param points: Points to insert
        """
        if pose_number not in self._octrees:
            self._octrees[pose_number] = self._octree_type(
                self._octree_config, self._corner_min, self._edge_length
            )
        self._octrees[pose_number].insert_points(points)
