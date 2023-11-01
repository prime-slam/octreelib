from typing import List, Dict, Callable, Any, Tuple

import numpy as np

from octreelib.grid.grid_base import GridBase, GridConfigBase

from octreelib.internal.point import RawPointCloud, RawPoint, PointCloud
from octreelib.internal.voxel import StaticStoringVoxel
from octreelib.octree.multi_pose_octree import MultiPoseOctree

__all__ = ["Grid", "GridConfig"]


class GridConfig(GridConfigBase):
    pass


class Grid(GridBase):
    """
    This class implements GridBase interface.
    The space is divided into the grid of voxels with the given edge_size.
    Each voxel in a grid stores points in the form of a MultiPoseOctree.
    """

    def __init__(self, grid_config: GridConfig):
        super().__init__(grid_config)

        # workaround for restricting the type of octree for this grid
        self.grid_config.octree_type = MultiPoseOctree

        # {pose -> list of voxel coordinates}
        self.pose_voxel_coordinates: Dict[int, List[RawPoint]] = {}

        # {voxel coordinates -> octree}
        self.octrees: Dict[Tuple[float, float, float], grid_config.octree_type] = {}

    def insert_points(self, pose_number: int, points: RawPointCloud):
        """
        Insert points to the according octree.
        If an octree for this pos does not exist, a new octree is created
        :param pose_number: pos to which the cloud is inserted
        :param points: point cloud
        """
        # register pose if it is not registered yet
        if pose_number not in self.pose_voxel_coordinates:
            self.pose_voxel_coordinates[pose_number] = []

        # convert points to PointsWithPose, which is a subclass of np.ndarray
        points = PointCloud(points)

        for point in points:
            # get coords of voxel into which the point is inserted
            voxel_coordinates = self._get_voxel_for_point(point)
            # voxel_coordinates_hashable = voxel_coordinates.tolist()
            voxel_coordinates_hashable = (
                float(voxel_coordinates[0]),
                float(voxel_coordinates[1]),
                float(voxel_coordinates[2]),
            )

            # create octree in the voxel if it does not exist yet
            if voxel_coordinates_hashable not in self.octrees:
                self.octrees[voxel_coordinates_hashable] = self.grid_config.octree_type(
                    self.grid_config.octree_config,
                    voxel_coordinates,
                    self.grid_config.grid_voxel_edge_length,
                )

            self.octrees[voxel_coordinates_hashable].insert_points(
                [point.with_pose(pose_number)]
            )

    def _get_voxel_for_point(self, point: RawPoint) -> RawPoint:
        """
        Method to get coordinates of a voxel where the given point would be stored.
        :param point: Point.
        :return: Corner of the voxel in the grid, where an appropriate octree for the point resides.
        """
        point = point[:3]
        grid_voxel_edge_length = self.grid_config.grid_voxel_edge_length
        return point // grid_voxel_edge_length * grid_voxel_edge_length

    def map_leaf_points(self, function: Callable[[RawPointCloud], RawPointCloud]):
        """
        Transforms point cloud in each leaf node of each octree using the function
        :param function: transformation function PointCloud -> PointCloud
        """
        for voxel_coordinates in self.octrees:
            self.octrees[voxel_coordinates].map_leaf_points(function)

    def get_leaf_points(self, pose_number: int) -> List[StaticStoringVoxel]:
        """
        :param pose_number: Pose number.
        :return: List of voxels. Each voxel is a representation of a leaf node.
        Each voxel has the same corner, edge_length and points as one of the leaf nodes.
        """
        return sum(
            [
                octree.get_leaf_points_for_pose(pose_number)
                for octree in self.octrees.values()
            ],
            [],
        )

    def get_points(self, pose_number: int) -> RawPointCloud:
        """
        :param pose_number: Pose number.
        :return: All points inside the grid.
        """
        return np.vstack(
            [
                octree.get_points_for_pose(pose_number)
                for octree in self.octrees.values()
            ]
        )

    def subdivide(self, subdivision_criteria: List[Callable[[RawPointCloud], bool]]):
        """
        Subdivides all octrees based on all points and given filtering criteria.
        :param subdivision_criteria: List of bool functions which represent criteria for subdivision.
        If any of the criteria returns **true**, the octree node is subdivided.
        """
        for voxel_coordinates in self.octrees:
            self.octrees[voxel_coordinates].subdivide(subdivision_criteria)

    def filter(self, filtering_criteria: List[Callable[[RawPointCloud], bool]]):
        """
        Filters nodes of each octree with points by criteria
        :param filtering_criteria: Filtering Criteria
        """
        for voxel_coordinates in self.octrees:
            self.octrees[voxel_coordinates].filter(filtering_criteria)

    def n_leaves(self, pose_number: int) -> int:
        """
        :param pose_number: Pose number.
        :return: Number of leaves in all octrees, which store points for given pose.
        """
        return sum(
            [octree.n_leaves_for_pose(pose_number) for octree in self.octrees.values()]
        )

    def n_points(self, pose_number: int) -> int:
        """
        :param pose_number: Pose number.
        :return: Number of points for given pose.
        """
        return sum(
            [octree.n_points_for_pose(pose_number) for octree in self.octrees.values()]
        )

    def n_nodes(self, pose_number: int) -> int:
        """
        :param pose_number: Pose number.
        :return: Number of nodes in all octrees, which store points for given pose
        (either themselves, or through their child nodes).
        """
        return sum(
            [octree.n_nodes_for_pose(pose_number) for octree in self.octrees.values()]
        )