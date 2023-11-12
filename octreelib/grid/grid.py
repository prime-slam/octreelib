import random
from typing import List, Dict, Callable, Optional, Tuple

import k3d
import numpy as np

from octreelib.grid.grid_base import (
    GridBase,
    GridConfigBase,
    GridVisualizationType,
    VisualizationConfig,
)
from octreelib.internal.point import PointCloud, Point
from octreelib.internal.voxel import Voxel, VoxelBase
from octreelib.octree_manager import OctreeManager

__all__ = ["Grid", "GridConfig"]


class GridConfig(GridConfigBase):
    _compatible_octree_manager_types = [OctreeManager]


class Grid(GridBase):
    """
    This class implements GridBase interface.
    The space is divided into the grid of voxels with the given edge size,
    which is defined in the GridConfig.
    Each voxel in a grid stores points in the form of a MultiPoseOctree.

    :param grid_config: GridConfig
    """

    def __init__(self, grid_config: GridConfig):
        super().__init__(grid_config)

        # {pose -> list of voxels}
        self.__pose_voxel_coordinates: Dict[int, List[VoxelBase]] = {}

        # {voxel -> octree manager}
        self.__octrees: Dict[VoxelBase, grid_config.octree_manager_type] = {}

    def insert_points(self, pose_number: int, points: PointCloud):
        """
        Insert points to the grid
        :param pose_number: pose to which the cloud is inserted
        :param points: point cloud
        """
        if pose_number in self.__pose_voxel_coordinates:
            raise ValueError(f"Cannot insert points to existing pose {pose_number}")

        # register pose
        self.__pose_voxel_coordinates[pose_number] = []

        # distribute points to voxels
        voxel_indices = (
            (
                (points - self._grid_config.corner)
                // self._grid_config.grid_voxel_edge_length
            )
            * self._grid_config.grid_voxel_edge_length
        ).astype(int)
        distributed_points = {}
        unique_indices = np.unique(voxel_indices, axis=0)
        for unique_id in unique_indices:
            mask = np.where((voxel_indices == unique_id).all(axis=1))
            distributed_points[tuple(unique_id)] = points[mask]

        # insert points to octrees
        for voxel_coordinates, voxel_points in distributed_points.items():
            target_voxel = VoxelBase(
                np.array(voxel_coordinates),
                self._grid_config.grid_voxel_edge_length,
            )
            if target_voxel not in self.__octrees:
                self.__octrees[target_voxel] = self._grid_config.octree_manager_type(
                    self._grid_config.octree_config,
                    np.array(voxel_coordinates),
                    self._grid_config.grid_voxel_edge_length,
                )

            self.__pose_voxel_coordinates[pose_number].append(target_voxel)
            self.__octrees[target_voxel].insert_points(pose_number, voxel_points)

    def map_leaf_points(self, function: Callable[[PointCloud], PointCloud]):
        """
        Transforms point cloud in each leaf node of each octree using the function
        :param function: transformation function PointCloud -> PointCloud
        """
        for voxel_coordinates in self.__octrees:
            self.__octrees[voxel_coordinates].map_leaf_points(function)

    def get_leaf_points(self, pose_number: int) -> List[Voxel]:
        """
        :param pose_number: Pose number.
        :return: List of voxels. Each voxel is a representation of a leaf node.
        Each voxel has the same corner, edge_length and points as one of the leaf nodes.
        """
        return sum(
            [
                self.__octrees[voxel_coordinates].get_leaf_points(pose_number)
                for voxel_coordinates in self.__pose_voxel_coordinates[pose_number]
            ],
            [],
        )

    def get_points(self, pose_number: int) -> PointCloud:
        """
        :param pose_number: Pose number.
        :return: All points inside the grid.
        """
        return np.vstack(
            [octree.get_points(pose_number) for octree in self.__octrees.values()]
        )

    def subdivide(
        self,
        subdivision_criteria: List[Callable[[PointCloud], bool]],
        pose_numbers: Optional[List[int]] = None,
    ):
        """
        Subdivides all octrees based on all points and given subdivision criteria.
        :param pose_numbers: List of pose numbers to subdivide.
        :param subdivision_criteria: List of bool functions which represent criteria for subdivision.
        If any of the criteria returns **true**, the octree node is subdivided.
        """
        for voxel_coordinates in self.__octrees:
            self.__octrees[voxel_coordinates].subdivide(
                subdivision_criteria, pose_numbers
            )

    def filter(self, filtering_criteria: List[Callable[[PointCloud], bool]]):
        """
        Filters nodes of each octree with points by criteria
        :param filtering_criteria: Filtering Criteria
        """
        for voxel_coordinates in self.__octrees:
            self.__octrees[voxel_coordinates].filter(filtering_criteria)

    def visualize(self, config: VisualizationConfig = VisualizationConfig()) -> None:
        """
        Produces `.html` file with Grid
        """
        plot = k3d.Plot()
        random.seed(config.seed)
        poses_number = len(self.__octrees.keys()) + 1

        if config.type is GridVisualizationType.POSE:
            for pose_number in range(poses_number):
                color = random.randrange(0, 0xFFFFFF)
                points = self.get_points(pose_number=pose_number)

                plot += k3d.points(
                    positions=points,
                    point_size=config.point_size,
                    color=color,
                )
        elif config.type is GridVisualizationType.VOXEL:
            voxels_colors = {}
            for pose_number in range(poses_number):
                leaves = self.get_leaf_points(pose_number=pose_number)
                for leaf in leaves:
                    if leaf.id not in voxels_colors.keys():
                        color = random.randrange(0, 0xFFFFFF)
                        voxels_colors[leaf.id] = color

                    plot += k3d.points(
                        positions=leaf.get_points(),
                        point_size=config.point_size,
                        color=voxels_colors[leaf.id],
                    )

        vertices = []
        for pose_number in range(poses_number):
            # TODO: Draw full grid including empty voxels
            leaves = self.get_leaf_points(pose_number=pose_number)
            for leaf in leaves:
                vertices.append(leaf.all_corners)

        for vertex in vertices:
            plot += k3d.lines(
                vertices=vertex,
                # Represents tracing of voxel corners
                # Each line - separate face of the voxel
                indices=[
                    [0, 2, 2, 6, 6, 4, 4, 0],  # Yeah, that's weird
                    [0, 1, 1, 5, 5, 4, 4, 0],  # but I didn't invent other way
                    [0, 1, 1, 3, 3, 2, 2, 0],  # to draw voxels edges
                    [1, 3, 3, 7, 7, 5, 5, 1],
                    [2, 3, 3, 7, 7, 6, 6, 2],
                    [4, 5, 5, 7, 7, 6, 6, 4],
                ],
                width=config.line_width_size,
                color=config.line_color,
                indices_type="segment",
            )

        with open(config.filepath, "w") as f:
            f.write(plot.get_snapshot())

    def n_leaves(self, pose_number: int) -> int:
        """
        :param pose_number: Pose number.
        :return: Number of leaves in all octrees, which store points for given pose.
        """
        return sum([octree.n_leaves(pose_number) for octree in self.__octrees.values()])

    def n_points(self, pose_number: int) -> int:
        """
        :param pose_number: Pose number.
        :return: Number of points for given pose.
        """
        return sum([octree.n_points(pose_number) for octree in self.__octrees.values()])

    def n_nodes(self, pose_number: int) -> int:
        """
        :param pose_number: Pose number.
        :return: Number of nodes in all octrees, which store points for given pose
        (either themselves, or through their child nodes).
        """
        return sum([octree.n_nodes(pose_number) for octree in self.__octrees.values()])

    def __get_voxel_for_point(self, point: Point) -> Point:
        """
        Method to get coordinates of a voxel where the given point would be stored.
        :param point: Point.
        :return: Corner of the voxel in the grid, where an appropriate octree for the point resides.
        """
        point = point[:3]
        grid_voxel_edge_length = self._grid_config.grid_voxel_edge_length
        return point // grid_voxel_edge_length * grid_voxel_edge_length
