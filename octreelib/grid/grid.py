import random
from typing import List, Dict, Callable, Optional

import k3d
import numpy as np

from octreelib.grid.grid_base import (
    GridBase,
    GridConfigBase,
    GridVisualizationType,
    VisualizationConfig,
)
from octreelib.internal.point import RawPointCloud, RawPoint
from octreelib.internal.voxel import Voxel
from octreelib.octree_manager import MultiPoseManager

__all__ = ["Grid", "GridConfig"]


class GridConfig(GridConfigBase):
    _compatible_octree_types = [MultiPoseManager]


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

        # {pose -> list of voxel coordinates}
        self.__pose_voxel_coordinates: Dict[int, List[RawPoint]] = {}

        # {voxel coordinates hash -> octree}
        self.__octrees: Dict[int, grid_config.octree_type] = {}

    def insert_points(self, pose_number: int, points: RawPointCloud):
        """
        Insert points to the grid
        :param pose_number: pose to which the cloud is inserted
        :param points: point cloud
        """
        if pose_number in self.__pose_voxel_coordinates:
            raise ValueError(f"Cannot insert points to existing pose {pose_number}")

        # register pose
        self.__pose_voxel_coordinates[pose_number] = []

        for point in points:
            # get coords of voxel into which the point is inserted
            voxel_coordinates = self.__get_voxel_for_point(point)
            voxel_coordinates_hash = hash(
                (voxel_coordinates[0], voxel_coordinates[1], voxel_coordinates[2])
            )

            # create octree in the voxel if it does not exist yet
            if voxel_coordinates_hash not in self.__octrees:
                self.__octrees[voxel_coordinates_hash] = self._grid_config.octree_type(
                    self._grid_config.octree_config,
                    voxel_coordinates,
                    self._grid_config.grid_voxel_edge_length,
                )

            self.__octrees[voxel_coordinates_hash].insert_points(
                point.reshape((1, 3)), pose_number
            )

    def map_leaf_points(self, function: Callable[[RawPointCloud], RawPointCloud]):
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
            [octree.get_leaf_points(pose_number) for octree in self.__octrees.values()],
            [],
        )

    def get_points(self, pose_number: int) -> RawPointCloud:
        """
        :param pose_number: Pose number.
        :return: All points inside the grid.
        """
        return np.vstack(
            [octree.get_points(pose_number) for octree in self.__octrees.values()]
        )

    def subdivide(
        self,
        subdivision_criteria: List[Callable[[RawPointCloud], bool]],
        pose_numbers: Optional[List[int]] = None,
    ):
        """
        Subdivides all octrees based on all points and given subdivision criteria.
        :param subdivision_criteria: List of bool functions which represent criteria for subdivision.
        If any of the criteria returns **true**, the octree node is subdivided.
        """
        for voxel_coordinates in self.__octrees:
            self.__octrees[voxel_coordinates].subdivide(
                subdivision_criteria, pose_numbers
            )

    def filter(self, filtering_criteria: List[Callable[[RawPointCloud], bool]]):
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

    def __get_voxel_for_point(self, point: RawPoint) -> RawPoint:
        """
        Method to get coordinates of a voxel where the given point would be stored.
        :param point: Point.
        :return: Corner of the voxel in the grid, where an appropriate octree for the point resides.
        """
        point = point[:3]
        grid_voxel_edge_length = self._grid_config.grid_voxel_edge_length
        return point // grid_voxel_edge_length * grid_voxel_edge_length
