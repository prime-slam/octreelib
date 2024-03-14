from dataclasses import dataclass
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
from octreelib.internal.point import PointCloud
from octreelib.internal.voxel import Voxel, VoxelBase
from octreelib.ransac.cuda_ransac import CudaRansac

__all__ = ["Grid", "GridConfig"]


@dataclass
class GridConfig(GridConfigBase):
    """
    Config for Grid

    octree_manager_type: type of OctreeManager used.
        OctreeManager is responsible for managing octrees for different poses.
    octree_type: type of Octree used
        Octree is the data structure used for storing points.
    octree_config: This config will be forwarded to the octrees.
    debug: True enables debug mode.
    voxel_edge_length: Initial size of voxels.
    corner: Corner of a grid.
    """

    pass


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
        Insert points to the according octree.
        If an octree for this pose does not exist, a new octree is created
        :param pose_number: Pose number to which the cloud is inserted.
        :param points: Point cloud to be inserted.
        """
        # if pose_number in self.__pose_voxel_coordinates:
        #     raise ValueError(f"Cannot insert points to existing pose {pose_number}")

        # Register pose
        self.__pose_voxel_coordinates[pose_number] = []

        # Distribute points to voxels
        voxel_indices = (
            (points - self._grid_config.corner)
            // self._grid_config.voxel_edge_length
            * self._grid_config.voxel_edge_length
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

        # Insert points to octrees
        for voxel_coordinates, voxel_points in zip(
            unique_voxel_indices, grouped_points
        ):
            target_voxel = VoxelBase(
                np.array(voxel_coordinates),
                self._grid_config.voxel_edge_length,
            )
            if target_voxel not in self.__octrees:
                self.__octrees[target_voxel] = self._grid_config.octree_manager_type(
                    self._grid_config.octree_type,
                    self._grid_config.octree_config,
                    np.array(voxel_coordinates),
                    self._grid_config.voxel_edge_length,
                )

            self.__pose_voxel_coordinates[pose_number].append(target_voxel)
            self.__octrees[target_voxel].insert_points(pose_number, voxel_points)

    def map_leaf_points(
        self,
        function: Callable[[PointCloud], PointCloud],
        pose_numbers: Optional[List[int]] = None,
    ):
        """
        Transform point cloud in each node of each octree using the function
        :param function: Transformation function PointCloud -> PointCloud. It is applied to each leaf node.
        :param pose_numbers: List of pose numbers to map.
        """
        for voxel_coordinates in self.__octrees:
            self.__octrees[voxel_coordinates].map_leaf_points(function, pose_numbers)

    def map_leaf_points_cuda(self, n_poses_per_batch: int = 10):
        """
        transform point cloud in the node using the function
        """

        # processing is done in batches to avoid running out of memory
        batches = [
            list(
                range(
                    i,
                    min(i + n_poses_per_batch, len(self.__pose_voxel_coordinates)),
                )
            )
            for i in range(0, len(self.__pose_voxel_coordinates), n_poses_per_batch)
        ]

        # find the maximum number of leaf voxels across all batches
        # this is needed to initialize the random number generators on the GPU
        max_leaf_voxels = max(
            [self.sum_of_leaves(batch_pose_numbers) for batch_pose_numbers in batches]
        )
        ransac = CudaRansac(max_n_blocks=max_leaf_voxels, n_threads_per_block=1024)

        # process each batch
        for batch_pose_numbers in batches:
            # `combined_point_cloud` is a concatenation of ALL point clouds
            # `block_sizes` is a list of sizes of point clouds for each leaf node
            # `pose_dividers` is a list of indices where combined_point_cloud is divided by pose
            # these are used to split the combined_point_cloud into separate point clouds
            # for each pose after the kernel is done
            point_clouds = []
            block_sizes = []
            pose_dividers = [0]
            for pose_number in batch_pose_numbers:
                combined_point_cloud = self.get_points(pose_number)
                point_clouds.append(combined_point_cloud)
                block_sizes.append(
                    np.array(
                        [
                            len(v.get_points())
                            for v in self.get_leaf_points(pose_number)
                        ],
                        dtype=np.int32,
                    )
                )
                pose_dividers.append(pose_dividers[-1] + block_sizes[-1].sum())

            combined_point_cloud = np.vstack(point_clouds)
            block_sizes = np.concatenate(block_sizes)
            block_start_indices = np.cumsum(np.concatenate(([0], block_sizes[:-1])))
            pose_dividers = np.cumsum(pose_dividers)

            # run the kernel
            maximum_mask = ransac.evaluate(
                combined_point_cloud,
                block_sizes,
                block_start_indices,
            )

            # split the combined point cloud into separate point clouds for each pose,
            # apply the masks from the kernel
            # and insert them into the octrees
            self.map_leaf_points(
                lambda x: np.empty((0, 3), dtype=float), pose_numbers=batch_pose_numbers
            )
            for i, pose_number in enumerate(batch_pose_numbers):
                mask = maximum_mask[pose_dividers[i] : pose_dividers[i + 1]]
                point_cloud = combined_point_cloud[
                    pose_dividers[i] : pose_dividers[i + 1]
                ]
                self.insert_points(pose_number, point_cloud[mask == 1])

    def get_leaf_points(self, pose_number: int, non_empty: bool = True) -> List[Voxel]:
        """
        :param pose_number: The desired pose number.
        :param non_empty: If True, returns only non-empty voxels.
        :return: List of voxels. Each voxel is a representation of a leaf node.
        Each voxel has the same corner, edge_length and points as one of the leaf nodes.
        """
        return sum(
            [
                self.__octrees[voxel_coordinates].get_leaf_points(
                    non_empty, pose_number
                )
                for voxel_coordinates in self.__pose_voxel_coordinates[pose_number]
            ],
            [],
        )

    def get_points(self, pose_number: int) -> PointCloud:
        """
        Returns points for a specific pose number.
        :param pose_number: The desired pose number.
        :return: Points belonging to the pose.
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
        Filters nodes of each octree with points by criterion
        :param filtering_criteria: List of bool functions which represent criteria for filtering.
            If any of the criteria returns **false**, the point cloud in octree leaf is removed.
        """
        for voxel_coordinates in self.__octrees:
            self.__octrees[voxel_coordinates].filter(filtering_criteria)

    def visualize(self, config: VisualizationConfig = VisualizationConfig()) -> None:
        """
        Produces `.html` file with Grid
        """
        plot = k3d.Plot()
        random.seed(config.seed)
        poses_numbers = self.__pose_voxel_coordinates.keys()
        unused_voxel_color = 0x000000  # Black

        if config.type is GridVisualizationType.POSE:
            for pose_number in poses_numbers:
                color = random.randrange(0, 0xFFFFFF)
                leaves = self.get_leaf_points(pose_number=pose_number)
                for leaf in leaves:
                    if leaf.id in config.unused_voxels:
                        plot += k3d.points(
                            positions=leaf.get_points(),
                            point_size=config.point_size,
                            color=unused_voxel_color,
                        )

                        continue

                    plot += k3d.points(
                        positions=leaf.get_points(),
                        point_size=config.point_size,
                        color=color,
                    )
        elif config.type is GridVisualizationType.VOXEL:
            voxels_colors = {}
            for pose_number in poses_numbers:
                leaves = self.get_leaf_points(pose_number=pose_number)
                for leaf in leaves:
                    if leaf.id not in voxels_colors.keys():
                        color = random.randrange(0, 0xFFFFFF)
                        if leaf.id in config.unused_voxels:
                            color = unused_voxel_color

                        voxels_colors[leaf.id] = color

                    plot += k3d.points(
                        positions=leaf.get_points(),
                        point_size=config.point_size,
                        color=voxels_colors[leaf.id],
                    )

        vertices = []
        for pose_number in poses_numbers:
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
        :param pose_number: The desired pose number.
        :return: Number of leaf nodes in the octree for given pose number.
        """
        return sum([octree.n_leaves(pose_number) for octree in self.__octrees.values()])

    def n_points(self, pose_number: int) -> int:
        """
        :param pose_number: The desired pose number.
        :return: Number of points of an octree for given pose number.
        """
        return sum([octree.n_points(pose_number) for octree in self.__octrees.values()])

    def n_nodes(self, pose_number: int) -> int:
        """
        :param pose_number: The desired pose number.
        :return: Number of nodes of an octree for given pose number.
        """
        return sum([octree.n_nodes(pose_number) for octree in self.__octrees.values()])

    def sum_of_leaves(self, pose_numbers: Optional[List[int]] = None) -> int:
        return sum(
            [octree.sum_of_leaves(pose_numbers) for octree in self.__octrees.values()]
        )
