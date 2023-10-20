from typing import List, Dict, Callable, Any, Tuple

from octreelib.grid.grid_base import GridBase, GridConfigBase
from octreelib.internal import Point, PointCloud, StaticStoringVoxel, PointWithPose

__all__ = ["GridWithPoints", "GridWithPointsConfig"]


class GridWithPointsConfig(GridConfigBase):
    pass


class GridWithPoints(GridBase):
    def n_leafs(self, pose_number: int) -> int:
        """
        :param pose_number: Pose number.
        :return: Number of leafs in all octrees, which store points for given pose.
        """
        return sum(
            [octree.n_leafs_for_pose(pose_number) for octree in self.octrees.values()]
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

    def subdivide(self, subdivision_criteria: List[Callable[[PointCloud], bool]]):
        """
        Subdivides all octrees based on all points and given filtering criteria.
        :param subdivision_criteria: List of bool functions which represent criteria for subdivision.
        If any of the criteria returns **true**, the octree node is subdivided.
        """
        for voxel_coordinates in self.octrees:
            self.octrees[voxel_coordinates].subdivide(subdivision_criteria)

    def filter(self, filtering_criteria: List[Callable[[PointCloud], bool]]):
        """
        Filters nodes of each octree with points by criteria
        :param filtering_criteria: Filtering Criteria
        """
        for voxel_coordinates in self.octrees:
            self.octrees[voxel_coordinates].filter(filtering_criteria)

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

    def map_leaf_points(self, function: Callable[[PointCloud], PointCloud]):
        """
        Transforms point cloud in each leaf node of each octree using the function
        :param function: transformation function PointCloud -> PointCloud
        """
        for voxel_coordinates in self.octrees:
            self.octrees[voxel_coordinates].map_leaf_points(function)

    def get_points(self, pose_number: int) -> List[Point]:
        return sum()

    def merge(self, merger: Any):
        raise NotImplementedError("This method is Not Supported")

    # method to get coordinates of a voxel where the given point would be
    def _get_voxel_for_point(self, point: Point) -> Point:
        """
        Method to get coordinates of a voxel where the given point would be stored.
        :param point: Point.
        :return: Corner of the voxel in the grid, where an appropriate octree for the point resides.
        """
        min_voxel_size = self.grid_config.min_voxel_size
        return point // min_voxel_size * min_voxel_size

    def insert_points(self, pose_number: int, points: List[Point]):
        # register pose if it is not registered yet
        if pose_number not in self.pose_voxel_coordinates:
            self.pose_voxel_coordinates[pose_number] = []

        points = [PointWithPose(point, pose_number) for point in points]

        for point in points:
            # get coords of voxel into which the point is inserted
            voxel_coordinates = self._get_voxel_for_point(point)
            voxel_coordinates_hashable = (
                int(voxel_coordinates[0]),
                int(voxel_coordinates[1]),
                int(voxel_coordinates[2]),
            )

            # create Dict[coordinates, octree] if it does not exist yes
            # if pose_number not in self.octrees:
            #     self.octrees = {}

            # create octree in the voxel if it does not exist yet
            if voxel_coordinates_hashable not in self.octrees:
                self.octrees[voxel_coordinates_hashable] = self.grid_config.octree_type(
                    self.grid_config.octree_config,
                    voxel_coordinates,
                    self.grid_config.min_voxel_size,
                )

            self.octrees[voxel_coordinates_hashable].insert_points([point])

    def __init__(self, grid_config: GridWithPointsConfig):
        super().__init__(grid_config)

        # {pose -> list of voxel coordinates}
        self.pose_voxel_coordinates: Dict[int, List[Point]] = {}

        # {voxel coordinates -> octree}
        self.octrees: Dict[Tuple[int, int, int], grid_config.octree_type] = {}
