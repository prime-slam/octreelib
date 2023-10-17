from typing import List, Dict, Callable, Any, Tuple

from octreelib.grid.grid_base import GridBase, GridConfigBase
from octreelib.internal import Point, PointCloud

__all__ = ["GridWithPoints", "GridWithPointsConfig"]


class GridWithPointsConfig(GridConfigBase):
    pass


class GridWithPoints(GridBase):
    def n_leafs(self, pose_number: int):
        return sum([octree.n_leafs for octree in self.octrees[pose_number].values()])

    def n_points(self, pose_number: int):
        return sum([octree.n_points for octree in self.octrees[pose_number].values()])

    def n_nodes(self, pose_number: int):
        return sum([octree.n_nodes for octree in self.octrees[pose_number].values()])

    def subdivide(self, subdivision_criteria: List[Callable[[PointCloud], bool]]):
        for pose_number in self.octrees:
            for voxel_coordinates in self.octrees[pose_number]:
                self.octrees[pose_number][voxel_coordinates].subdivide(
                    subdivision_criteria
                )

    def filter(self, filtering_criteria: List[Callable[[PointCloud], bool]]):
        for pose_number in self.octrees:
            for voxel_coordinates in self.octrees[pose_number]:
                self.octrees[pose_number][voxel_coordinates].filter(filtering_criteria)

    def get_leaf_points(self, pose_number: int) -> List[PointCloud]:
        return sum(
            [octree.get_leaf_points() for octree in self.octrees[pose_number].values()],
            [],
        )

    def map_leaf_points(self, function: Callable[[PointCloud], PointCloud]):
        for pose_number in self.octrees:
            for voxel_coordinates in self.octrees[pose_number]:
                self.octrees[pose_number][voxel_coordinates].map_leaf_points(function)

    def get_points(self, pose_number: int) -> List[Point]:
        raise NotImplementedError("This method is Not Supported")

    def merge(self, merger: Any):
        raise NotImplementedError("This method is Not Supported")

    # method to get coordinates of a voxel where the given point would be
    def _get_voxel_for_point(self, point: Point) -> Point:
        min_voxel_size = self.grid_config.min_voxel_size
        return point // min_voxel_size * min_voxel_size

    def insert_points(self, pose_number: int, points: List[Point]) -> None:
        # register pose if it is not registered yet
        if pose_number not in self.pose_voxel_coordinates:
            self.pose_voxel_coordinates[pose_number] = []

        for point in points:
            # get coords of voxel into which the point is inserted
            voxel_coordinates = self._get_voxel_for_point(point)
            voxel_coordinates_hashable = (
                int(voxel_coordinates[0]),
                int(voxel_coordinates[1]),
                int(voxel_coordinates[2]),
            )

            # create Dict[coordinates, octree] if it does not exist yes
            if pose_number not in self.octrees:
                self.octrees[pose_number] = {}

            # create octree in the voxel if it does not exist yet
            if voxel_coordinates_hashable not in self.octrees[pose_number]:
                self.octrees[pose_number][
                    voxel_coordinates_hashable
                ] = self.grid_config.octree_type(
                    self.grid_config.octree_config,
                    voxel_coordinates,
                    self.grid_config.min_voxel_size,
                )

            self.octrees[pose_number][voxel_coordinates_hashable].insert_points([point])

    def __init__(self, grid_config: GridWithPointsConfig):
        super().__init__(grid_config)

        # {pose -> list of voxel coordinates}
        self.pose_voxel_coordinates: Dict[int, List[Point]] = {}

        # {voxel coordinates -> {pose_number -> octree}}
        self.octrees: Dict[
            int, Dict[Tuple[int, int, int], grid_config.octree_type]
        ] = {}
