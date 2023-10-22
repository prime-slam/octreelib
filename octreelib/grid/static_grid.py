import numpy as np

from dataclasses import dataclass
from typing import List, Generic, Any, Callable, Type, Tuple, Dict

from octreelib.grid.grid_base import GridBase, GridConfigBase
from octreelib.internal import Point, T, PointCloud, StoringVoxel
from octreelib.internal.geometry import boxes_intersection


__all__ = ["StaticGrid", "StaticGridConfig"]


@dataclass
class StaticGridConfig(GridConfigBase):
    pass


class StaticGrid(GridBase):
    merged_octrees = None

    def __init__(self, grid_config: StaticGridConfig):
        super().__init__(grid_config)
        self.octrees: Dict[int, T] = {}

    def n_leafs(self, pose_number: int):
        return self.octrees[pose_number].n_leafs

    def n_nodes(self, pose_number: int):
        return self.octrees[pose_number].n_nodes

    def n_points(self, pose_number: int):
        return self.octrees[pose_number].n_points

    def merge(self, merger: Any):
        points = sum([octree.get_points() for octree in self.octrees.values()], [])
        self.merged_octrees = [self._make_octree(points)]

    def filter(self, filtering_criteria: List[Callable[[PointCloud], bool]]):
        for pose_number in self.octrees:
            self.octrees[pose_number].filter(filtering_criteria)
            if not all(
                [
                    criterion(self.octrees[pose_number].get_points)
                    for criterion in filtering_criteria
                ]
            ):
                self.octrees.pop(pose_number)

    def map_leaf_points(self, function: Callable[[PointCloud], PointCloud]):
        for pose_number in self.octrees:
            self.octrees[pose_number].map_leaf_points(function)

    def subdivide(self, subdivision_criteria: List[Callable[[PointCloud], bool]]):
        for pose_number in self.octrees:
            self.octrees[pose_number].subdivide(subdivision_criteria)

    def get_points(self, pose_number: int) -> List[Point]:
        return self.octrees[pose_number].get_points()

    def _floor_point(self, x: Point) -> Point:
        min_voxel_size = self.grid_config.grid_voxel_edge_length
        return x // min_voxel_size * min_voxel_size

    def _ceil_point(self, x: Point) -> Point:
        return self._floor_point(x) + 1

    def _grid_voxel_index_for_point(self, x: Point) -> Point:
        return (
            (x - self.grid_config.corner)
            // self.grid_config.grid_voxel_edge_length
            * np.ones(3)
        )

    def _extend_coordinates_to_a_voxel(
        self, point_1: Point, point_2: Point
    ) -> Tuple[Point, Point]:
        return self._floor_point(point_1), self._ceil_point(point_2)

    def _make_octree(self, points: List[Point]):
        min_point = np.array(
            [
                min(points, key=lambda point: point[0])[0],
                min(points, key=lambda point: point[1])[1],
                min(points, key=lambda point: point[2])[2],
            ]
        )
        max_point = np.array(
            [
                max(points, key=lambda point: point[0])[0],
                max(points, key=lambda point: point[1])[1],
                max(points, key=lambda point: point[2])[2],
            ]
        )
        min_point, max_point = self._extend_coordinates_to_a_voxel(min_point, max_point)
        edge_length = max(
            [
                float(max_point[0] - min_point[0]),
                float(max_point[1] - min_point[1]),
                float(max_point[2] - min_point[2]),
            ]
        )
        return self.grid_config.octree_type(
            self.grid_config.octree_config, min_point, edge_length
        )

    def insert_points(self, pose_number: int, points: List[Point]) -> None:
        if pose_number in self.octrees:
            raise ValueError(
                f"The pose number {pose_number} is already in the grid. You must insert into a different pose number."
            )
        self.octrees[pose_number] = self._make_octree(points)
        self.octrees[pose_number].insert_points(points)

    def _intersect_octree_pair(
        self, first_pos_number: int, second_pos_number: int
    ) -> PointCloud:
        first_octree = self.octrees[first_pos_number]
        second_octree = self.octrees[second_pos_number]

        intersection_box = boxes_intersection(
            first_octree.bounding_box,
            second_octree.bounding_box,
        )
        return first_octree.get_points_inside_box(
            intersection_box
        ) + second_octree.get_points_inside_box(intersection_box)

    def get_leaf_points(self, pose_number: int) -> List[StoringVoxel]:
        return self.octrees[pose_number].get_leaf_points()
