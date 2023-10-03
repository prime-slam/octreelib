from dataclasses import dataclass

import numpy as np

from . import GridConfigBase
from .grid_base import GridBase

from internal import Point, T, PointCloud
from typing import List, Generic, Any, Callable


__all__ = ["StaticGrid", "StaticGridConfig"]


@dataclass
class StaticGridConfig(GridConfigBase):
    pass


class StaticGrid(GridBase, Generic[T]):
    def merge(self, merger: Any):
        # 💀 ahh... fell kinda clueless. gonna do some research
        # meanwhile
        print(1)
        raise NotImplementedError

    def filter(self, finter_criterion: Callable[[PointCloud], bool]):
        for pos in self.octrees:
            self.octrees[pos].filter()
            if self.octrees[pos].n_points == 0:
                self.octrees.pop(pos)

    def subdivide(self, subdivision_criteria: List[Callable[[PointCloud], bool]]):
        for octree in self.octrees.values():
            octree.subdivide(subdivision_criteria)

    def get_points(self, pos: int) -> List[Point]:
        return self.octrees[pos].get_points()

    def _make_octree(self, points: List[Point]):
        min_x = min(points, key=lambda point: point[0])[0]
        min_y = min(points, key=lambda point: point[1])[1]
        min_z = min(points, key=lambda point: point[2])[2]
        max_x = max(points, key=lambda point: point[0])[0]
        max_y = max(points, key=lambda point: point[1])[1]
        max_z = max(points, key=lambda point: point[2])[2]
        corner = np.array([min_x, min_y, min_z])
        edge_length = max(
            [float(max_x - min_x), float(max_y - min_y), float(max_z - min_z)]
        )
        return self.grid_config.octree_type(
            self.grid_config.octree_config, corner, edge_length
        )

    def insert_points(self, pos: int, points: List[Point]) -> None:
        if pos in self.octrees:
            raise ValueError(
                f"The pose {pos} is already in the grid. You must insert into a different pose."
            )
        self.octrees[pos] = self._make_octree(points)
        self.octrees[pos].insert_points(points)
