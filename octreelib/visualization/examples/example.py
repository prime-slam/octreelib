import numpy as np

import views
from grid import StaticGrid, StaticGridConfig
from octree import OctreeConfig, Octree

grid = StaticGrid(StaticGridConfig(Octree, OctreeConfig()))
points_0 = [
    np.array([0, 0, 1]),
    np.array([0, 0, 2]),
    np.array([0, 0, 3]),
    np.array([9, 9, 8]),
    np.array([9, 9, 9]),
]
grid.insert_points(0, points_0)
points_1 = [
    np.array([12 + 1, 0, 1]),
    np.array([12 + 4, 0, 2]),
    np.array([12 + 0, 2, 3]),
    np.array([12 + 9, 3, 8]),
    np.array([12 + 5, 9, 9]),
]
grid.insert_points(1, points_1)

grid.subdivide([lambda points: len(points) > 2])

views.visualize_grid(grid)
