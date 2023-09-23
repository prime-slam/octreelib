import dataclasses

from typing import Callable
from typing import List
from typing import Optional

from octreelib.internal import Voxel
from octreelib.internal.typing import PointCloud


@dataclasses.dataclass
class OcTreeNode(Voxel):
    """
    points: stores points of a node

    children: stores children of a node

    When subdivided, all points are **transferred** to children
    and are not stored in the parent node.
    """
    points: Optional[PointCloud]
    children: Optional[List["OcTreeNode"]]
    has_children: bool

    def insert_points(self, points: PointCloud):
        raise NotImplementedError

    def get_points(self) -> PointCloud:
        raise NotImplementedError

    def subdivide(self):
        raise NotImplementedError

    @property
    def n_nodes(self):
        """
        :return: number of nodes
        """
        raise NotImplementedError

    @property
    def n_leafs(self):
        """
        :return: number of leafs a.k.a. number of nodes which have points
        """
        return

    @property
    def n_points(self):
        """
        :return: number of points in the octree
        """
        return


class OcTree(Voxel):
    """
    Octree stores points of a **single** pos

    root: root node of an octree
    """

    root: OcTreeNode

    def __init__(self):
        raise NotImplementedError

    @property
    def n_nodes(self):
        """
        :return: number of nodes
        """
        raise NotImplementedError

    @property
    def n_leafs(self):
        """
        :return: number of leafs a.k.a. number of nodes which have points 
        """
        return

    @property
    def n_points(self):
        """
        :return: number of points in the octree
        """
        return

    def insert_points(self, points: PointCloud):
        raise NotImplementedError

    def get_points(self) -> PointCloud:
        raise NotImplementedError

    def subdivide(self, subdivision_criteria: List[Callable[[PointCloud], bool]]):
        raise NotImplementedError
