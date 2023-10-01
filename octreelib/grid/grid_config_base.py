from abc import ABC
from dataclasses import dataclass

from internal import T
from octreelib.octree import OctreeConfigBase

__all__ = ["GridConfigBase"]


@dataclass
class GridConfigBase(ABC):
    octree_type: T
    octree_config: OctreeConfigBase
    debug: bool = False
