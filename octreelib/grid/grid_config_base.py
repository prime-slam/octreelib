from abc import ABC
from dataclasses import dataclass

from octreelib.octree import OctreeConfigBase

__all__ = ["GridConfigBase"]


@dataclass
class GridConfigBase(ABC):
    octree_config: OctreeConfigBase
    debug: bool = False
