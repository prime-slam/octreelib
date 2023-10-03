from abc import ABC
from dataclasses import dataclass
from typing import Type

from internal import T
from octreelib.octree.octree_base import OctreeConfigBase

__all__ = ["GridConfigBase"]


@dataclass
class GridConfigBase(ABC):
    octree_type: Type[T]
    octree_config: OctreeConfigBase
    debug: bool = False
