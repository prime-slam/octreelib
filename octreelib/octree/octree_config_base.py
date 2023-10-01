from dataclasses import dataclass

__all__ = ["OctreeConfigBase"]

from typing import Generic

from internal import T


@dataclass
class OctreeConfigBase(Generic[T]):
    """
    Config for OcTree
    """
    node_type: T
