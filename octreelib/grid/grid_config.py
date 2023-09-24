from dataclasses import dataclass

from octreelib.octree import OctreeConfig


@dataclass
class GridConfig:
    octree_config: OctreeConfig
    debug: bool = False
