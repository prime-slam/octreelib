import dataclasses

from octreelib.octree import OctreeConfig


@dataclasses.dataclass
class GridConfig:
    octree_config: OctreeConfig
    debug: bool = False
