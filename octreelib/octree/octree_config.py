import abc
import dataclasses


@dataclasses.dataclass
class OctreeConfig:
    """
    Config for OcTree

    ____

    **max_depth:** max depth of an octree
    """

