import dataclasses


@dataclasses.dataclass
class OcTreeConfig:
    """
    Config for OcTree

    ____

    **max_depth:** max depth of an octree
    """

    max_depth: int
