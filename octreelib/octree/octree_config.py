import dataclasses


@dataclasses.dataclass
class OcTreeConfig:
    """
    Config for OcTree

    ____

    **max_depth:** max depth of an octre
    """
    max_depth: int
