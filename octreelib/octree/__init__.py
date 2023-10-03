"""
This module contains base classes for Octree, OctreeNode and OctreeConfig
"""

import octreelib.octree.octree as octree_module
import octreelib.octree.octree_base as octree_base_module

from octreelib.octree.octree import *
from octreelib.octree.octree_base import *

__all__ = octree_base_module.__all__ + octree_module.__all__
