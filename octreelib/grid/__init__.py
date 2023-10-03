"""
This module contains base classes for Grid and GridConfig
"""
import octreelib.grid.grid_base as grid_base_module
import octreelib.grid.static_grid as static_grid_module

from octreelib.grid.grid_base import *
from octreelib.grid.static_grid import *

__all__ = grid_base_module.__all__ + static_grid_module.__all__
