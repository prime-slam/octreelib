"""
This module contains base classes for Grid and GridConfig
"""
import octreelib.grid.grid_config_base as grid_config_base_module
import octreelib.grid.grid_base as grid_base_module

from octreelib.grid.grid_config_base import *
from octreelib.grid.grid_base import *

__all__ = grid_config_base_module.__all__ + grid_base_module.__all__
