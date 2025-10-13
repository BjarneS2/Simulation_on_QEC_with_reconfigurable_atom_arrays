"""
from src import *        
    --> Only Lattice (because it's listed in __all__).
from src import Lattice  
    --> Only Lattice (standard explicit import, unaffected by __all__).
"""

from .surface_code import Lattice

__all__ = ["Lattice"]