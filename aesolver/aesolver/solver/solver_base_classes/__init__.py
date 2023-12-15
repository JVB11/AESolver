'''Python submodule containing base classes that generate grid- or profile-specific subclasses that are used to solve the AE equations to generate a grid of models or their coupling coefficient profiles.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
'''
# import base classes
from .quadratic_solver import QuadraticAESolver, PhysicsError, InlistError


# export the base classes
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = [
    'QuadraticAESolver', 'PhysicsError', 'InlistError'
]