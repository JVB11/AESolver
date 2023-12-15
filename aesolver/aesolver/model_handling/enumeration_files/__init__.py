"""Python initialization file for enumeration files used to load models.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# custom imports
from .enumeration_dimensions import DimGyre, ReImPoly
from .enumeration_read import EnumerationReadInfo as EnumReadInfo
from .enumeration_models import (
    GYREDetailFiles,
    GYRESummaryFiles,
    MESAProfileFiles,
    PolytropeModelFiles,
    PolytropeOscillationModelFiles,
)


# define version, author and auto-import packages
__version__ = '1.0.0'
__author__ = 'Jordan Van Beeck'
__all__ = [
    'DimGyre',
    'EnumReadInfo',
    'GYREDetailFiles',
    'GYRESummaryFiles',
    'MESAProfileFiles',
    'PolytropeModelFiles',
    'PolytropeOscillationModelFiles',
    'ReImPoly',
]
