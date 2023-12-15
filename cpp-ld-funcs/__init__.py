"""Initialization script for the python package that is interfaced with C++ limb-darkening function methods using Cython.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import logging module
import logging

# import classes directly from the modules
from .ld_funcs import ldf  # type: ignore


# set up the basic configuration for the root logger
logging.basicConfig(
    format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True,
    level=logging.INFO,
)
logger = logging.getLogger()


__all__ = ['ldf']


# test message
logger.debug('ld_funcs package imported')
