"""Initialization file for the python module that contains a class to compute Hough functions and their derivatives.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# version + author
__version__ = '0.2.1'
__author__ = 'Jordan Van Beeck'


# import the HoughFunction class and its normalizing equivalent
from .hough_function import HoughFunction as HF
from .hough_normalizer import HoughNormalizer as HNorm


# make HF and HNorm available directly for import
__all__ = ['HF', 'HNorm']
