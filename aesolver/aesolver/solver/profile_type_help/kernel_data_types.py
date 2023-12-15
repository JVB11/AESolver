"""Module containing typing objects shared among the kernel-plotting modules in submodule Solver.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""

# imports
import numpy as np
from typing import TypedDict, NotRequired


# define the full output dictionary for
# radial kernel plot data-gathering functions/methods
class RadialKernelDict(TypedDict):
    kernel: np.ndarray
    legend: list[str]
    symmetric: bool
    multiplying_factor: NotRequired[float]
    legend_title: NotRequired[str]
    s_e_multipliers: NotRequired[list[str]]


# define the full output dictionary for
# angular kernel plot data-gathering functions/methods
class AngularKernelDict(TypedDict):
    kernel: np.ndarray
    legend: list[str]
    symmetric: bool
    mul_with_mu: NotRequired[bool]
    div_by_sin: NotRequired[bool]
    multiplying_factor: NotRequired[float]
    legend_string: NotRequired[list[str]]
