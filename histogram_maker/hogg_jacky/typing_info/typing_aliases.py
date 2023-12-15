"""Python module defining typing aliases and specific types for the jackknife-likelihood-computing module.

Author: Jordan Van Beeck <jordanvanbeeck@hotmail.com>
"""
# import modules
from typing import TypeAlias
import numpy as np
import numpy.typing as npt
from collections.abc import Iterator


# define type aliases
B_arr: TypeAlias = npt.NDArray[np.bool_]
D_var: TypeAlias = np.float64 | float
I_var: TypeAlias = np.int32 | int
t_arr: TypeAlias = npt.NDArray
D_arr: TypeAlias = npt.NDArray[np.float64]
D_list: TypeAlias = list[D_var]
I_arr: TypeAlias = npt.NDArray[np.int32]
I_list: TypeAlias = list[I_var]
Dtype_var: TypeAlias = npt.DTypeLike
Iter_var: TypeAlias = Iterator[tuple[D_var, I_var]]
Iter_var3: TypeAlias = Iterator[tuple[D_var, I_var, D_var]]
