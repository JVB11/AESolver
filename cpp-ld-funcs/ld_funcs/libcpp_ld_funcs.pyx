# distutils: language=c++

from ld_funcs.libs.libcpp_ld_funcs cimport LimbDarkeningFunctions
from libcpp cimport bool as cpp_bool
from libcpp.string cimport string as cpp_string
cimport cython
cimport numpy as np
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _comp(
    int asize, double[:] kap, double[:] res,
    cpp_string myselvar, int num_threads, cpp_bool paral
    ):
    # create pointer to LimbDarkeningFunctions object
    ld_ptr = new LimbDarkeningFunctions(
        asize, &kap[0], &res[0], myselvar, num_threads, paral
        )
    # compute the LD function and store result in passed array
    ld_ptr.compute()
    # delete heap allocated pointer objects
    del ld_ptr


def compute(
    double[:] kap, double[:] res,
    str myselvar, int num_threads, cpp_bool paral
    ):
    # compute the array size
    cdef int asize = kap.shape[0]
    # compute the ld function
    _comp(
        asize, kap, res, myselvar.encode('utf-8'),
        num_threads, paral
        )
