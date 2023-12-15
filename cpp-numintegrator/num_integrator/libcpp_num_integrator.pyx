# distutils: language=c++

from num_integrator.libs.libcpp_num_integrator cimport NumericalDoubleInputIntegrator
from libcpp cimport bool as cpp_bool
from libcpp.string cimport string as cpp_string
cimport cython
cimport numpy as np
import numpy as np


# c/cython function wrapper
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _integrate(
    int arrsize, cpp_string integration_method, cpp_bool equally_spaced,
    double[::1] integrand, double[::1] variable, int num_threads,
    cpp_bool parallel_comp
    ):
    # initialize the numerical integration object
    ni_ptr = new NumericalDoubleInputIntegrator(
        arrsize, &integrand[0], &variable[0], equally_spaced, integration_method,
        num_threads, parallel_comp
        )
    # perform numerical integration
    cdef double integration_result = ni_ptr.integrate()
    # delete heap allocated pointer objects
    del ni_ptr
    # return the integration result
    return integration_result

# define function that performs numerical integration using memoryviews
def integrate(
    str integration_method, cpp_bool equally_spaced,
    double[::1] integrand, double[::1] variable,
    int num_threads, cpp_bool parallel_comp
    ):
    # compute the array size
    cdef int arrsize = integrand.shape[0]
    # call the c/cython function wrapper
    return _integrate(
        arrsize, integration_method.encode('utf-8'), equally_spaced, integrand, variable,
        num_threads, parallel_comp
        )
