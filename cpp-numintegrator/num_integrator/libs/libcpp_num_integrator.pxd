# distutils: language=c++

from libcpp cimport bool as cpp_bool
from libcpp.string cimport string as cpp_string

# declare interface for C++ class that performs numerical_integration
cdef extern from "num_integration.h" nogil:
    cdef cppclass NumericalDoubleInputIntegrator:
        NumericalDoubleInputIntegrator(int, double*, double*, cpp_bool, cpp_string, int, cpp_bool) except+
        double integrate()
