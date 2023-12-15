# distutils: language=c++

from libcpp cimport bool as cpp_bool
from libcpp.string cimport string as cpp_string

# declare interface for C++ class for disc_integrals objects
cdef extern from "disc_integrals.h" nogil:
    cdef cppclass DiscIntegration:
        DiscIntegration(
            int, double*, double*, double*, double*,
            cpp_string, cpp_string, int, cpp_bool) except+
        double compute_first_disc_integral()
        double compute_second_disc_integral()
