# distutils: language=c++

from libcpp cimport bool as cpp_bool
from libcpp.string cimport string as cpp_string

# declare interface for C++ class for LimbDarkeningFunctions objects
cdef extern from "ld_functions.h" nogil:
    cdef cppclass LimbDarkeningFunctions:
        LimbDarkeningFunctions(int arrsize, double* mus, double* lds, cpp_string selvar, int num_threads, cpp_bool paral) except+
        void compute()
