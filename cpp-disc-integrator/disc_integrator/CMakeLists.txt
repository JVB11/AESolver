cmake_minimum_required(VERSION 3.22)
project("DiscInteg")

find_package(GTest REQUIRED)
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories("../../cpp-numintegrator/num_integrator"
    "../../cpp-ld-funcs/ld_funcs")
add_executable(a.out main.cpp disc_integrals.cpp disc_integrals_tests.cpp
../../cpp-numintegrator/num_integrator/num_integration.cpp
../../cpp-ld-funcs/ld_funcs/ld_functions.cpp)
target_link_libraries(a.out ${GTEST_LIBRARIES})

#ctest framework
enable_testing()
add_test(FirstDITrapzEddington a.out)
add_test(FirstDISimpsonEddington a.out)
add_test(SecondDITrapzEddington a.out)
add_test(SecondDISimpsonEddington a.out)
