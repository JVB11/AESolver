cmake_minimum_required(VERSION 3.22)
project("NumInteg")

find_package(GTest REQUIRED)
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(a.out main.cpp num_integration.cpp num_integration_tests.cpp)
target_link_libraries(a.out ${GTEST_LIBRARIES})

#ctest framework
enable_testing()
add_test(Trapz a.out)
add_test(Simpson a.out)
