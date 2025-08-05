cmake_minimum_required(VERSION 3.15)
project(StochasticLA VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compile options for better debugging and optimization
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find Eigen3 - handle both standard installations and module systems
find_package(Eigen3 3.3 QUIET NO_MODULE)

if(NOT Eigen3_FOUND)
    # For module systems - look in CMAKE_PREFIX_PATH or common locations
    find_path(EIGEN3_INCLUDE_DIR 
        NAMES Eigen/Dense
        HINTS ${CMAKE_PREFIX_PATH}
              /u/sw/toolchains/eigen/3.4.0/include
              /usr/include/eigen3
              /usr/local/include/eigen3
        NO_DEFAULT_PATH
    )
    
    if(EIGEN3_INCLUDE_DIR)
        message(STATUS "Found Eigen3 at: ${EIGEN3_INCLUDE_DIR}")
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        target_include_directories(Eigen3::Eigen INTERFACE ${EIGEN3_INCLUDE_DIR})
        set(Eigen3_FOUND TRUE)
    else()
        message(FATAL_ERROR "Eigen3 not found! Make sure to load the eigen module.")
    endif()
endif()

# Include directories
include_directories(include)

# Create the library (header-only interface library)
add_library(StochasticLA INTERFACE)

# Specify the include directories for the library
target_include_directories(StochasticLA INTERFACE
    ${CMAKE_SOURCE_DIR}/include
)

# Link with Eigen3
target_link_libraries(StochasticLA INTERFACE Eigen3::Eigen)

# Set C++ standard for the library
target_compile_features(StochasticLA INTERFACE cxx_std_17)

# Enable testing
enable_testing()

# Create test executable
add_executable(test_basic tests/test_basic.cpp)

# Link test with our library
target_link_libraries(test_basic StochasticLA)

# Add the test to CTest
add_test(NAME BasicTest COMMAND test_basic)

# Set test properties
set_tests_properties(BasicTest PROPERTIES
    TIMEOUT 30
    PASS_REGULAR_EXPRESSION "All tests PASSED"
)
