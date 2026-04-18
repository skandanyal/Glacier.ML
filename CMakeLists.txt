cmake_minimum_required(VERSION 3.20)
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

project(GlacierML LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------------------------------
# Dependencies
# --------------------------------------------------
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)

# --------------------------------------------------
# Library: glacier_logr
# --------------------------------------------------
# SOURCE LINKING: Implementation files are added here.
add_library(glacier_logr STATIC
        src/Models/LogR/LogR_impl.cpp
        src/Models/LogR/core/LogRCore_impl.cpp
)

# HEADER LINKING: Search paths for .hpp files.
# 'include' allows: #include "Glacier/Models/LogisticRegression.hpp"
# 'src/Models/LogR/core' allows: #include "LogRCore.hpp"
target_include_directories(glacier_logr PUBLIC
        include
        src
)

target_link_libraries(glacier_logr PUBLIC
        Eigen3::Eigen
        Boost::boost
        BLAS::BLAS
        OpenMP::OpenMP_CXX
)

# --------------------------------------------------
# GoogleTest Infrastructure
# --------------------------------------------------
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)
enable_testing()

# --------------------------------------------------
# Test Bench
# --------------------------------------------------
add_executable(test_logr tests/LogR/test_LogR.cpp)
target_link_libraries(test_logr PRIVATE glacier_logr GTest::gtest_main)
gtest_discover_tests(test_logr)

