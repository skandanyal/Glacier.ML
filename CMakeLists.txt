cmake_minimum_required(VERSION 3.28.3)
project(Glacier_ML)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

set(EIGEN_INCLUDE_DIR /home/skandan-c-y/cpp_libraries/eigen)                                      # eg: "D:/EigenLibrary/eigen-3.4.0"
set(BOOST_INCLUDE_DIR /home/skandan-c-y/cpp_libraries/boost_1_88_0)

# path until folder which contains bin
include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${BOOST_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/Utils)

# Enable high optimization for SIMD and fast math
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -fopt-info-vec-optimized")

# Find OpenMP, BLAS
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

add_executable(MLR
        MultipleLinearRegression/MultipleLinearRegressionFlow.h
        MultipleLinearRegression/MultipleLinearRegressionExample.cpp
        MultipleLinearRegression/MultipleLinearRegressionImplementation.cpp
        Utils/utilities.h
        Utils/utilities.cpp
)
target_compile_definitions(MLR PRIVATE DEBUG_MODE=1)

add_executable(SLR
        SimpleLinearRegression/SimpleLinearRegressionFlow.h
        SimpleLinearRegression/SimpleLinearRegressionExample.cpp
        SimpleLinearRegression/SimpleLinearRegressionImplementation.cpp
        Utils/utilities.h
        Utils/utilities.cpp
)
target_compile_definitions(SLR PRIVATE DEBUG_MODE=1)

add_executable(LogR
        LogisticRegression/LogisticRegressionExample.cpp
        LogisticRegression/LogisticRegressionFlow.h
        LogisticRegression/LogisticRegressionImplementation.cpp
        Utils/utilities.h
        Utils/utilities.cpp
)
target_compile_definitions(LogR PRIVATE DEBUG_MODE=1)

add_executable(KNNR
        KNNRegressor/KNNRegressorExample.cpp
        KNNRegressor/KNNRegressorFlow.h
        KNNRegressor/KNNRegressorImplementation.cpp
        Utils/utilities.h
        Utils/utilities.cpp
)
target_compile_definitions(KNNR PRIVATE DEBUG_MODE=1)

add_executable(KNNC
        Utils/utilities.h
        Utils/utilities.cpp
)
target_compile_definitions(KNNC PRIVATE DEBUG_MODE=1)

# Link OpenMP for openMP target
if(OpenMP_CXX_FOUND)
    target_link_libraries(MLR PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(SLR PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(LogR PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(KNNR PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(KNNC PUBLIC OpenMP::OpenMP_CXX)
endif()

# Link BLAS to the targets
if(BLAS_FOUND)
    target_link_libraries(MLR PUBLIC BLAS::BLAS)
    target_link_libraries(SLR PUBLIC BLAS::BLAS)
    target_link_libraries(LogR PUBLIC BLAS::BLAS)
    target_link_libraries(KNNR PUBLIC BLAS::BLAS)
    target_link_libraries(KNNC PUBLIC BLAS::BLAS)
endif()

# https://github.com/ben-strasser/fast-cpp-csv-parser?
# use this some day in the future to integrate a fast csv parser
# BLAS will be implemented someday in the future :)
