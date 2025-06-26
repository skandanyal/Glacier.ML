cmake_minimum_required(VERSION 3.28)
project(Glacier_Dev)

set(CMAKE_CXX_STANDARD 17)
set(EIGEN_INCLUDE_DIR "D:/EigenLibrary/eigen-3.4.0")

include_directories(${EIGEN_INCLUDE_DIR})

add_executable(SLR
        Simple_Linear_regression/SimpleLinearRegressionImplementation.cpp
        Simple_Linear_regression/SimpleLinearRegressionFlow.h
        Simple_Linear_regression/SimpleLinearRegressionExample.cpp)

add_executable(
        MLR
        Multiple_Linear_Regression/MultipleLinearRegression.h
        Multiple_Linear_Regression/MultipleLinearRegressionImplementation.cpp
        Multiple_Linear_Regression/MultipleLinearRegressionExample.cpp
)

add_executable(
        test
        testing.cpp
)
