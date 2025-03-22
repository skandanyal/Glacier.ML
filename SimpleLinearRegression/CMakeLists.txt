cmake_minimum_required(VERSION 3.28)
project(Glacier_Dev)

set(CMAKE_CXX_STANDARD 17)

add_executable(SLR
        Simple_Linear_regression/SimpleLinearRegressionImplementation.cpp
        Simple_Linear_regression/SimpleLinearRegressionFlow.h
        Simple_Linear_regression/SimpleLinearRegressionExample.cpp)
