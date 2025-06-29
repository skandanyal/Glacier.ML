cmake_minimum_required(VERSION 3.28)
project(__name_of_your_project__)

set(CMAKE_CXX_STANDARD 20)
set(EIGEN_INCLUDE_DIR /home/skandan-c-y/cpp_libraries/eigen)
set(BOOST_INCLUDE_DIR /home/skandan-c-y/cpp_libraries/boost_1_88_0)

#path until folder which contains bin
include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${BOOST_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/Utils)

#add_definitions(-DDEBUG_MODE = 0)

add_executable(SLR
        Simple_Linear_regression/SimpleLinearRegressionImplementation.cpp
        Simple_Linear_regression/SimpleLinearRegressionFlow.h
        Simple_Linear_regression/SimpleLinearRegressionExample.cpp
        Utils/utilities.h
        Utils/utilities.cpp
)
