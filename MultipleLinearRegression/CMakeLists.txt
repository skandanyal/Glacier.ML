cmake_minimum_required(VERSION 3.28)
project(__name_of_your_project__)

set(CMAKE_CXX_STANDARD 17)
set(EIGEN_INCLUDE_DIR __path_to_Eigen__)                                         # eg: "D:/EigenLibrary/eigen-3.4.0"
                                                                                 # path until folder which contains bin
include_directories(${EIGEN_INCLUDE_DIR})

add_executable(
        MLR
        Multiple_Linear_Regression/MultipleLinearRegression.h
        Multiple_Linear_Regression/MultipleLinearRegressionImplementation.cpp
)
