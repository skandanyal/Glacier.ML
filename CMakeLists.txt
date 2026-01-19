cmake_minimum_required(VERSION 3.20)

cmake_policy(SET CMP0167 NEW)

project(Glacier_ML LANGUAGES CXX)

# ----------------------------
# Language standard
# ----------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ----------------------------
# Dependencies
# ----------------------------
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)

# ----------------------------
# Glacier compiled library
# ----------------------------
file(GLOB_RECURSE GLACIER_SOURCES
        src/*.cpp
)

add_library(Glacier STATIC ${GLACIER_SOURCES})

target_include_directories(Glacier
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(Glacier
        PUBLIC
        Eigen3::Eigen
        Boost::boost
        BLAS::BLAS
        OpenMP::OpenMP_CXX
)

# ----------------------------
# Compile options per build type
# ----------------------------
target_compile_options(Glacier
        PRIVATE
        $<$<CONFIG:Debug>:-g -O0>
        $<$<CONFIG:Release>:-O3 -march=native -ffast-math>
        $<$<CONFIG:Benchmark>:-O3 -march=native -g -ffast-math>
)

# ----------------------------
# Compile definitions per build type
# ----------------------------
target_compile_definitions(Glacier
        PUBLIC
        $<$<CONFIG:Debug>:GLACIER_DEBUG>
        $<$<CONFIG:Release>:GLACIER_RELEASE>
        $<$<CONFIG:Benchmark>:GLACIER_BENCHMARK>
)

# ----------------------------
# Executables (benchmark / drivers)
# ----------------------------
set(BENCHMARK_DRIVERS
        Benchmark_drivers/Classification_Models/KNNClassifierDriver.cpp
        Benchmark_drivers/Regression_Models/KNNRegressorDriver.cpp
        Benchmark_drivers/Classification_Models/LogisticRegressionDriver.cpp
        Benchmark_drivers/Regression_Models/MultipleLinearRegressionDriver.cpp
        Benchmark_drivers/Regression_Models/SimpleLinearRegressionDriver.cpp
        Benchmark_drivers/Classification_Models/SVMClassifierDriver.cpp
        Benchmark_drivers/Regression_Models/SVMRegressorDriver.cpp
)

# Auto-create executables from list
foreach(DRIVER ${BENCHMARK_DRIVERS})
    get_filename_component(DRIVER_NAME ${DRIVER} NAME_WE)
    add_executable(${DRIVER_NAME} ${DRIVER})
    target_link_libraries(${DRIVER_NAME} PRIVATE Glacier)
endforeach()
