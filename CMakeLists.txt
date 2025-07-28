cmake_minimum_required(VERSION 3.14)
project(mygrad LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(mygrad STATIC
    src/helper.cpp
    src/layers.cpp
    src/loss.cpp
    src/model.cpp
    src/optim.cpp
    src/tensor.cpp
)

target_include_directories(mygrad PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
