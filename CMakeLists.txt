cmake_minimum_required(VERSION 3.14)
project(mygrad LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# fetch-assets and friends, reachable from the example build trees too
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/assets.cmake)

add_library(mygrad STATIC
    src/conv2d.cpp
    src/helper.cpp
    src/layers.cpp
    src/linearLayer.cpp
    src/loss.cpp
    src/model.cpp
    src/optim.cpp
    src/tensor.cpp
    src/threadPool.cpp
)

target_include_directories(mygrad PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Tests are opt-in. Catch2 is a test-only dependency and is not fetched unless
# this is on, so consuming mygrad through add_subdirectory (as both examples do)
# stays free of it and libmygrad.a keeps linking against nothing extra.
option(MYGRAD_BUILD_TESTS "Build the mygrad test suite (fetches Catch2)" OFF)

if(MYGRAD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
