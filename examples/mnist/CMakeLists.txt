cmake_minimum_required(VERSION 3.14)
project(mnist_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find mygrad from the source tree
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../ mygrad_build EXCLUDE_FROM_ALL)

add_executable(mnist main.cpp)
target_link_libraries(mnist PRIVATE mygrad)