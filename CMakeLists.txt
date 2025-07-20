cmake_minimum_required(VERSION 3.10)

set (CMAKE_CXX_STANDARD 20)

project("net")

file(GLOB MAIN_SOURCES "*.cpp" "src/*.cpp")

list(REMOVE_ITEM MAIN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test.cpp")

add_executable(net ${MAIN_SOURCES})