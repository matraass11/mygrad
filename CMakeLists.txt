cmake_minimum_required(VERSION 3.14)
project(mygrad LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Create the static library
add_library(mygrad STATIC
    src/helper.cpp
    src/layers.cpp
    src/loss.cpp
    src/model.cpp
    src/optim.cpp
    src/tensor.cpp
)

# Set public include path for consumers
target_include_directories(mygrad PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Optional: If you want to define an install rule later
# install(TARGETS mygrad EXPORT mygradTargets DESTINATION lib)
# install(DIRECTORY include/mygrad DESTINATION include)
