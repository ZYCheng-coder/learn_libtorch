cmake_minimum_required(VERSION 3.20)
project(libtorch_ws)

set(CMAKE_CXX_STANDARD 11)

set(Torch_DIR /home/czy/tools/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#include_directories()

add_executable(libtorch_ws main.cpp)
target_link_libraries(libtorch_ws  "${TORCH_LIBRARIES}")
set_property(TARGET libtorch_ws PROPERTY CXX_STANDARD 14)