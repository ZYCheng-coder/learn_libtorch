cmake_minimum_required(VERSION 3.20)
project(ws_2)

set(CMAKE_CXX_STANDARD 11)

set(Torch_DIR /home/czy/tools/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status: ")
message(STATUS "> version: ${OpenCV_VERSION}")
message(STATUS "> libraries: ${OpenCV_LIBS}")
message(STATUS "> include: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

FILE(GLOB SRC_LIST_CPP "${PROJECT_SOURCE_DIR}/src/*.cpp")
FILE(GLOB INCLUDE_LIST_H "${PROJECT_SOURCE_DIR}/include/*.h")

add_executable(ws_2 main.cpp ${SRC_LIST_CPP} ${INCLUDE_LIST_H} )
target_link_libraries(ws_2 "${TORCH_LIBRARIES}" ${OpenCV_LIBS})
set_property(TARGET ws_2 PROPERTY CXX_STANDARD 14)