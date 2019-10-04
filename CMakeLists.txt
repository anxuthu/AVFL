cmake_minimum_required(VERSION 2.8.9)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wno-unused-variable -std=c++11 -DCMAKE_BUILD_TYPE=Release -fopenmp")

project(AFD)
set(dir ${PROJECT_SOURCE_DIR})
message("project source dir: " ${dir})
set(EXECUTABLE_OUTPUT_PATH ${dir}/bin)
message("project binary dir: " ${EXECUTABLE_OUTPUT_PATH})

#set(CMAKE_CXX_COMPILER "mpicxx")
set(CMAKE_CXX_COMPILER "/usr/local/openmpi-4.0.1/bin/mpicxx")
include(CheckCXXCompilerFlag)

file(GLOB headers "${dir}/src/*.h")
file(GLOB objs "${dir}/src/*.cpp")

#include_directories("/ihome/hhuang/anx6/usr/include")
#add_library(armadillo SHARED IMPORTED)
#set_property(TARGET armadillo
#   PROPERTY IMPORTED_LOCATION "/ihome/hhuang/anx6/usr/lib64/libarmadillo.so")

add_executable(afd "afd.cpp" ${objs} ${headers})
target_link_libraries(afd gflags armadillo)
