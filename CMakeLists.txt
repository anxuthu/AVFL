cmake_minimum_required(VERSION 2.8.9)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wno-unused-variable -std=c++11 -DCMAKE_BUILD_TYPE=Release -fopenmp")

project(AFD)
set(dir ${PROJECT_SOURCE_DIR})
message("project source dir: " ${dir})
set(EXECUTABLE_OUTPUT_PATH ${dir}/bin)
message("project binary dir: " ${EXECUTABLE_OUTPUT_PATH})

set(CMAKE_CXX_COMPILER "/home/an/third_party/bin/mpicxx")
include(CheckCXXCompilerFlag)

file(GLOB headers "${dir}/src/*.h")
file(GLOB objs "${dir}/src/*.cpp")

include_directories("/home/an/third_party/bin/include")
link_directories("/home/an/third_party/lib64")

add_executable(afd "afd.cpp" ${objs} ${headers})
add_executable(afdmc "afd_mc.cpp" ${objs} ${headers})
add_executable(fdmlmc "fdml_mc.cpp" ${objs} ${headers})
target_link_libraries(afd gflags armadillo)
target_link_libraries(afdmc gflags armadillo)
target_link_libraries(fdmlmc gflags armadillo)
