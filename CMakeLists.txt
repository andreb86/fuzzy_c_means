cmake_minimum_required(VERSION 3.12)
project(fuzzy_c_means)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_COMPILER mpic++)
set(CMAKE_CXX_FLAGS "-Wall -mtune=native -fopenmp -O3")
add_executable(cmeans main.cpp)