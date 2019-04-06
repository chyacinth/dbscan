cmake_minimum_required(VERSION 2.8.12)
project(dbscan)

add_definitions("-std=c++11")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(dbscan src/dbscan.cpp)
target_link_libraries(dbscan ${CONAN_LIBS})
target_include_directories(dbscan PRIVATE "${PROJECT_SOURCE_DIR}/include")