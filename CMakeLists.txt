cmake_minimum_required(VERSION 3.12)
project(dbscan)

add_definitions("-std=c++11")
add_compile_definitions("HMLP_USE_MPI")

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find MPI
find_package(MPI REQUIRED)

# For supporting CMake < 3.9:

if(NOT TARGET MPI::MPI_CXX)
    add_library(MPI::MPI_CXX IMPORTED INTERFACE)

    set_property(TARGET MPI::MPI_CXX
            PROPERTY INTERFACE_COMPILE_OPTIONS ${MPI_CXX_COMPILE_FLAGS})
    set_property(TARGET MPI::MPI_CXX
            PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_PATH}")
    set_property(TARGET MPI::MPI_CXX
            PROPERTY INTERFACE_LINK_LIBRARIES ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})
endif()

message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")

if (NOT APPLE)

    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup()
    #add_executable(dbscan src/dbscan.cpp)
    #target_include_directories(dbscan PRIVATE "${PROJECT_SOURCE_DIR}/include")

    #target_link_libraries(dbscan ${CONAN_LIBS})

    #target_compile_options(dbscan PRIVATE "$<$<CONFIG:Release>:-O3>")

    #target_compile_options(dbscan PRIVATE -Wall -Werror -Wextra -pedantic)

endif()

add_subdirectory(test)