find_package(OpenMP REQUIRED)

find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")

# testSingleLinkageTree

add_executable(testSingleLinkageTree test.cpp)

target_link_libraries(testSingleLinkageTree PUBLIC OpenMP::OpenMP_CXX)

target_link_libraries(testSingleLinkageTree PUBLIC MPI::MPI_CXX)

target_include_directories(testSingleLinkageTree PRIVATE "${PROJECT_SOURCE_DIR}/include")

target_compile_options(testSingleLinkageTree PRIVATE "$<$<CONFIG:Release>:-O3>")

target_compile_options(testSingleLinkageTree PRIVATE -Wall -Werror -Wextra -pedantic)

# testSingleLinkageTree