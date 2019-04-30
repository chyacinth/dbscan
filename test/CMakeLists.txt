add_definitions("-std=c++11")

# testSingleLinkageTree
#add_executable(testSingleLinkageTree test.cpp)
#target_include_directories(testSingleLinkageTree PRIVATE "${PROJECT_SOURCE_DIR}/include")
#target_link_libraries(testSingleLinkageTree PUBLIC OpenMP::OpenMP_CXX)
#target_link_libraries(testSingleLinkageTree PUBLIC MPI::MPI_CXX)
#target_compile_options(testSingleLinkageTree PRIVATE "$<$<CONFIG:Release>:-O3>")
#target_compile_options(testSingleLinkageTree PRIVATE -Wall -Werror -Wextra -pedantic)

# testBoruvka
#add_executable(testBoruvka test_boruvka.cpp)
#target_include_directories(testBoruvka PRIVATE "${PROJECT_SOURCE_DIR}/include")
#target_link_libraries(testBoruvka PUBLIC OpenMP::OpenMP_CXX PUBLIC atomic)
#target_link_options(testBoruvka PUBLIC -latomic)
#target_compile_options(testBoruvka PRIVATE "$<$<CONFIG:Release>:-O3>")


# testDistributedBoruvka
if (NOT APPLE)

    #include(${CMAKE_BINARY_DIR}/../conanbuildinfo.cmake)
    #conan_basic_setup()
    add_executable(testDistributedBoruvka test_distributed_boruvka.cpp)
    target_link_libraries(testDistributedBoruvka PUBLIC MPI::MPI_CXX)
    target_link_libraries(testDistributedBoruvka PUBLIC ${CONAN_LIBS})
    target_link_libraries(testDistributedBoruvka PUBLIC MPI::MPI_CXX PUBLIC atomic)
    target_include_directories(testDistributedBoruvka PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_compile_options(testDistributedBoruvka PRIVATE "$<$<CONFIG:Release>:-O3>")
    #target_compile_options(testDistributedBoruvka PRIVATE -Wall -Werror -Wextra -pedantic)
    #target_link_options(testDistributedBoruvka )

endif()