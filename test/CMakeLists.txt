add_executable(avx_kernels_sort_test test_sort_op/bitonic_sort_test.cpp)
add_executable(avx_kernels_reduce_sum_test test_reduce_op/reduce_sum_test.cpp)

target_include_directories(avx_kernels_sort_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src/sort_op/include)
target_include_directories(avx_kernels_reduce_sum_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src/reduce_op/include)

# Link against your library target by name
target_link_libraries(avx_kernels_sort_test PRIVATE avxkernels)
target_link_libraries(avx_kernels_reduce_sum_test PRIVATE avxkernels)
