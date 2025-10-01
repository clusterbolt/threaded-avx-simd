cmake_minimum_required(VERSION 3.28)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-O3 -fPIC -mavx2 -mfma)

# Source files
set(SOURCE_FILES
    sort_op/bitonic_sort_with_simd.cpp
    reduce_op/reduce_sum_with_threads_and_simd.cpp
    matmul_op/matmul_with_simd.cpp
)

add_library(avxkernels STATIC ${SOURCE_FILES})

target_include_directories(avxkernels PUBLIC ${PROJECT_SOURCE_DIR}/common)

target_include_directories(avxkernels PRIVATE
                            ${CMAKE_CURRENT_SOURCE_DIR}/sort_op/include
                            ${CMAKE_CURRENT_SOURCE_DIR}/reduce_op/include
                            ${CMAKE_CURRENT_SOURCE_DIR}/matmul_op/include)

set_target_properties(avxkernels PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
