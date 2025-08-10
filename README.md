# threaded-avx-simd
Simple AVX2 SIMD based parallel threaded library with optimized:
reduce sum and sort for float

To build:
mkdir -p bin && cd bin/
cmake ..
make

To run:
./test/avx_kernels_reduce_sum_test <array-size>
./test/avx_kernels_sort_test <size> <dir[0/1]> <OPTIONAL: alt = unidirectional>
