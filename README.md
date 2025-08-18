# threaded-avx-simd<br>
Simple AVX2 SIMD based parallel threaded library with optimized:<br>
reduce sum and sort for float<br>
<br>
To build:<br>
mkdir -p bin && cd bin/<br>
cmake ..<br>
make<br>
<br>
To run:<br>
./test/avx_kernels_reduce_sum_test <array-size> <br>
./test/avx_kernels_sort_test <size> <dir[0/1]> <br>
./test/avx_kernels_matmul_test <sizeM> <sizeN> <sizeP> <br>
