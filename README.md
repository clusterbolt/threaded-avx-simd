# threaded-avx-sum
Simple AVX2 SIMD based parallel threaded reduce sum

To build:
g++ reduce_op_with_threads_and_simd.cpp -mavx2

To run:
./a.out <array-size>
