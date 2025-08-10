#include <thread>
#include <vector>
#include <immintrin.h>
#include <cassert>
#include <cmath>
#include <mutex>
#include <numeric> // For std::accumulate
#include "reduce_sum_with_threads_and_simd.h"

#ifdef DEBUG
            std::mutex print_mtx; // Mutex for printing
#endif

constexpr int FLOAT_VEC_SIZE = 8;

void ReduceSumFunc(float *dataArray, unsigned int arrSize, float& sum)
{
    std::vector<std::thread> threads;
    int nThreads = std::thread::hardware_concurrency();
    int nBlocks = (arrSize + nThreads - 1) / nThreads;

    std::vector<float> lSum(nThreads, 0.0f);
    for (int i = 0; i < nThreads; ++i) {
        threads.emplace_back([=, &lSum, &dataArray]() {
            int j = i * nBlocks;
            int limit = std::min((i + 1) * nBlocks, (int)arrSize);
#ifdef DEBUG
            {
                std::lock_guard<std::mutex> lock(print_mtx);
                std::cout << "Thread " << i << " processing from " << j << " to " << limit << std::endl;
                std::cout << "Full chunks " << (limit - j) / FLOAT_VEC_SIZE << std::endl;
                std::cout << "Remaining elements " << (limit - j) % FLOAT_VEC_SIZE << std::endl;
            }
#endif
            __m256 vsum = _mm256_setzero_ps();

            // Process chunks of 8 floats using AVX2
            for (; j + FLOAT_VEC_SIZE - 1 < limit; j += FLOAT_VEC_SIZE) {
                __m256 v = _mm256_loadu_ps(&dataArray[j]);
                vsum = _mm256_add_ps(vsum, v);
            }

            // Horizontal add within vsum vector
            __m128 low = _mm256_castps256_ps128(vsum);             // lower 128 bits
            __m128 high = _mm256_extractf128_ps(vsum, 1);          // upper 128 bits
            __m128 sum128 = _mm_add_ps(low, high);                 // sum low and high parts

            // Now horizontally sum 4 floats in sum128
            sum128 = _mm_hadd_ps(sum128, sum128); // [a1 + a2] [a3 + a4] ....
            sum128 = _mm_hadd_ps(sum128, sum128); // [a1 + a2 + a3 + a4] [a1 + a2 + a3 + a4] ....

            float simdSum = _mm_cvtss_f32(sum128);

            // add remaining elements
            for (; j < limit; ++j) {
                simdSum += dataArray[j];
            }

            lSum[i] = simdSum;
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    sum = 0.0f;
    for (size_t i = 0; i < nThreads; ++i) {
        sum += lSum[i];
    }
}
