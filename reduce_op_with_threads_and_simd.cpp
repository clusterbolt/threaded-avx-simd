#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <immintrin.h>
#include <cassert>
#include <cmath>
#include <mutex>
#include <numeric> // For std::accumulate

// #define DEBUG  // Uncomment to enable debug prints
#ifdef DEBUG
            std::mutex print_mtx; // Mutex for printing
#endif

constexpr int FLOAT_VEC_SIZE = 8;

using std::chrono::high_resolution_clock, std::chrono::milliseconds;

float RefSum(const float* dataArray, size_t size, int nThreads) {
    int nBlocks = (size + nThreads - 1) / nThreads;
    float sumThread{0.0f};
    int i = 0;
    for (; i < nThreads; ++i) { // Simulate thread processing
        int j = i * nBlocks;
        int limit = std::min((i + 1) * nBlocks, (int)size);
        float sumPerElement[FLOAT_VEC_SIZE] = {0.0f};
        for (; j + FLOAT_VEC_SIZE - 1 < limit; j += FLOAT_VEC_SIZE) {
            for (size_t k = 0; k < FLOAT_VEC_SIZE; ++k) {
                sumPerElement[k] += dataArray[j + k];
            }
        }
        sumPerElement[0] += sumPerElement[4];
        sumPerElement[1] += sumPerElement[5];
        sumPerElement[2] += sumPerElement[6];
        sumPerElement[3] += sumPerElement[7];

        sumPerElement[0] += sumPerElement[1];
        sumPerElement[2] += sumPerElement[3];

        sumPerElement[0] += sumPerElement[2];

        for (; j < limit; ++j) {
            sumPerElement[0] += dataArray[j];
        }
        sumThread += sumPerElement[0];
    }
    return sumThread;
}

int main(int argc, char* argv[])
{
    srand(static_cast<unsigned int>(time(0))); // Seed for random number generation
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <size>" << std::endl;
        return 1;
    }

    int nThreads = std::thread::hardware_concurrency();
    size_t size = 0;
    size = std::stoul(argv[1]);
    int nBlocks = (size + nThreads - 1) / nThreads;
    float* dataArray = new float[size];

    std::vector<std::thread> threads;
    for (size_t i = 0; i < size; ++i) {
        dataArray[i] = rand() % 1000 / 100.0f; // Random float between 0.0 and 9.99
    }

    std::vector<float> lSum(nThreads, 0.0f);
    high_resolution_clock::time_point start = high_resolution_clock::now();
    for (int i = 0; i < nThreads; ++i) {
        threads.emplace_back([=, &lSum, &dataArray]() {
            int j = i * nBlocks;
            int limit = std::min((i + 1) * nBlocks, (int)size);
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
    float sum = 0.0f;
    for (size_t i = 0; i < nThreads; ++i) {
        sum += lSum[i];
    }
    std::cout << "Time taken: " <<
        std::chrono::duration_cast<milliseconds>(high_resolution_clock::now() - start).count() <<
        " ms" << std::endl;
    std::cout << "Sum: " << sum << std::endl;

    float sumRef = RefSum(dataArray, size, nThreads);
    std::cout << "Ref sum: " << sumRef << std::endl;
    // Find the difference when interpreted as integers
    int ulpDiff = std::fabs(*reinterpret_cast<int*>(&sum) - *reinterpret_cast<int*>(&sumRef));
    std::cout<< "ULP difference: " << ulpDiff << std::endl;
    // Standard library accumulate for comparison
    std::cout << "Std accumulate sum: " <<
        static_cast<float>(std::accumulate(dataArray, dataArray + size, 0.0f))<< std::endl;

    delete[] dataArray;
}
