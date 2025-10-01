#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <immintrin.h>
#include <cassert>
#include <cmath>
#include <mutex>
#include <numeric> // For std::accumulate
#include "api.h"

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
        std::cerr << "Usage: " << argv[0] << " <arrSize>" << std::endl;
        return 1;
    }

    size_t arrSize = 0;
    arrSize = std::stoul(argv[1]);
    float* dataArray = new float[arrSize];

    for (size_t i = 0; i < arrSize; ++i) {
        dataArray[i] = rand() % 1000 / 100.0f; // Random float between 0.0 and 9.99
    }

    int nThreads = std::thread::hardware_concurrency();

    float sumRes;
    high_resolution_clock::time_point start = high_resolution_clock::now();
    reduce_sum_execute(reduceSumType_t::REDUCE_SUM, dataArray, arrSize, sumRes);
    std::cout << "Optimized time taken: " <<
        std::chrono::duration_cast<milliseconds>(high_resolution_clock::now() - start).count() <<
        " ms" << std::endl;
    std::cout << "Sum: " << sumRes << std::endl;
    start = high_resolution_clock::now();
    float sumRef = RefSum(dataArray, arrSize, nThreads);
    std::cout << "Reference time taken: " <<
        std::chrono::duration_cast<milliseconds>(high_resolution_clock::now() - start).count() <<
        " ms" << std::endl;
    std::cout << "Ref sum: " << sumRef << std::endl;
    // Find the difference when interpreted as integers
    int ulpDiff = std::fabs(*reinterpret_cast<int*>(&sumRes) - *reinterpret_cast<int*>(&sumRef));
    std::cout<< "ULP difference: " << ulpDiff << std::endl;
    // Standard library accumulate for comparison
    std::cout << "Std accumulate sum: " <<
        static_cast<float>(std::accumulate(dataArray, dataArray + arrSize, 0.0f))<< std::endl;

    delete[] dataArray;
}
