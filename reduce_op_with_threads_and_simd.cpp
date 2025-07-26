#include<iostream>
#include<chrono>
#include<thread>
#include<vector>
#include<immintrin.h>
#include<cassert>
#include<cmath>

float RefSum(const float* dataArray, size_t size) {
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        sum += dataArray[i];
    }
    return sum;
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
    int offset = (size + nThreads - 1) / nThreads;
    float* dataArray = new float[size];

    std::vector<std::thread> threads;
    for (size_t i = 0; i < size; ++i) {
        dataArray[i] = rand() % 1000 / 100.0f; // Random float between 0.0 and 9.99
    }
    float *lSum = new float[nThreads];

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nThreads; ++i) {
        threads.emplace_back([=, &lSum, &dataArray]() {
            float localSum = 0.0f;

            int j = i * offset;
            int limit = std::min((i + 1) * offset, (int)size);
            __m256 vsum = _mm256_setzero_ps();

            // Process chunks of 8 floats using AVX2
            for (; j + 7 < limit; j += 8) {
                __m256 v = _mm256_loadu_ps(&dataArray[j]);
                vsum = _mm256_add_ps(vsum, v);
            }

            // Horizontal add within vsum vector
            // AVX2 has no single instruction to horizontally sum all 8 floats, so do in steps:
            __m128 low = _mm256_castps256_ps128(vsum);             // lower 128 bits
            __m128 high = _mm256_extractf128_ps(vsum, 1);          // upper 128 bits
            __m128 sum128 = _mm_add_ps(low, high);                  // sum low and high parts

            // Now horizontally sum 4 floats in sum128
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);

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
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;
    std::cout << "Sum: " << sum << std::endl;

    std::cout<<RefSum(dataArray, size) << std::endl;
    // assert(std::fabs(RefSum(dataArray, size) - sum) < 2e-3);

    delete[] dataArray;
    delete[] lSum;
}