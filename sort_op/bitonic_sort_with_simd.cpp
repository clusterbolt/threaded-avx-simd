/*
    Ascending Bitonic Sort using AVX2 SIMD (8 element vectors) and manual loop unrolling.
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <immintrin.h>
#include <chrono>

// #define TEST  // Uncomment to enable test mode
constexpr int FLOAT_VEC_SIZE = 8;
constexpr int SORT_SIZE = 8;
constexpr int OUTER_SIZE = 2000;

void HandleStage(__m256& vec0, const int offsetsArray[FLOAT_VEC_SIZE], const int signArray[FLOAT_VEC_SIZE]) {
    __m256i offsets = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(offsetsArray));
    __m256i sign = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(signArray));

    __m256 vec1 = _mm256_permutevar8x32_ps(vec0, offsets);

    __m256 mask = _mm256_cmp_ps(vec0, vec1, _CMP_GT_OQ); // Compare vec0 and vec1
    __m256i maskInt = _mm256_castps_si256(mask); // Cast comparison result to integer mask
    maskInt = maskInt ^ sign; // Apply sign mask

    mask = _mm256_castsi256_ps(maskInt); // Cast back to float for blending
    vec0 = _mm256_blendv_ps(vec0, vec1, mask);
}

#ifdef TEST
    void PrintData(const std::vector<std::vector<float>>& data, const std::string& title) {
        std::cout << title;
        for (const auto& row : data) {
            for (const auto& val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }
#else
    void PrintData(...) {
        // No-op in non-test mode
    }
#endif

int main() {
    std::vector<std::vector<float>> in0(OUTER_SIZE, std::vector<float>(SORT_SIZE)), 
        out0(OUTER_SIZE, std::vector<float>(SORT_SIZE));
    auto& out0Ref = in0;
#ifndef TEST
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);
    for (int k = 0; k < OUTER_SIZE; ++k) {
        for (int i = 0; i < SORT_SIZE; ++i) {
            in0[k][i] = dis(gen);
        }
    }
#else
    for (int k = 0; k < OUTER_SIZE; ++k) {
        for (int i = 0; i < SORT_SIZE; ++i) {
            in0[k][i] = (float) i * 10.0f; // Example data for testing
        }
    }
#endif

    PrintData(in0, "Input: \n");

    auto refTimeTotal = std::chrono::high_resolution_clock::duration::zero();
    for (int k = 0; k < OUTER_SIZE; ++k) {
        std::copy(in0[k].begin(), in0[k].end(), out0[k].begin()); // in to out
        auto refTimeStart = std::chrono::high_resolution_clock::now();
        std::sort(out0Ref[k].begin(), out0Ref[k].end()); // sort in place
        refTimeTotal += (std::chrono::high_resolution_clock::now() - refTimeStart);
    }
    std::cout << "Reference time: " << 
        std::chrono::duration_cast<std::chrono::microseconds>(refTimeTotal).count() << " us" << std::endl;

    auto refTimeStart = std::chrono::high_resolution_clock::now();

    const int c_m = std::log2(FLOAT_VEC_SIZE); // Number of sort stages
    int offsetsArray[3][FLOAT_VEC_SIZE] = {{1, 0, 3, 2, 5, 4, 7, 6},
                                            {2, 3, 0, 1, 6, 7, 4, 5},
                                            {4, 5, 6, 7, 0, 1, 2, 3}};
    int signArray[c_m * (c_m + 1) / 2][FLOAT_VEC_SIZE] = {{0, -1, -1, 0, 0, -1, -1, 0}, // 1.1
                                                            {0, 0, -1, -1, -1, -1, 0, 0}, // 2.1
                                                            {0, -1, 0, -1, -1, 0, -1, 0}, // 2.2
                                                            {0, 0, 0, 0, -1, -1, -1, -1}, // 3.1
                                                            {0, 0, -1, -1, 0, 0, -1, -1}, // 3.2
                                                            {0, -1, 0, -1, 0, -1, 0, -1}}; // 3.3
    
    for (int k = 0; k < OUTER_SIZE; k += 2) {
       __m256 vec0 = _mm256_loadu_ps(out0[k].data()); // Load 8 floats into a vector
       __m256 vec1 = _mm256_loadu_ps(out0[k + 1].data()); // Load 8 floats into a vector
        for (int i = 0; i < c_m; ++i) {
            // std::cout << "Stage " << i + 1 << ": \n";
            int stageOffset = i * (i + 1) / 2; // Offset for signArray
            for (int j = 0; j <= i; j++)
            {
                // std::cout << "Sub Stage " << j + 1 << ": \n";
                HandleStage(vec0, offsetsArray[i - j], signArray[stageOffset + j]);
                HandleStage(vec1, offsetsArray[i - j], signArray[stageOffset + j]);
            }
        }

        _mm256_storeu_ps(out0[k].data(), vec0); // Store the permuted vector back to out0
        _mm256_storeu_ps(out0[k + 1].data(), vec1); // Store the permuted vector back to out0
    }
    std::cout << "SIMD time: " << 
        std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::high_resolution_clock::now() - refTimeStart).count() << " us" << std::endl;
    PrintData(out0, "Output: \n");
}