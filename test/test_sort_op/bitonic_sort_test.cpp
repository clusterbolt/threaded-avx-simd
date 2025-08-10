/*
    Bitonic Sort test.
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <limits>
#include "bitonic_sort_with_simd.h"

// #define PRINT_DEBUG  // Uncomment to enable debug prints
#define RANDOM_DATA  // Uncomment to use random data instead of a fixed pattern
constexpr unsigned int FLOAT_VEC_SIZE = 8;
constexpr unsigned int OUTER_SIZE = 10;

#ifdef PRINT_DEBUG
    #define PRINT_DATA(DATA, TITLE)                                      \
            std::cout << TITLE;                                          \
            for (const auto& row : DATA) {                               \
                for (const auto& val : row) {                            \
                    std::cout << val << " ";                             \
                }                                                        \
                std::cout << std::endl;                                  \
            }
#else
    #define PRINT_DATA(DATA, TITLE) (void)0
#endif

bool my_less(float a, float b) { return a < b; }
bool my_greater(float a, float b) { return a > b; }

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <size> <dir[0/1]> <alt = unidirectional>" << std::endl;
        return 1;
    }
    unsigned int sortSize = std::stoul(argv[1]);
    bool sortDir = std::stoi(argv[2]) != 0; // true for ascending, false for descending
    std::vector<std::vector<float>> in0(OUTER_SIZE, std::vector<float>(sortSize)),
        out0(OUTER_SIZE, std::vector<float>(sortSize));
    std::string sortType = (argc > 3) ? argv[3] : "";
    auto& out0Ref = in0;
#ifdef RANDOM_DATA
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);
    for (int k = 0; k < OUTER_SIZE; ++k) {
        for (int i = 0; i < sortSize; ++i) {
            in0[k][i] = dis(gen);
        }
    }
#else
    for (int k = 0; k < OUTER_SIZE; ++k) {
        for (int i = 0; i < sortSize; ++i) {
            in0[k][i] = (float) (sortSize - (i % sortSize) - 1) * 10.0f; // Example data for testing
        }
    }
#endif

    PRINT_DATA(in0, "Input: \n");

    auto comp = sortDir ? my_less : my_greater;
    auto refTimeTotal = std::chrono::high_resolution_clock::duration::zero();
    for (int k = 0; k < OUTER_SIZE; ++k) {
        std::copy(in0[k].begin(), in0[k].end(), out0[k].begin()); // in to out
        auto refTimeStart = std::chrono::high_resolution_clock::now();
        std::sort(out0Ref[k].begin(), out0Ref[k].end(), comp); // sort in place
        refTimeTotal += (std::chrono::high_resolution_clock::now() - refTimeStart);
    }
    std::cout << "Reference time: " <<
        std::chrono::duration_cast<std::chrono::microseconds>(refTimeTotal).count() << " us" << std::endl;

    auto kernelTimeTotal = std::chrono::high_resolution_clock::duration::zero();
    if (sortType == "alt")
    {
        auto kernelTimeStart = std::chrono::high_resolution_clock::now();
        BitonicSortAltFunc(out0, OUTER_SIZE, sortSize, sortDir);
        kernelTimeTotal += (std::chrono::high_resolution_clock::now() - kernelTimeStart);
    }
    else
    {
        auto kernelTimeStart = std::chrono::high_resolution_clock::now();
        BitonicSortFunc(out0, OUTER_SIZE, sortSize, sortDir);
        kernelTimeTotal += (std::chrono::high_resolution_clock::now() - kernelTimeStart);
    }
    std::cout << "SIMD time: " <<
        std::chrono::duration_cast<std::chrono::microseconds>(kernelTimeTotal).count() << " us" << std::endl;

    PRINT_DATA(out0, "Output: \n");

    if (out0 != in0)
    {
        std::cout << "Sort mismatch" << std::endl;
    }
}