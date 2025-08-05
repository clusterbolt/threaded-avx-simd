/*
    Ascending Bitonic Sort using AVX2 SIMD (8 element vectors) and manual loop unrolling.
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <immintrin.h>
#include <chrono>
#include <limits>

// #define TEST  // Uncomment to enable test mode
constexpr int FLOAT_VEC_SIZE = 8;
constexpr int OUTER_SIZE = 182;
constexpr int c_m = std::log2(FLOAT_VEC_SIZE); // Number of in-vector sort stages
constexpr int c_totalInVecStages = c_m * (c_m + 1) / 2; // Total number of in-vector sort stages

/*
Y
Most Significant 1 is what we care about
xxxxxx1xxxxxxxx | Y >> 1 -> Y
xxxxxx11xxxxxxx | Y >> 2 -> Y
xxxxxx1111xxxxx | Y >> 4 -> Y
xxxxxx11111111x | Y >> 8 -> Y
xxxxxx111111111
Y += 1
xxxxx1000000000
*/
inline int NextPow2(int n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

void HandleStage(__m256& vec0, const int offsetsArray[FLOAT_VEC_SIZE], const int signArray[FLOAT_VEC_SIZE], bool currentSortDir) {
    __m256i offsets = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(offsetsArray));
    __m256i sign = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(signArray));

    __m256i signFlip = _mm256_sub_epi32(_mm256_set1_epi32(-1), sign);
    sign = _mm256_blendv_epi8(sign, signFlip, _mm256_set1_epi32(currentSortDir ? 0 : -1)); // Apply sort direction

    __m256 vec1 = _mm256_permutevar8x32_ps(vec0, offsets);

    __m256 mask = _mm256_cmp_ps(vec0, vec1, _CMP_GT_OQ); // Compare vec0 and vec1
    __m256i maskInt = _mm256_castps_si256(mask); // Cast comparison result to integer mask
    maskInt = maskInt ^ sign; // Apply sign mask

    mask = _mm256_castsi256_ps(maskInt); // Cast back to float for blending
    vec0 = _mm256_blendv_ps(vec0, vec1, mask);
}

void HandleFullVectors(__m256& vec0,
                       const int offsetsArray[3][FLOAT_VEC_SIZE],
                       const int signArray[c_totalInVecStages][FLOAT_VEC_SIZE],
                       bool currentSortDir)
{
    for (int i = 0; i < c_m; ++i)
    {
        // Stages
        int stageOffset = i * (i + 1) / 2; // Offset for signArray
        for (int j = 0; j <= i; j++) {
            // Sub Stages
            HandleStage(vec0, offsetsArray[i - j], signArray[stageOffset + j], currentSortDir);
        }
    }
}

#ifdef TEST
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
        std::cerr << "Usage: " << argv[0] << " <size> <dir[0/1]>" << std::endl;
        return 1;
    }
    unsigned int sortSize = std::stoul(argv[1]);
    bool sortDir = std::stoi(argv[2]) != 0;
    ; // true for ascending, false for descending
    std::vector<std::vector<float>> in0(OUTER_SIZE, std::vector<float>(sortSize)),
        out0(OUTER_SIZE, std::vector<float>(sortSize));
    auto& out0Ref = in0;
#ifndef TEST
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

    auto refTimeStart = std::chrono::high_resolution_clock::now();

    // Stores the unique elementwise comparison offset within the vector for the comparisons within the vector
    int offsetsArray[3][FLOAT_VEC_SIZE] = {{1, 0, 3, 2, 5, 4, 7, 6},
                                            {2, 3, 0, 1, 6, 7, 4, 5},
                                            {4, 5, 6, 7, 0, 1, 2, 3}};
    // The direction per element for the first 3 stages (all within the vector)
    int signArray[c_totalInVecStages][FLOAT_VEC_SIZE] = {{0, -1, -1, 0, 0, -1, -1, 0}, // 1.1
                                                            {0, 0, -1, -1, -1, -1, 0, 0}, // 2.1
                                                            {0, -1, 0, -1, -1, 0, -1, 0}, // 2.2
                                                            {0, 0, 0, 0, -1, -1, -1, -1}, // 3.1
                                                            {0, 0, -1, -1, 0, 0, -1, -1}, // 3.2
                                                            {0, -1, 0, -1, 0, -1, 0, -1}}; // 3.3
    unsigned int sortLog2Size = std::log2(sortSize);
    unsigned int sortPaddedSize = sortSize;
    if (sortSize & (sortSize - 1))
    {
        sortPaddedSize = NextPow2(sortSize);
        sortLog2Size++;
        float padElement = -std::numeric_limits<float>::max();
        if (sortDir)
        {
            padElement = std::numeric_limits<float>::max();
        }
        for (int k = 0; k < OUTER_SIZE; ++k) {
            out0[k].resize(sortPaddedSize, padElement); // Pad with max/min value
        }
    }

    for (int k = 0; k < OUTER_SIZE; k ++) {
        // Create bitonic sequences from Stages 1.1 to 3.3
        bool currentSortDir = sortDir;
        float *vec0_ptr = reinterpret_cast<float*>(out0[k].data());
        for (int l = 0; l < sortPaddedSize; l += FLOAT_VEC_SIZE) {
            __m256 vec0 = _mm256_loadu_ps(vec0_ptr); // Load 8 floats into a vector
            HandleFullVectors(vec0, offsetsArray, signArray, currentSortDir);
            _mm256_storeu_ps(vec0_ptr, vec0); // Store the permuted vector back to out0
            currentSortDir = !currentSortDir; // Alternate sort direction for next vector
            vec0_ptr += FLOAT_VEC_SIZE;
        }

        for (int i = c_m; i < sortLog2Size; i++)
        {
            int pow2Size = std::pow(2, i + 1); //  num of elements attached to a sign change
            for (int j = 0; i - j >= c_m; j++)
            {
                currentSortDir = sortDir;
                int boxOffset = std::pow(2, (i - j + 1));
                for (int x = 0; x < sortPaddedSize; x += boxOffset) // relevant when we have minimum 2 vectors
                {
                    __m256 signFlip = _mm256_castsi256_ps(_mm256_set1_epi32(0));
                    if (x > 0 && (x % pow2Size == 0))
                    {
                        currentSortDir = !currentSortDir;
                    }
                    if (!currentSortDir)
                    {
                        signFlip = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
                    }
                    float *vec00_ptr = reinterpret_cast<float*>(out0[k].data() + x);
                    float *vec01_ptr = reinterpret_cast<float*>(out0[k].data() + x + (boxOffset / 2));
                    for (int y = 0; y < (boxOffset / (2 * FLOAT_VEC_SIZE)); y++) // num Vector pairs
                    {
                        __m256 vec00 = _mm256_loadu_ps(vec00_ptr);
                        __m256 vec01 = _mm256_loadu_ps(vec01_ptr);
                        __m256 mask00 = _mm256_cmp_ps(vec00, vec01, _CMP_GT_OQ);
                        __m256 mask01 = _mm256_cmp_ps(vec00, vec01, _CMP_LE_OQ);
                        __m256 temp00 = _mm256_blendv_ps(vec00, vec01, mask00);
                        __m256 temp01 = _mm256_blendv_ps(vec00, vec01, mask01);
                        vec00 = _mm256_blendv_ps(temp00, temp01, signFlip);
                        vec01 = _mm256_blendv_ps(temp01, temp00, signFlip);
                        _mm256_storeu_ps(vec00_ptr, vec00);
                        _mm256_storeu_ps(vec01_ptr, vec01);
                        vec00_ptr += FLOAT_VEC_SIZE;
                        vec01_ptr += FLOAT_VEC_SIZE;
                    }
                }
            }
            currentSortDir = sortDir;
            float* vec0_ptr = reinterpret_cast<float*>(out0[k].data());
            for (int l = 0; l < sortPaddedSize; l += FLOAT_VEC_SIZE) {
                if (l > 0 && (l % pow2Size == 0))
                {
                    currentSortDir = !currentSortDir; // Alternate sort direction for next box
                }
                __m256 vec0 = _mm256_loadu_ps(vec0_ptr); // Load 8 floats into a vector
                HandleFullVectors(vec0, offsetsArray, signArray, currentSortDir);
                _mm256_storeu_ps(vec0_ptr, vec0); // Store the permuted vector back to out0
                vec0_ptr += FLOAT_VEC_SIZE;
            }
        }
    }
    std::cout << "SIMD time: " <<
        std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::high_resolution_clock::now() - refTimeStart).count() << " us" << std::endl;

    if (sortSize & (sortSize - 1))
    {
        for (int k = 0; k < OUTER_SIZE; ++k) {
            out0[k].resize(sortSize); // Remove padding
        }
    }

    PRINT_DATA(out0, "Output: \n");

    if (out0 != in0)
    {
        std::cout << "Sort mismatch" << std::endl;
    }
}