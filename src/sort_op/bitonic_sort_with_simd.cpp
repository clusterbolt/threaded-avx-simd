/*
    Bitonic Sort using AVX2 SIMD.
*/

#include <vector>
#include <array>
#include <immintrin.h>
#include <limits>
#include <cmath>
#include <thread>
#include "bitonic_sort_with_simd.h"
#include <iostream>

void bitonic_sort_execute(bitonicSortType_t type, std::vector<std::vector<float>>& out0, unsigned int outerSize, unsigned int sortSize, bool sortDir)
{
    std::unique_ptr<IBitonicSortOp> strategy;
    switch(type)
    {
        case bitonicSortType_t::BITONIC_SORT:
            strategy = std::make_unique<BitonicSortDefaultOp>();
            break;
        case bitonicSortType_t::BITONIC_SORT_ALT:
            strategy = std::make_unique<BitonicSortAltOp>();
            break;
        default:
            strategy = std::make_unique<BitonicSortDefaultOp>();
    };

    strategy->execute(out0, outerSize, sortSize, sortDir);
}

constexpr int FLOAT_VEC_SIZE = 8;
constexpr int LOG_FLOAT_VEC_SIZE = 3;
constexpr int c_m = std::log2(FLOAT_VEC_SIZE); // Number of in-vector sort stages
constexpr int c_totalInVecStages = c_m * (c_m + 1) / 2; // Total number of in-vector sort stages

// Stores the unique elementwise comparison offset within the vector for the comparisons within the vector
constexpr int offsetsArray[LOG_FLOAT_VEC_SIZE][FLOAT_VEC_SIZE] = {{1, 0, 3, 2, 5, 4, 7, 6},
                                        {2, 3, 0, 1, 6, 7, 4, 5},
                                        {4, 5, 6, 7, 0, 1, 2, 3}};
// The direction per element for the first 3 stages (all within the vector)
constexpr int signArray[c_totalInVecStages][FLOAT_VEC_SIZE] = {{0, -1, -1, 0, 0, -1, -1, 0}, // 1.1
                                                        {0, 0, -1, -1, -1, -1, 0, 0}, // 2.1
                                                        {0, -1, 0, -1, -1, 0, -1, 0}, // 2.2
                                                        {0, 0, 0, 0, -1, -1, -1, -1}, // 3.1
                                                        {0, 0, -1, -1, 0, 0, -1, -1}, // 3.2
                                                        {0, -1, 0, -1, 0, -1, 0, -1}}; // 3.3

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

/* Handle the in-vector bitonic sort
example 1:
vec0    =  2 --> 1 // Assume arrow head represents the position for greater of the two
sign    =  0    -1 // indicates the direction of the comparison based on the arrow above
vec1    =  1     2 // permuted vec0 (notice that this is the desired order after the sort)
mask    = -1     0 // result of the comparison (vec0 > vec1)
maskInt = -1    -1 // mask ^ sign (indicates flip, i.e. use vec1 instead of vec0 for this pair)
vec0    =  1     2 // sorted

example 2:
vec0    =  2 <-- 1 // Already sorted as per the direction
sign    = -1     0
vec1    =  1     2
mask    = -1     0 // result of the comparison (vec0 > vec1)
maskInt =  0     0 // mask ^ sign (no flip)
vec0    =  2     1 // no change
*/
inline void HandleStage(__m256& vec0, const int offsetsArray[FLOAT_VEC_SIZE], const int signArray[FLOAT_VEC_SIZE], bool currentSortDir) {
    // Load offsets and sign into vectors
    __m256i offsets = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(offsetsArray));
    __m256i sign = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(signArray));

    // Create a sign mask based on the current sort direction
    __m256i signFlip = _mm256_sub_epi32(_mm256_set1_epi32(-1), sign);
    sign = _mm256_blendv_epi8(sign, signFlip, _mm256_set1_epi32(currentSortDir ? 0 : -1)); // Apply sort direction

    // Form permuted vectors which indicate the offsets that vec0 should be compared against
    __m256 vec1 = _mm256_permutevar8x32_ps(vec0, offsets);

    __m256 mask = _mm256_cmp_ps(vec0, vec1, _CMP_GT_OQ); // Compare vec0 and vec1
    __m256i maskInt = _mm256_castps_si256(mask); // Cast comparison result to integer mask
    maskInt = maskInt ^ sign; // Apply sign mask

    mask = _mm256_castsi256_ps(maskInt); // Cast back to float for blending
    vec0 = _mm256_blendv_ps(vec0, vec1, mask);
}

// Handles the last 3 sub stages of any stage (all of which have comparisons within the vector)
inline void HandleFullVectors(__m256& vec0, bool currentSortDir)
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

void BitonicSortDefaultOp::execute(std::vector<std::vector<float>>& out0, unsigned int outerSize, unsigned int sortSize, bool sortDir) {
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
        for (int k = 0; k < outerSize; ++k) {
            out0[k].resize(sortPaddedSize, padElement); // Pad with max/min value
        }
    }

    for (int k = 0; k < outerSize; k ++) {
        // Create bitonic sequences from Stages 1.1 to 3.3
        float *vec0_ptr = reinterpret_cast<float*>(out0[k].data());
        for (int l = 0; l < sortPaddedSize; l += FLOAT_VEC_SIZE) {
            __m256 vec0 = _mm256_loadu_ps(vec0_ptr + l);
            HandleFullVectors(vec0, ((l >> LOG_FLOAT_VEC_SIZE) & 1) ^ sortDir);
            _mm256_storeu_ps(vec0_ptr + l, vec0);
        }

        for (int i = c_m; i < sortLog2Size; i++)
        {
            int stageSize = std::pow(2, i + 1); //  num of elements attached to a sign change
            for (int j = 0; i - j >= c_m; j++)
            {
                bool currentSortDir = !sortDir;
                int boxOffset = std::pow(2, (i - j + 1));
                for (int y = 0; y < sortPaddedSize; y += stageSize) // num of boxes
                {
                    currentSortDir = !currentSortDir; // Alternate sort direction for next box
                    __m256 signFlip = _mm256_castsi256_ps(_mm256_set1_epi32(currentSortDir ? 0 : -1));
                    for (int x = y; x < y + stageSize; x += boxOffset) // relevant when we have minimum 2 vectors
                    {
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
            }
            bool currentSortDir = sortDir;
            float* vec0_ptr = reinterpret_cast<float*>(out0[k].data());
            for (int l = 0; l < sortPaddedSize; l += stageSize) {
                for (int m = l; m < l + stageSize; m += FLOAT_VEC_SIZE) {
                    __m256 vec0 = _mm256_loadu_ps(vec0_ptr); // Load 8 floats into a vector
                    HandleFullVectors(vec0, currentSortDir);
                    _mm256_storeu_ps(vec0_ptr, vec0); // Store the permuted vector back to out0
                    vec0_ptr += FLOAT_VEC_SIZE;
                }
                currentSortDir = !currentSortDir; // Alternate sort direction for next box
            }
        }
    }

    if (sortSize & (sortSize - 1))
    {
        for (int k = 0; k < outerSize; ++k) {
            out0[k].resize(sortSize); // Remove padding
        }
    }
}

/* Alternate Bitonic Sort Functions (Unidirectional) */

// At Substage 1 the comparisons are between set of elements in the opposite order
constexpr int offsetsArraySub1Alt[LOG_FLOAT_VEC_SIZE][FLOAT_VEC_SIZE] = {{1, 0, 3, 2, 5, 4, 7, 6},
                                                                        {3, 2, 1, 0, 7, 6, 5, 4},
                                                                        {7, 6, 5, 4, 3, 2, 1, 0}};

// The directions within the vector for the first 3 stages when ascending is the sort direction
constexpr int signArrayAscAlt[c_totalInVecStages][FLOAT_VEC_SIZE] = {{0, -1, 0, -1, 0, -1, 0, -1}, // 1.1
                                                                    {0, 0, -1, -1, 0, 0, -1, -1}, // 2.1
                                                                    {0, -1, 0, -1, 0, -1, 0, -1}, // 2.2
                                                                    {0, 0, 0, 0, -1, -1, -1, -1}, // 3.1
                                                                    {0, 0, -1, -1, 0, 0, -1, -1}, // 3.2
                                                                    {0, -1, 0, -1, 0, -1, 0, -1}}; // 3.3

// The directions within the vector for the first 3 stages when descending is the sort direction
constexpr int signArrayDscAlt[c_totalInVecStages][FLOAT_VEC_SIZE] = {{-1, 0, -1, 0, -1, 0, -1, 0}, // 1.1
                                                                    {-1, -1, 0, 0, -1, -1, 0, 0}, // 2.1
                                                                    {-1, 0, -1, 0, -1, 0, -1, 0}, // 2.2
                                                                    {-1, -1, -1, -1, 0, 0, 0, 0}, // 3.1
                                                                    {-1, -1, 0, 0, -1, -1, 0, 0}, // 3.2
                                                                    {-1, 0, -1, 0, -1, 0, -1, 0}}; // 3.3

inline void HandleStageAlt(__m256& vec0, const int* offsetsArray, const int* signArray) {
    // Load offsets and sign into vectors
    __m256i offsets = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(offsetsArray));
    __m256i sign = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(signArray));

    // Form permuted vectors which indicate the offsets that vec0 should be compared against
    __m256 vec1 = _mm256_permutevar8x32_ps(vec0, offsets);

    __m256 mask = _mm256_cmp_ps(vec0, vec1, _CMP_GT_OQ); // Compare vec0 and vec1
    __m256i maskInt = _mm256_castps_si256(mask); // Cast comparison result to integer mask
    maskInt = maskInt ^ sign; // Apply sign mask

    mask = _mm256_castsi256_ps(maskInt); // Cast back to float for blending
    vec0 = _mm256_blendv_ps(vec0, vec1, mask);
}

// Handles last 3 stages with their sub stages
inline void HandleFullVectorsAlt(__m256& vec0, bool sortDir)
{
    const int (*signArray)[8] = sortDir ? signArrayAscAlt : signArrayDscAlt;
    for (int i = 0; i < c_m; ++i)
    {
        // Stages
        int stageOffset = i * (i + 1) / 2; // Offset for signArray
        HandleStageAlt(vec0, offsetsArraySub1Alt[i], signArray[stageOffset]);
        for (int j = 1; j <= i; j++) {
            // Sub Stages
            HandleStageAlt(vec0, offsetsArray[i - j], signArray[stageOffset + j]);
        }
    }
}

void BitonicSortAltOp::execute(std::vector<std::vector<float>>& out0, unsigned int outerSize, unsigned int sortSize, bool sortDir) {
    unsigned int sortLog2Size = std::log2(sortSize);
    unsigned int sortPaddedSize = sortSize;
    if (sortSize & (sortSize - 1) || sortSize < FLOAT_VEC_SIZE)
    {
        sortPaddedSize = std::max(NextPow2(sortSize), FLOAT_VEC_SIZE);
        sortLog2Size = std::log2(sortPaddedSize);
        float padElement = -std::numeric_limits<float>::max();
        if (sortDir)
        {
            padElement = std::numeric_limits<float>::max();
        }
        for (int k = 0; k < outerSize; ++k) {
            out0[k].resize(sortPaddedSize, padElement); // Pad with max/min value
        }
    }

    for (int k = 0; k < outerSize; k ++) {
        // Create bitonic sequences from Stages 1.1 to 3.3
        float *vec0_ptr = reinterpret_cast<float*>(out0[k].data());
        for (int l = 0; l < sortPaddedSize; l += FLOAT_VEC_SIZE) {
            __m256 vec0 = _mm256_loadu_ps(vec0_ptr + l);
            HandleFullVectorsAlt(vec0, sortDir);
            _mm256_storeu_ps(vec0_ptr + l, vec0);
        }

        __m256 signFlip = _mm256_castsi256_ps(_mm256_set1_epi32(sortDir ? 0 : -1));
        __m256i offsetsReverse = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(offsetsArraySub1Alt[2]));
        for (int i = c_m; i < sortLog2Size; i++)
        {
            /* The first substage requires reversing the order of elements in the second vector
               The remaining substages work the same as the bidirectional Bitonic Sort except that
               given sort direction is used consistently */
            int stageSize = std::pow(2, i + 1);
            if (sortDir)
            {
                for (int x = 0; x < sortPaddedSize; x += stageSize) // num of boxes
                {
                    float *vec00_ptr = reinterpret_cast<float*>(out0[k].data() + x);
                    float *vec01_ptr = reinterpret_cast<float*>(out0[k].data() + x + stageSize - FLOAT_VEC_SIZE);
                    for (int y = 0; y < (stageSize / (2 * FLOAT_VEC_SIZE)); y++) // num Vector pairs
                    {
                        __m256 vec00 = _mm256_loadu_ps(vec00_ptr);
                        __m256 vec01 = _mm256_loadu_ps(vec01_ptr);
                        vec01 = _mm256_permutevar8x32_ps(vec01, offsetsReverse);
                        __m256 mask00 = _mm256_cmp_ps(vec00, vec01, _CMP_GT_OQ);
                        __m256 mask01 = _mm256_cmp_ps(vec00, vec01, _CMP_LE_OQ);
                        __m256 temp00 = _mm256_blendv_ps(vec00, vec01, mask00);
                        __m256 temp01 = _mm256_blendv_ps(vec00, vec01, mask01);
                        _mm256_storeu_ps(vec00_ptr, temp00);
                        _mm256_storeu_ps(vec01_ptr, temp01);
                        vec00_ptr += FLOAT_VEC_SIZE;
                        vec01_ptr -= FLOAT_VEC_SIZE;
                    }
                }
                for (int j = 1; i - j >= c_m; j++)
                {
                    int boxOffset = std::pow(2, (i - j + 1));
                    for (int y = 0; y < sortPaddedSize; y += stageSize) // num of boxes
                    {
                        for (int x = y; x < y + stageSize; x += boxOffset) // relevant when we have minimum 2 vectors
                        {
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
                                _mm256_storeu_ps(vec00_ptr, temp00);
                                _mm256_storeu_ps(vec01_ptr, temp01);
                                vec00_ptr += FLOAT_VEC_SIZE;
                                vec01_ptr += FLOAT_VEC_SIZE;
                            }
                        }
                    }
                }
            }
            else
            {
                for (int x = 0; x < sortPaddedSize; x += stageSize) // num of boxes
                {
                    float *vec00_ptr = reinterpret_cast<float*>(out0[k].data() + x);
                    float *vec01_ptr = reinterpret_cast<float*>(out0[k].data() + x + stageSize - FLOAT_VEC_SIZE);
                    for (int y = 0; y < (stageSize / (2 * FLOAT_VEC_SIZE)); y++) // num Vector pairs
                    {
                        __m256 vec00 = _mm256_loadu_ps(vec00_ptr);
                        __m256 vec01 = _mm256_loadu_ps(vec01_ptr);
                        vec01 = _mm256_permutevar8x32_ps(vec01, offsetsReverse);
                        __m256 mask00 = _mm256_cmp_ps(vec00, vec01, _CMP_LE_OQ);
                        __m256 mask01 = _mm256_cmp_ps(vec00, vec01, _CMP_GT_OQ);
                        __m256 temp00 = _mm256_blendv_ps(vec00, vec01, mask00);
                        __m256 temp01 = _mm256_blendv_ps(vec00, vec01, mask01);
                        _mm256_storeu_ps(vec00_ptr, temp00);
                        _mm256_storeu_ps(vec01_ptr, temp01);
                        vec00_ptr += FLOAT_VEC_SIZE;
                        vec01_ptr -= FLOAT_VEC_SIZE;
                    }
                }
                for (int j = 1; i - j >= c_m; j++)
                {
                    int boxOffset = std::pow(2, (i - j + 1));
                    for (int y = 0; y < sortPaddedSize; y += stageSize) // num of boxes
                    {
                        for (int x = y; x < y + stageSize; x += boxOffset) // relevant when we have minimum 2 vectors
                        {
                            float *vec00_ptr = reinterpret_cast<float*>(out0[k].data() + x);
                            float *vec01_ptr = reinterpret_cast<float*>(out0[k].data() + x + (boxOffset / 2));
                            for (int y = 0; y < (boxOffset / (2 * FLOAT_VEC_SIZE)); y++) // num Vector pairs
                            {
                                __m256 vec00 = _mm256_loadu_ps(vec00_ptr);
                                __m256 vec01 = _mm256_loadu_ps(vec01_ptr);
                                __m256 mask00 = _mm256_cmp_ps(vec00, vec01, _CMP_LE_OQ);
                                __m256 mask01 = _mm256_cmp_ps(vec00, vec01, _CMP_GT_OQ);
                                __m256 temp00 = _mm256_blendv_ps(vec00, vec01, mask00);
                                __m256 temp01 = _mm256_blendv_ps(vec00, vec01, mask01);
                                _mm256_storeu_ps(vec00_ptr, temp00);
                                _mm256_storeu_ps(vec01_ptr, temp01);
                                vec00_ptr += FLOAT_VEC_SIZE;
                                vec01_ptr += FLOAT_VEC_SIZE;
                            }
                        }
                    }
                }
            }
            float* vec0_ptr = reinterpret_cast<float*>(out0[k].data());
            for (int l = 0; l < sortPaddedSize; l += stageSize) {
                for (int m = l; m < l + stageSize; m += FLOAT_VEC_SIZE) {
                    __m256 vec0 = _mm256_loadu_ps(vec0_ptr); // Load 8 floats into a vector
                    HandleFullVectors(vec0, sortDir);
                    _mm256_storeu_ps(vec0_ptr, vec0); // Store the permuted vector back to out0
                    vec0_ptr += FLOAT_VEC_SIZE;
                }
            }
        }
    }

    if (sortSize != sortPaddedSize)  // If padding was applied
    {
        for (int k = 0; k < outerSize; ++k) {
            out0[k].resize(sortSize); // Remove padding
        }
    }
}