// All Ops APIs
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Reduce Sum algorithm types
typedef enum {
    MAT_MUL,
    MAT_MUL_GATHER
} matMulType_t;

void mat_mul_execute(matMulType_t type, float* matIn1, float* matIn2, float* matOut, size_t sizeM, size_t sizeN, size_t sizeP);

typedef enum {
    BITONIC_SORT,
    BITONIC_SORT_ALT
} bitonicSortType_t;

void bitonic_sort_execute(bitonicSortType_t type, std::vector<std::vector<float>>&, unsigned int, unsigned int, bool);

typedef enum {
    REDUCE_SUM
} reduceSumType_t;

void reduce_sum_execute(reduceSumType_t type, float *dataArray, unsigned int arrSize, float& sum);

#ifdef __cplusplus
}
#endif