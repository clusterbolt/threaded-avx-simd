#include <vector>
#include <immintrin.h>
#include "matmul_with_simd.h"

constexpr int FLOAT_VEC_SIZE = 8;

void mat_mul_execute(matMulType_t type, float* matIn1, float* matIn2, float* matOut, size_t sizeM, size_t sizeN, size_t sizeP)
{
    std::unique_ptr<IMatMulOp> strategy;
    switch(type)
    {
        case matMulType_t::MAT_MUL:
            strategy = std::make_unique<MatMulDefaultOp>();
            break;
        case matMulType_t::MAT_MUL_GATHER:
            strategy = std::make_unique<MatMulGatherOp>();
            break;
        default:
            strategy = std::make_unique<MatMulDefaultOp>();
    };

    strategy->execute(matIn1, matIn2, matOut, sizeM, sizeN, sizeP);
}

void MatMulGatherOp::execute(float* matIn1, float* matIn2, float* matOut, size_t sizeM, size_t sizeN, size_t sizeP)
{
    // Offsets = 0, P, 2P, 3P, 4P, 5P, 6P, 7P
    __m256i idx = _mm256_setr_epi32(0, sizeP, sizeP << 1, sizeP + (sizeP << 1), sizeP << 2, (sizeP << 2) + sizeP, (sizeP << 2) + (sizeP << 1), (sizeP << 3) - sizeP);
    int i = 0;
    for (; i + 1 < sizeM; i += 2)
    {
        for (int j = 0; j < sizeP; j++)
        {
            __m256 mAcc = _mm256_setzero_ps();
            __m256 mAcc_2 = _mm256_setzero_ps();
            int k = 0;
            float scalarSum = 0.0f;
            float scalarSum_2 = 0.0f;
            __m256 m2   = _mm256_i32gather_ps(&matIn2[k * sizeP + j], idx, 4);
            for (; k + FLOAT_VEC_SIZE - 1 < sizeN; k += FLOAT_VEC_SIZE)
            {
                __m256 m1 = _mm256_loadu_ps(&matIn1[sizeN * i + k]);
                __m256 m1_2 = _mm256_loadu_ps(&matIn1[sizeN * (i + 1) + k]);
                mAcc = _mm256_fmadd_ps(m1, m2, mAcc); // Fused multiply-add
                mAcc_2 = _mm256_fmadd_ps(m1_2, m2, mAcc_2); // Fused multiply-add
                m2   = _mm256_i32gather_ps(&matIn2[(k + FLOAT_VEC_SIZE) * sizeP + j], idx, 4);
            }
            for (; k < sizeN; k++)
            {
                scalarSum += matIn1[sizeN * i + k] * matIn2[k * sizeP + j];
                scalarSum_2 += matIn1[sizeN * (i + 1) + k] * matIn2[k * sizeP + j];
            }
            // Horizontal add within vsum vector
            {
                __m128 low = _mm256_castps256_ps128(mAcc);             // lower 128 bits
                __m128 high = _mm256_extractf128_ps(mAcc, 1);          // upper 128 bits
                __m128 sum128 = _mm_add_ps(low, high);                 // sum low and high parts

                // Now horizontally sum 4 floats in sum128
                sum128 = _mm_hadd_ps(sum128, sum128); // [a1 + a2] [a3 + a4] ....
                sum128 = _mm_hadd_ps(sum128, sum128); // [a1 + a2 + a3 + a4] [a1 + a2 + a3 + a4] ....

                scalarSum += _mm_cvtss_f32(sum128); // Add scalar sum to the result
                matOut[sizeP * i + j] = scalarSum;
            }
            {
                __m128 low = _mm256_castps256_ps128(mAcc_2);             // lower 128 bits
                __m128 high = _mm256_extractf128_ps(mAcc_2, 1);          // upper 128 bits
                __m128 sum128 = _mm_add_ps(low, high);                 // sum low and high parts

                // Now horizontally sum 4 floats in sum128
                sum128 = _mm_hadd_ps(sum128, sum128); // [a1 + a2] [a3 + a4] ....
                sum128 = _mm_hadd_ps(sum128, sum128); // [a1 + a2 + a3 + a4] [a1 + a2 + a3 + a4] ....

                scalarSum_2 += _mm_cvtss_f32(sum128); // Add scalar sum to the result
                matOut[sizeP * (i + 1) + j] = scalarSum_2;
            }
        }
    }

    for (; i < sizeM; i++)
    {
        for (int j = 0; j < sizeP; j++)
        {
            __m256 mAcc = _mm256_setzero_ps();
            int k = 0;
            float scalarSum = 0.0f;
            float scalarSum_2 = 0.0f;
            __m256 m2   = _mm256_i32gather_ps(&matIn2[k * sizeP + j], idx, 4);
            for (; k + FLOAT_VEC_SIZE - 1 < sizeN; k += FLOAT_VEC_SIZE)
            {
                __m256 m1 = _mm256_loadu_ps(&matIn1[sizeN * i + k]);
                mAcc = _mm256_fmadd_ps(m1, m2, mAcc); // Fused multiply-add
                m2   = _mm256_i32gather_ps(&matIn2[(k + FLOAT_VEC_SIZE) * sizeP + j], idx, 4);
            }
            for (; k < sizeN; k++)
            {
                scalarSum += matIn1[sizeN * i + k] * matIn2[k * sizeP + j];
            }
            // Horizontal add within vsum vector
            {
                __m128 low = _mm256_castps256_ps128(mAcc);             // lower 128 bits
                __m128 high = _mm256_extractf128_ps(mAcc, 1);          // upper 128 bits
                __m128 sum128 = _mm_add_ps(low, high);                 // sum low and high parts

                // Now horizontally sum 4 floats in sum128
                sum128 = _mm_hadd_ps(sum128, sum128); // [a1 + a2] [a3 + a4] ....
                sum128 = _mm_hadd_ps(sum128, sum128); // [a1 + a2 + a3 + a4] [a1 + a2 + a3 + a4] ....

                scalarSum += _mm_cvtss_f32(sum128); // Add scalar sum to the result
                matOut[sizeP * i + j] = scalarSum;
            }
        }
    }
}

void MatMulDefaultOp::execute(float* matIn1, float* matIn2, float* matOut, size_t sizeM, size_t sizeN, size_t sizeP)
{
    int i = 0;
    __m256 mAccZero = _mm256_setzero_ps();
    for (; i + 1 < sizeM; i += 1)
    {
        int j = 0;
        int k;
        __m256 mAcc;
        for (; j + 1 < sizeN; j += 2)
        {
            __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(-(j != 0))); // = 0 when j == 0, -1 otherwise
            __m256 m11 = _mm256_set1_ps(matIn1[sizeN * i + j]);
            __m256 m12 = _mm256_set1_ps(matIn1[sizeN * i + j + 1]);
            k = 0;
            for (; k + FLOAT_VEC_SIZE - 1 < sizeP; k += FLOAT_VEC_SIZE)
            {
                __m256 mAccNew = _mm256_loadu_ps(&matOut[sizeP * i + k]);
                mAcc = _mm256_blendv_ps(mAccZero, mAccNew, mask);
                __m256 m21 = _mm256_loadu_ps(&matIn2[sizeP * j + k]);
                __m256 m22 = _mm256_loadu_ps(&matIn2[sizeP * (j + 1) + k]);
                __m256 mOut1 = _mm256_fmadd_ps(m11, m21, mAcc); // Fused multiply-add
                __m256 mOut2 = _mm256_fmadd_ps(m12, m22, mOut1);
                _mm256_storeu_ps(&matOut[sizeP * i + k], mOut2);
            }

            if (j == 0)
            {
                for (; k < sizeP; k++)
                {
                    matOut[sizeP * i + k] = matIn1[sizeN * i + j] * matIn2[sizeP * j + k];
                    matOut[sizeP * i + k] += matIn1[sizeN * i + j + 1] * matIn2[sizeP * (j + 1) + k];
                }
            }
            else
            {
                for (; k < sizeP; k++)
                {
                    matOut[sizeP * i + k] += matIn1[sizeN * i + j] * matIn2[sizeP * j + k];
                    matOut[sizeP * i + k] += matIn1[sizeN * i + j + 1] * matIn2[sizeP * (j + 1) + k];
                }
            }
        }

        if (j < sizeN)
        {
            __m256 m11 = _mm256_set1_ps(matIn1[sizeN * i + j]);
            for (k = 0; k + FLOAT_VEC_SIZE - 1 < sizeP; k += FLOAT_VEC_SIZE)
            {
                __m256 mAcc = _mm256_loadu_ps(&matOut[sizeP * i + k]);
                __m256 m2 = _mm256_loadu_ps(&matIn2[sizeP * j + k]);
                mAcc = _mm256_fmadd_ps(m11, m2, mAcc); // Fused multiply-add
                _mm256_storeu_ps(&matOut[sizeP * i + k], mAcc);
            }

            for (; k < sizeP; k++)
            {
                matOut[sizeP * i + k] += matIn1[sizeN * i + j] * matIn2[sizeP * j + k];
            }
        }
    }
}
