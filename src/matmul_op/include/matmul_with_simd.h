#pragma once

extern "C" {
    void MatMulFunc(float* matIn1, float* matIn2, float* matOut, size_t sizeM, size_t sizeN, size_t sizeP);
    void MatMulGatherFunc(float* matIn1, float* matIn2, float* matOut, size_t sizeM, size_t sizeN, size_t sizeP);
}