#pragma once

#include "api.h"
#include <memory>

class IMatMulOp {
public:
    virtual ~IMatMulOp() = default;
    virtual void execute(float* matIn1, float* matIn2, float* matOut, size_t sizeM, size_t sizeN, size_t sizeP) = 0;
};

class MatMulDefaultOp : public IMatMulOp {
public:
    void execute(float* matIn1, float* matIn2, float* matOut, size_t sizeM, size_t sizeN, size_t sizeP) override;
};

class MatMulGatherOp : public IMatMulOp {
public:
    void execute(float* matIn1, float* matIn2, float* matOut, size_t sizeM, size_t sizeN, size_t sizeP) override;
};