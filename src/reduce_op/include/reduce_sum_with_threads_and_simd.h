#pragma once

#include "api.h"
#include <memory>

class IReduceSumOp {
public:
    virtual ~IReduceSumOp() = default;
    virtual void execute(float *dataArray, unsigned int arrSize, float& sum) = 0;
};

class ReduceSumDefaultOp : public IReduceSumOp {
public:
    void execute(float *dataArray, unsigned int arrSize, float& sum) override;
};