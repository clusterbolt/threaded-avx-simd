#pragma once

#include "api.h"
#include <memory>

class IBitonicSortOp {
public:
    virtual ~IBitonicSortOp() = default;
    virtual void execute(std::vector<std::vector<float>>&, unsigned int, unsigned int, bool) = 0;
};

class BitonicSortDefaultOp : public IBitonicSortOp {
public:
    void execute(std::vector<std::vector<float>>&, unsigned int, unsigned int, bool) override;
};

class BitonicSortAltOp : public IBitonicSortOp {
public:
    void execute(std::vector<std::vector<float>>&, unsigned int, unsigned int, bool) override;
};