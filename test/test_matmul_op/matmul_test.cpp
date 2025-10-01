#include <iostream>
#include <chrono>
#include <vector>
#include <cassert> // assert
#include <cmath> // std::fabs
#include "api.h"

void TestEqualityWithULP(const std::vector<float>& in1, const std::vector<float>& in2, unsigned int ulpDiff)
{
    assert(in1.size() == in2.size());
    for (int i = 0; i < in1.size(); i++)
    {
        float ref = in1[i];
        float out = in2[i];
        if (std::fabs(*reinterpret_cast<int*>(&ref) - *reinterpret_cast<int*>(&out)) > ulpDiff)
        {
            std::cout << std::fabs(*reinterpret_cast<int*>(&ref) - *reinterpret_cast<int*>(&out)) << std::endl;
        }
        assert(ulpDiff >= std::fabs(*reinterpret_cast<int*>(&ref) - *reinterpret_cast<int*>(&out)));
    }
}

void PrintMatrix(const std::vector<float>& in, const size_t stride)
{
    int col = 0;
    for (auto& e : in)
    {
        std::cout << " " << e;
        col ++;
        if (col == stride)
        {
            std::cout << "\n";
            col = 0;
        }
    }
    std::cout << std::endl;
}

constexpr int FLOAT_VEC_SIZE = 8;

using std::chrono::high_resolution_clock, std::chrono::milliseconds;

void RefMatMul(const std::vector<float>& matIn1, const std::vector<float>& matIn2, std::vector<float>& matOut, size_t m, size_t n, size_t p)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < p; j++)
        {
            matOut[p * i + j] = 0.0f;
            for (int k = 0; k < n; k++)
            {
                matOut[p * i + j] += matIn1[n * i + k] * matIn2[k * p + j];
            }
        }
    }
}

int main(int argc, char* argv[])
{
    srand(static_cast<unsigned int>(time(0))); // Seed for random number generation
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " sizeM sizeN sizeP" << std::endl;
        return 1;
    }

    size_t sizeM = 0, sizeN = 0, sizeP = 0;
    sizeM = std::stoul(argv[1]);
    sizeN = std::stoul(argv[2]);
    sizeP = std::stoul(argv[3]);
    std::vector<float> matIn1(sizeM * sizeN), matIn2(sizeN * sizeP), matOut(sizeM * sizeP);
    std::vector<float> matOutRef(sizeM * sizeP);

    for (size_t i = 0; i < matIn1.size(); ++i) {
        matIn1[i] = rand() % 1000 / 100.0f; // Random float between 0.0 and 9.99
    }
    for (size_t i = 0; i < matIn2.size(); ++i) {
        matIn2[i] = rand() % 1000 / 100.0f; // Random float between 0.0 and 9.99
    }

    // PrintMatrix(matIn1, sizeN);
    // PrintMatrix(matIn2, sizeP);

    high_resolution_clock::time_point start = high_resolution_clock::now();
    RefMatMul(matIn1, matIn2, matOutRef, sizeM, sizeN, sizeP);
    std::cout << "Reference time taken: " <<
        std::chrono::duration_cast<milliseconds>(high_resolution_clock::now() - start).count() <<
        " ms" << std::endl;

    start = high_resolution_clock::now();
    mat_mul_execute(matMulType_t::MAT_MUL_GATHER, matIn1.data(), matIn2.data(), matOut.data(), sizeM, sizeN, sizeP);
    std::cout << "Optimized with gather time taken: " <<
        std::chrono::duration_cast<milliseconds>(high_resolution_clock::now() - start).count() <<
        " ms" << std::endl;
    TestEqualityWithULP(matOutRef, matOut, 50);

    start = high_resolution_clock::now();
    mat_mul_execute(matMulType_t::MAT_MUL, matIn1.data(), matIn2.data(), matOut.data(), sizeM, sizeN, sizeP);
    std::cout << "Optimized with partial product reuse time taken: " <<
        std::chrono::duration_cast<milliseconds>(high_resolution_clock::now() - start).count() <<
        " ms" << std::endl;

    // PrintMatrix(matOut, sizeP);
    TestEqualityWithULP(matOutRef, matOut, 50);
}
