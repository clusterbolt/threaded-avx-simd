#include<iostream>
#include<chrono>
#include<thread>
#include<vector>

int main(int argc, char* argv[])
{
    srand(static_cast<unsigned int>(time(0))); // Seed for random number generation
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <size>" << std::endl;
        return 1;
    }

    int nThreads = std::thread::hardware_concurrency(); // Get the number of threads supported by the system
    size_t size = 0;
    size = std::stoul(argv[1]);
    int offset = (size + nThreads - 1) / nThreads;
    // std::cout << "Number of threads, offset: " << nThreads << ", " << offset << std::endl;
    float* dataArray = new float[size];

    std::vector<std::thread> threads;
    for (size_t i = 0; i < size; ++i) {
        dataArray[i] = rand() % 1000 / 100.0f; // Random float between 0.0 and 9.99
    }

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    float *lSum = new float[nThreads];
    for (int i = 0; i < nThreads; ++i) {
        threads.emplace_back([=,&lSum,&dataArray]() {
            float localSum = 0.0f;
            int start = i * offset;
            int limit = std::min((i + 1) * offset, (int)size);
            // #pragma unroll 2
            for (size_t j = start; j < limit; j++) {
                localSum += dataArray[j];
            }
            lSum[i] = localSum;
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    float sum = 0.0f;
    for (size_t i = 0; i < nThreads; ++i) {
        sum += lSum[i];
    }
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;
    std::cout << "Sum: " << sum << std::endl;

    delete[] dataArray;
    delete[] lSum;
}