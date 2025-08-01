#include<iostream>
#include<chrono>


int main(int argc, char* argv[])
{
    srand(static_cast<unsigned int>(time(0))); // Seed for random number generation
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <size>" << std::endl;
        return 1;
    }

    size_t size = 0;
    size = std::stoul(argv[1]);
    float* dataArray = new float[size];

    for (size_t i = 0; i < size; ++i) {
        dataArray[i] = rand() % 1000 / 100.0f; // Random float between 0.0 and 9.99
    }
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        sum += dataArray[i];
    }
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms" << std::endl;
    std::cout << "Sum: " << sum << std::endl;

    delete[] dataArray;
}