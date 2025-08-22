#include <iostream>
#include <vector> 
#include <utility>


#include <array>
#include <thread>
#include <functional>
#include <mutex>
#include <chrono>

#include "mygrad/mygrad.hpp"

#include "processData.hpp"


int main() {

    using namespace mygrad;

    // Dataset catsData = loadCatImages(0.8, 0.1, 0.1);

    // Conv2d c(2, 4, 3, 1, 1); // 2 in channels, 4 out channels, 3 kernel size, 1 stride, s1 padding size
    // Tensor t(std::vector(18, 2.),

    //         {1, 2, 3, 3}); 

    // ---
    std::array<int, 8> arr {};


    for (size_t i = 0; i < ThreadPool::size(); i++) {
        ThreadPool::push([&arr, i] { 
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            for (size_t j = 0; j < 9999999; j++) {
                arr[i] += j;
            }
        });
    }

    ThreadPool::waitUntilDone();


    for (size_t i = 0; i < 8; i++) {
        std::cout << arr[i] << "\n";
    }
    // ---



    // std::cout << t.dimensions;
            // 1 batch, 2 channels, 3x3

    // for (size_t i = 0; i < c.kernels.length; i++) {
    //     c.kernels.data[i] = 1;
    // }


    // c.print();
    // c.forward(t);
    // c.outputTensor.print();

    for (size_t i = 0; i < c.outputTensor.length; i++) {
        c.outputTensor.grads[i] = 1;
    }
    
    c.backward();
    c.biases.printGrad();
}