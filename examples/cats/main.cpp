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

    Conv2d c(2, 4, 3, 1, 1); // 2 in channels, 4 out channels, 3 kernel size, 1 stride, 1 padding size
    MaxPool2d m(2);
    Tensor t(std::vector(18, 2.),

            {1, 2, 3, 3}); 
            // 1 batch, 2 channels, 3x3



    for (size_t i = 0; i < c.kernels.length; i++) {
        c.kernels.data[i] = 1;
    }

    c.forward(t);
    m.forward(c.outputTensor);
    m.outputTensor.print();
    std::cout << m.outputTensor.dimensions;

    for (size_t i = 0; i < m.outputTensor.length; i++) {
        m.outputTensor.grads[i] = 1;
    }
    
    m.backward(); 
    c.backward();
    // c.outputTensor.printGrad();
    c.biases.printGrad();
    c.kernels.printGrad();
}