#include <iostream>
#include <vector> 
#include <utility>


#include <array>

#include "mygrad/mygrad.hpp"

#include "processData.hpp"


int main() {
    // Dataset catsData = loadCatImages(0.8, 0.1, 0.1);

    TensorDims d {1, 2, 3};
    d = {1, 2, 3, 4};
    d.print();

    // std::array<int, 3> a = {1, 2, 3};
    // std::initializer_list<int> b = {1, 2, 3, 4};
    // std::copy(b.begin(), b.end(), a.begin());
    // for (size_t i = 0; i < 4; i++) {
    //     std::cout << a[i] << "\n";
    // }



    // Conv2d c(2, 4, 3, 1, 1); // 2 in channels, 4 out channels, 3 kernel size, 1 stride, s1 padding size
    // Tensor t(std::vector(18, 2.),

    //         {1, 2, 3, 3}); 

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