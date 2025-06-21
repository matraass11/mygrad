#include <iostream>
#include <array>
#include <fstream>

#include "tensor.hpp"
#include "tensorStruct.hpp"
#include "functions.hpp"
#include "helper.hpp"
#include "model.hpp"

int main(){
    
    Tensor_s<4> input({0.2, 2.1, 0.4, 1}, {1, 4});
    Tensor_s<8> w1({1, 2, 3, 4, 5, 6, 7, 8}, {4, 2});
    Tensor_s<2> b1({0.3, 0.4}, {1, 2});
    Tensor_s<2> w2({5, 7}, {2, 1});
    Tensor_s<1> b2({0.21}, {1, 1});

    TensorMatmulProduct_s<2> p1(input.tensor, w1.tensor);
    TensorSum_s<2> s1(p1.tensor, b1.tensor);
    TensorMatmulProduct_s<1> p2(s1.tensor, w2.tensor);
    TensorSum_s<1> s2(p2.tensor, b2.tensor);

    MatMul2d m1(input.tensor, w1.tensor, p1.tensor), m2(s1.tensor, w2.tensor, p2.tensor);
    Sum sum1(p1.tensor, b1.tensor, s1.tensor), sum2(p2.tensor, b2.tensor, s2.tensor);
    m1.forward(), sum1.forward(), m2.forward(), sum2.forward();

    s2.tensor.gradArrayPtr[0] = 1;
    sum2.backward(), m2.backward(), sum1.backward(), m1.backward();

    s2.tensor.print();
    b1.tensor.printGrad(); 

    // Model m;
    // m.print();
    // m.save("model");

    Model model2("model");
    model2.print();
    





    // std::cout << v << "\n";
} 