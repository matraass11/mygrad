#include <iostream>
#include <array>
#include "tensor.hpp"
#include "tensorStruct.hpp"

int main(){
    
    tensor_s<4> input({0.2, 2.1, 0.4, 1}, {1, 4});
    tensor_s<8> w1({1, 2, 3, 4, 5, 6, 7, 8}, {4, 2});
    tensor_s<2> b1({0.3, 0.4}, {1, 2});
    tensor_s<2> w2({5, 7}, {2, 1});
    tensor_s<1> b2({0.21}, {1, 1});

    tensor_matmul_product_s<2> p1(input.tensor, w1.tensor);
    tensor_sum_s<2> s1(p1.tensor, b1.tensor);
    tensor_matmul_product_s<1> p2(s1.tensor, w2.tensor);
    tensor_sum_s<2> s2(p2.tensor, b2.tensor);
    s2.tensor.print();

    s2.tensor.setAllGradsTo(1);
    s2.tensor.backwardFurther();
    w1.tensor.printGrad(); 

} 