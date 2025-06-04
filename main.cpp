#include <iostream>
#include <array>
#include "tensor.hpp"
#include "tensorStruct.hpp"

int main(){
    tensor_s<4> input({0.2, 2.1, 0.4, 1}, {1, 4});
    tensor_s<8> w1({1, 2, 3, 4, 5, 6, 7, 8}, {4, 2});
    tensor_s<2> w2({5, 7}, {2, 1});

    tensor_matmul_product_s<2> p1(input.tensor, w1.tensor);
    tensor_matmul_product_s<1> p2(p1.tensor, w2.tensor);

    p1.tensor.matMul2dIntoSelf(input.tensor, w1.tensor);
    p2.tensor.matMul2dIntoSelf(p1.tensor, w2.tensor);
    p2.tensor.print();

    p2.tensor.setAllGradsTo(1);
    p2.tensor.backwardFurther();
    p1.tensor.backwardFurther(); 
    w1.tensor.printGrad(); 

} 