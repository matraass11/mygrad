#include <iostream>
#include <array>
#include "tensor.hpp"
#include "tensorStruct.hpp"

int main(){
    tensorStruct<4> input({0.2, 2.1, 0.4, 1}, {1, 4});
    tensorStruct<8> w1({1, 2, 3, 4, 5, 6, 7, 8}, {4, 2});
    tensorStruct<2> w2({5, 7}, {2, 1});

    double p1Data[2], p1Grad[2] {};
    TensorMatMulProduct p1(p1Data, p1Grad, input.tensor, w1.tensor);
    double p2Data[1], p2Grad[1] {};
    TensorMatMulProduct p2(p2Data, p2Grad, p1, w2.tensor);

    p1.matMul2dIntoSelf(input.tensor, w1.tensor);
    p2.matMul2dIntoSelf(p1, w2.tensor);
    p2.print();

    p2.setAllGradsTo(1);
    p2.backwardFurther();
    p1.backwardFurther(); 
    w1.tensor.printGrad(); 

} 