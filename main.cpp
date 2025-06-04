#include <iostream>
#include <array>
#include "tensor.hpp"

int main(){
    double inputData[4] = {0.2, 2.1, 0.4, 1}, inputGrad[4] {};
    Tensor inputTensor(inputData, inputGrad, {1, 4});
    double w1Data[8] = {1, 2, 3, 4, 5, 6, 7, 8}, w1Grad[8] {};
    Tensor w1(w1Data, w1Grad, {4, 2});
    double w2Data[2] = {5, 7}, w2Grad[2] {};
    Tensor w2(w2Data, w2Grad, {2, 1});
    double p1Data[2], p1Grad[2] {};
    TensorMatMulProduct p1(p1Data, p1Grad, inputTensor, w1);
    double p2Data[1], p2Grad[1] {};
    TensorMatMulProduct p2(p2Data, p2Grad, p1, w2);

    p1.matMul2dIntoSelf(inputTensor, w1);
    p2.matMul2dIntoSelf(p1, w2);
    p2.print();

    p2.setAllGradsTo(1);
    p2.backwardFurther();
    p1.backwardFurther(); 
    w1.printGrad(); 

} 