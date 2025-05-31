#include <iostream>
#include <array>
#include "tensor.hpp"

int main(){
    double w1Data[8] = {1, 2, 3, 4, 5, 6, 7, 8}, w1Grad[8];
    Tensor w1(w1Data, w1Grad, {4, 2});
    double w2Data[8] = {5, 5, 6, 6, 7, 7}, w2Grad[8];
    Tensor w2(w2Data, w2Grad, {2, 3});
    double pData[8], pGrad[8];
    TensorMatMulProduct p(pData, pGrad, w1, w2);
    p.matMul2dIntoSelf(w1, w2);
    p.print();
} 