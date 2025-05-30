#include <iostream>
#include <array>
#include "tensor.hpp"

int main(){
    Tensor w1({1, 2, 3, 4, 5, 6, 7, 8}, {4, 2});
    Tensor w2({5, 5, 6, 6, 7, 7}, {2, 3});
    TensorMatMulProduct p(w1, w2);
    p.matMul2dIntoSelf(w1, w2);
    p.print();
} 