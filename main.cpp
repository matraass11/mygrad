#include <iostream>
#include <array>
#include "tensor.hpp"

int main(){
    TensorBase<8, 3> t({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2});
    t.print();
}