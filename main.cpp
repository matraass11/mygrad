#include <iostream>
#include "tensor.h"

#include <array>

int main(){
    Tensor weights = Tensor(std::vector<int> {3, 4}, 1);
    std::cout << weights << std::endl;
    // std::cout << weights << std::endl;
}