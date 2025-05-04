#include <iostream>
#include "tensor.h"

#include <array>

int main(){
    std::vector<int> dimensions{20, 3};
    // Tensor weights = Tensor(dimensions, {1, 2, 3, 4, 5, 6});
    Tensor other = singleValueTensor(std::vector<int> {2, 3, 4}, 4.6);

    Tensor normal = normDistTensor(dimensions);

    std::cout << normal << std::endl;
    // std::cout << weights << std::endl;
}