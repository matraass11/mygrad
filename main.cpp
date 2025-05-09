#include <iostream>
#include "tensor.h"

#include <array>

int main(){
    std::vector<int> dimensions{20, 3};
    // Tensor weights = Tensor(dimensions, {1, 2, 3, 4, 5, 6});
    // Tensor other = Tensor({2, 3, 4}, OneValueVector());

    // Tensor normal = normDistTensor(dimensions);

    Tensor initByVectors(std::vector<int> {1, 3}, std::vector<double> {1, 2, 3});
    Tensor initByLists({1, 3}, {1, 2, 3});


    std::cout << initByVectors << "\n" << initByLists << std::endl; 


    // std::cout << normal << std::endl;
    // std::cout << weights << std::endl;
}