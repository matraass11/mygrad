#include <iostream>
#include <array>
#include "tensor.hpp"

int main(){
    Tensor t({1, 2, 3, 4}, {2, 2}, 4);
    t.print(); 
    // std::cout << "hello\n";
} 