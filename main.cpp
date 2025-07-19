#include <iostream>
#include <array>
#include <fstream>

#include "src/tensor.hpp"
#include "src/model.hpp"
#include "src/loss.hpp"

#define MODELS_DIR "../models/"

int main() {

    CrossEntropyLoss l(3);

    Tensor logits( {1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 3} );
    Tensor labels( {0, 2, 1}, {3, 1} );
    std::cout << l.forward(logits, labels) << "\n";
    l.backward();
    logits.printGrad();
} 
