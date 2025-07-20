#include <iostream>
#include <array>
#include <fstream>

#include "src/tensor.hpp"
#include "src/model.hpp"
#include "src/loss.hpp"
#include "src/optim.hpp"

#define MODELS_DIR "../models/"

int main() {


    
    Tensor logits( {1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 3} );
    Tensor labels( {0, 2, 1}, {3, 1} );
    
    CrossEntropyLoss loss(3);
    Adam optim({&logits});


    for (int i = 0; i < 10; i++) {
        std::memset(logits.grads.get(), 0, logits.length * sizeof(dtype));
        dtype l = loss.forward(logits, labels);
        loss.backward();
        std::cout << "loss at " << i << ": " << l << "\n";
        logits.printGrad();
        optim.step();
    }
} 
