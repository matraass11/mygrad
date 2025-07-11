#include <iostream>
#include <array>
#include <fstream>

#include "src/tensor.hpp"
#include "src/functions.hpp"
#include "src/model.hpp"

#define MODELS_DIR "../models/"

int main(){

    Model m;

    m.print();

    Tensor input({2, 2, 3, 4}, {1, 4});
    Tensor& output = m.forward(input);

    output.print();
    m.backward();
    m.printGrads();
} 