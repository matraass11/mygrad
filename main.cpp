#include <iostream>
#include <array>
#include <fstream>

#include "src/tensor.hpp"
#include "src/functions.hpp"
#include "src/helper.hpp"
#include "src/model.hpp"

#define MODELS_DIR "../models/"

int main(){
    LinearLayer l1( 4, 2, {1, 2, 3, 4, 5, 4, 3, 2} );
    LinearLayer l2( 2, 1, {5, 6});


    Tensor input({2, 2, 3, 4}, {1, 4});
    Tensor l1out( {}, {1, 2});
    Tensor output( {}, {1, 1});
    l1.forward(input, l1out);
    l1out.print();

    l2.forward(l1out, output);
    output.print();
    output.grads[0] = 1;
    l2.backward();
    l1.backward();
    l1.weights.printGrad();
} 