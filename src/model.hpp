#pragma once

#include <iostream>
#include "tensor.hpp"
#include "layers.hpp"

#define INPUT_SIZE 28*28

class Model {
    LinearLayer l1 = LinearLayer( INPUT_SIZE, 200 );
    Tensor l1out;

    std::array<Tensor*, 2> parameters {
        &l1.weights, &l1.biases,
        // l2.weights, l2.biases,
        // l3.weights, l3.biases
    };

    // ReLU relu1()
    // LinearLayer l2( {200, 200} )
    // ReLU relu2()
    // LinearLayer l3( {200, 10} )
    // Softmax sm()
    // CrossEntropyLoss loss()

    // Tensor l1out, act1, l2out, act2, l3out, probs, loss


    Tensor& forward(Tensor& x) {
        
        l1.forward(x, l1out);
        // x = l1.forward(x); for now not out priority
        
        return l1out;
    };


public:
    Model(size_t initialBatchSize = 32);
    Model(const std::string& filename);

    void save(const std::string& filename);
    void zeroGrad();
    void forward();
    void backward();

    void print();
};