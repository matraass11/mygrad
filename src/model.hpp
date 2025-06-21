#pragma once

#include <iostream>
#include "tensor.hpp"
#include "tensorStruct.hpp"
#include "functions.hpp"

class Model {
    Tensor_s<10> w1;
    Tensor_s<10> b1;
    Tensor_s<10> w2;
    Tensor_s<10> b2;
    Tensor_s<10> w3;

    std::array<Tensor, 5> weights {
        w1.tensor,
        b1.tensor,
        w2.tensor, 
        b2.tensor,
        w3.tensor
    };

public:
    Model();
    Model(const std::string& filename);

    void save(const std::string& filename);
    void zeroGrad();
    void forward();
    void backward();

    void print();
};

template<size_t batchSize>
struct IntermediateTensors {

    Tensor_s<batchSize*100> pr1; // batchSize x 100
    Tensor_s<batchSize*100> sum1; 
    Tensor_s<batchSize*100> activations1;
};