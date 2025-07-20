#pragma once

#include <iostream>
#include "tensor.hpp"
#include "layers.hpp"
#include "helper.hpp"

#define INPUT_SIZE 4
#define NEURONS_N 5

class Model {
    LinearLayer l1 = LinearLayer( INPUT_SIZE, NEURONS_N, std::vector<dtype>(INPUT_SIZE*NEURONS_N, 1) );
    ReLU rl1 = ReLU();
    LinearLayer l2 = LinearLayer( NEURONS_N, NEURONS_N, std::vector<dtype>(NEURONS_N*NEURONS_N, 1) );
    ReLU rl2 = ReLU();
    LinearLayer l3 = LinearLayer( NEURONS_N, 1, std::vector<dtype>(NEURONS_N, 1) );

    std::vector<Tensor*> parameters {
        &l1.weights, &l1.biases,
        &l2.weights, &l2.biases,
        &l3.weights, &l3.biases
    };

    // ReLU relu1()
    // LinearLayer l2( {NEURONS_N, NEURONS_N} )
    // ReLU relu2()
    // LinearLayer l3( {NEURONS_N, 10} )
    // Softmax sm()
    // CrossEntropyLoss loss()


public:
    Model();

    void save(const std::string& filename) const;
    void load(const std::string& filename);
    void zeroGrad();
    Tensor& forward(Tensor& x);
    void backward();

    void print() const;
    void printGrads() const;
};