#pragma once

#include <iostream>
#include "tensor.hpp"
#include "layers.hpp"
#include "helper.hpp"

#define INPUT_SIZE 4
#define NEURONS_N 5

class Model {
    LinearLayer l1 = LinearLayer( INPUT_SIZE, NEURONS_N, std::vector<dtype>(INPUT_SIZE*NEURONS_N, 1) );
    LinearLayer l2 = LinearLayer( NEURONS_N, NEURONS_N, std::vector<dtype>(NEURONS_N*NEURONS_N, 1) );
    LinearLayer l3 = LinearLayer( NEURONS_N, 1, std::vector<dtype>(NEURONS_N, 1) );
    Tensor l1out;
    Tensor l2out;
    Tensor l3out;

    size_t currentBatchSize;

    std::array<Tensor *const, 6> parameters {
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

    // Tensor l1out, act1, l2out, act2, l3out, probs, loss


public:
    Model(size_t batchSize = 32);

    void save(const std::string& filename);
    void load(const std::string& filename);
    void zeroGrad();
    Tensor& forward(Tensor& x);
    void backward();

    void changeBatchSize(size_t newBatchSize);
    void print();
    void printGrads();
};