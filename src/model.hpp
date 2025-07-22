#pragma once

#include <iostream>
#include "tensor.hpp"
#include "layers.hpp"
#include "helper.hpp"

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define NEURONS_N 100

class Model {
    LinearLayer l1 = LinearLayer( INPUT_SIZE, NEURONS_N, KaimingWeightsVector(INPUT_SIZE, NEURONS_N) );
    ReLU rl1 = ReLU();
    LinearLayer l2 = LinearLayer( NEURONS_N, NEURONS_N, KaimingWeightsVector(NEURONS_N, NEURONS_N) );
    ReLU rl2 = ReLU();
    LinearLayer l3 = LinearLayer( NEURONS_N, OUTPUT_SIZE, KaimingWeightsVector(NEURONS_N, OUTPUT_SIZE) );

public:

    std::vector<Tensor*> parameters {
        &l1.weights, &l1.biases,
        &l2.weights, &l2.biases,
        &l3.weights, &l3.biases
    };

    Model();

    void save(const std::string& filename) const;
    void load(const std::string& filename);
    void zeroGrad();
    Tensor& forward(Tensor& x);
    void backward();

    void print() const;
    void printGrads() const;
};