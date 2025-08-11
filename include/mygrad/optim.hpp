#pragma once

#include <vector>
#include <cmath>

#include "tensor.hpp"
#include "model.hpp"

namespace mygrad {

class Adam {
public:

    Adam(const std::vector<Tensor*>& parameters, dtype learningRate = 0.001, 
         dtype beta1 = 0.9, dtype beta2 = 0.999, 
         dtype epsilon = std::exp(-8), dtype weightDecay = 0);


    void step();
    

private:

    dtype learningRate;
    dtype beta1, beta2;
    dtype epsilon;
    dtype weightDecay;
    int stepsMade = 0;


    struct dataAndGrads {
        dtype& data, & grad;
        dtype gradRunAvg, gradSqRunAvg;

        dataAndGrads(dtype& data, dtype& grad) : 
            data(data), grad(grad), 
            gradRunAvg(0), gradSqRunAvg(0) {}
    };

    std::vector<dataAndGrads> paramsAndGrads;

    inline size_t sizeOfParamsAndGradsFromParameters(std::vector<Tensor*> parameters) const {
        size_t size = 0;
        for (const Tensor *const param : parameters){
            size += param->length;
        }
        return size;
    }

};

} // namespace mygrad