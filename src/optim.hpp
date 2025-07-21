#pragma once

#include "tensor.hpp"
#include "model.hpp"

class Adam {
 
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

public:

    Adam(const std::vector<Tensor*>& parameters, dtype learningRate = 0.001, 
         dtype beta1 = 0.9, dtype beta2 = 0.999, 
         dtype epsilon = std::exp(-8), dtype weightDecay = 0) :

            learningRate(learningRate), beta1(beta1), beta2(beta2), 
            epsilon(epsilon), weightDecay(weightDecay), paramsAndGrads() {

                paramsAndGrads.reserve(sizeOfParamsAndGradsFromParameters(parameters));
                for (const Tensor* const param : parameters) {
                    for (int i = 0; i < param->length; i++) {
                        paramsAndGrads.push_back( dataAndGrads( param->data[i], param->grads[i] ) );
                    }
                }
            }



    void step() {
        stepsMade++;
        for (int i = 0; i < paramsAndGrads.size(); i++) {
            dtype& data = paramsAndGrads[i].data, &grad = paramsAndGrads[i].grad;
            dtype& gradRunAvg = paramsAndGrads[i].gradRunAvg, &gradSqRunAvg = paramsAndGrads[i].gradSqRunAvg;
            grad += weightDecay*data;
            gradRunAvg = ( beta1*gradRunAvg + (1 - beta1)*grad );
            gradSqRunAvg = ( beta2*gradSqRunAvg + (1 - beta2)*grad*grad );

            dtype gradRunAvgCorrected = gradRunAvg / (1 - std::pow(beta1, stepsMade));
            dtype gradSqRunAvgCorrected = gradSqRunAvg / (1 - std::pow(beta2, stepsMade));

            data -= learningRate*gradRunAvgCorrected / (std::sqrt(gradSqRunAvgCorrected) + epsilon);
        }
    }
    

private:
    inline size_t sizeOfParamsAndGradsFromParameters(std::vector<Tensor*> parameters) const {
        size_t size = 0;
        for (const Tensor *const param : parameters){
            size += param->length;
        }
        return size;
    }


};