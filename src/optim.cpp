#include <iostream>
#include "mygrad/optim.hpp"

namespace mygrad {

Adam::Adam(const std::vector<Tensor*>& parameters, dtype learningRate, 
           dtype beta1, dtype beta2, dtype epsilon, dtype weightDecay) :

        learningRate(learningRate), beta1(beta1), beta2(beta2), 
        epsilon(epsilon), weightDecay(weightDecay), paramsAndGrads() {

            paramsAndGrads.reserve(sizeOfParamsAndGradsFromParameters(parameters));
            for (const Tensor* const param : parameters) {
                for (int i = 0; i < param->length; i++) {
                    paramsAndGrads.push_back( dataAndGrads( param->data[i], param->grads[i] ) );
                }
            }
        }

void Adam::step() {
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

} // namespace mygrad