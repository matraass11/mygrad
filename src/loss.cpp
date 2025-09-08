#include <iostream>
#include <vector>
#include <stdexcept>
#include <numeric>
#include <functional>
#include <cmath>

#include "mygrad/loss.hpp"
#include "mygrad/helper.hpp"

namespace mygrad {

static const TensorDims defaultDimensions = {64, 10};
// we have to initialize the intermediate tensor with some dimensions, so we pick some arbitrary ones.
// when needed, the dimensions are adjusted

CrossEntropyLoss::CrossEntropyLoss() : currentSoftmaxOutput(Tensor::zeros(defaultDimensions)) {}


void CrossEntropyLoss::checkDimensions( Tensor& logits, const Tensor& labels ) {
    if (logits.dimensions[0] != labels.dimensions[0]) {
        std::cerr << logits.dimensions[0] << " != " << labels.dimensions[0] << "\n";
        throw std::runtime_error("logits' first dimension must be the same as labels' first dimension for cross entropy loss");
    }
    if (currentSoftmaxOutput.dimensions != logits.dimensions) {
        currentSoftmaxOutput = Tensor::zeros(logits.dimensions);
    } 
}


dtype CrossEntropyLoss::operator()( Tensor& logits, const Tensor& labels ) { 
    checkDimensions( logits, labels );
    setInputPointers( &logits, &labels );

    Tensor logSoftmaxedLogits = logits.substractColumn(logits.max(1).addColumn(logits.substractColumn(logits.max(1)).exp().sum(1).log()));
    currentSoftmaxOutput = logSoftmaxedLogits.exp();

    dtype loss = 0;
    size_t elementsInBatch = logits.dimensions[0];
    for (size_t i=0; i < elementsInBatch; i++) {
        loss -= logSoftmaxedLogits.at({ i, static_cast<size_t>(labels.data[i]) });
    }
    return loss/static_cast<dtype>(elementsInBatch);
} 

void CrossEntropyLoss::backward() {
    #ifndef NDEBUG
        if (!(labels) or !(logits)) throw std::runtime_error("backward before forward impossible");
    #endif
    
    for (size_t i=0; i < labels->length; i++) {
        currentSoftmaxOutput.at({ i, static_cast<size_t>(labels->data[i]) }) -= 1; // substract the one hot encoded vector of labels
    }

    for (size_t i=0; i < logits->length; i++) {
        logits->grads[i] += currentSoftmaxOutput.data[i] / static_cast<dtype>(logits->dimensions[0]);
    }

    setInputPointers( nullptr, nullptr );
}

void MSEloss::checkDimensions( Tensor& outputs, const Tensor& labels ) {
    if (outputs.dimensions[0] != labels.dimensions[0]) {
        std::cerr << outputs.dimensions[0] << " != " << labels.dimensions[0] << "\n";
        throw std::runtime_error("outputs' first dimension must be the same as labels' first dimension for cross entropy loss");
    }
}

dtype MSEloss::operator()( Tensor& outputs, const Tensor& labels ) {
    checkDimensions( outputs, labels );
    setInputPointers( &outputs, &labels );

    dtype loss = std::transform_reduce(
        outputs.data.get(), outputs.data.get() + outputs.length, labels.data.get(),
        0.0,
        std::plus(),
        [](dtype output, dtype label) {
            dtype diff = output - label;
            return diff * diff;
        }
    );
    if (reduction == "mean") loss /= outputs.length;
    else if (reduction == "sum") loss /= outputs.dimensions[0];
    return loss;
}

void MSEloss::backward() {

    #ifndef NDEBUG
        if (!(labels) or !(outputs)) throw std::runtime_error("backward before forward impossible");
    #endif
    
    for (size_t i = 0; i < outputs->length; i++) {
        dtype gradient = 2 * (outputs->data[i] - labels->data[i]);
        if (reduction == "mean") gradient /= outputs->length;
        else if (reduction == "sum") gradient /= outputs->dimensions[0];
        outputs->grads[i] += gradient;
    }

    setInputPointers( nullptr, nullptr );
}


dtype KLdivWithStandardNormal::operator()( Tensor& distribution, dtype beta ) {
    setInputPointers( &distribution );
    currentBeta = beta;

    #ifndef NDEBUG
        if (distribution.dimensions[1] % 2 != 0 or distribution.dimensions.size() != 2) 
            throw std::runtime_error("dimensions of distribution should be of size 2 and have n columns for means and n columns for logvariances");
    #endif

    dtype loss = 0;
    for (size_t i = 0; i < distribution.length; i+=2) {
        dtype mean = distribution.data[i], logvar = distribution.data[i + 1];
        loss += 1 + logvar - std::pow(mean, 2) - std::exp(logvar);
    }

    loss /= -2; // the math shorthand part
    loss /= (distribution.dimensions[0]); // the normalization part
    return loss * beta;
}

void KLdivWithStandardNormal::backward() {
    #ifndef NDEBUG
        if (!(distribution) or (!currentBeta)) throw std::runtime_error("backward before forward impossible");
    #endif

    const dtype divisor = -2 * static_cast<dtype>(distribution->dimensions[0]) / currentBeta;
    for (size_t i = 0; i < distribution->length; i+=2) {
        dtype mean = distribution->data[i], logvar = distribution->data[i + 1];
        distribution->grads[i] += (-2 * mean) / divisor;
        distribution->grads[i + 1] += (1 - std::exp(logvar)) / divisor;
    }

    setInputPointers( nullptr );
    currentBeta = 0;
}

} // namespace mygrad