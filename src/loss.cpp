#include <iostream>
#include <vector>
#include <stdexcept>
#include <numeric>
#include <functional>

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

    return std::transform_reduce(
        outputs.data.get(), outputs.data.get() + outputs.length, labels.data.get(),
        0.0,
        std::plus(),
        [](dtype output, dtype label) {
            dtype diff = output - label;
            return diff * diff;
        }
    ) / outputs.length;
}

void MSEloss::backward() {

    #ifndef NDEBUG
        if (!(labels) or !(outputs)) throw std::runtime_error("backward before forward impossible");
    #endif
    
    for (size_t i = 0; i < outputs->length; i++) {
        outputs->grads[i] += 2 * (outputs->data[i] - labels->data[i]);
    }

    setInputPointers( nullptr, nullptr );
}

} // namespace mygrad