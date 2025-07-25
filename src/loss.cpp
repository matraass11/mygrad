#include <iostream>

#include "loss.hpp"
#include "helper.hpp"

void CrossEntropyLoss::setInputPointers( Tensor* logits, const Tensor* labels ) {
    this->logits = logits, this->labels = labels;
}

void CrossEntropyLoss::checkDimensions( Tensor& logits, const Tensor& labels ) {
    if (logits.dimensions[0] != labels.dimensions[0]) {
        std::cerr << logits.dimensions[0] << " != " << labels.dimensions[0] << "\n";
        throw std::runtime_error("logits' first dimension must be the same as labels' first dimension for cross entropy loss");
    }
}


dtype CrossEntropyLoss::operator()( Tensor& logits, const Tensor& labels ) { 
    checkDimensions( logits, labels );
    setInputPointers( &logits, &labels );

    Tensor logSoftmaxedLogits = logits.substractColumn(logits.max(1).addColumn(logits.substractColumn(logits.max(1)).exp().sum(1).log()));
    currentSoftmaxOutput = logSoftmaxedLogits.exp();

    dtype loss = 0;
    size_t elementsInBatch = logits.dimensions[0];
    for (int i=0; i < elementsInBatch; i++) {
        loss -= logSoftmaxedLogits.at({ i, static_cast<int>(labels.data[i]) });
    }
    return loss/static_cast<dtype>(elementsInBatch);
} 

void CrossEntropyLoss::backward() {
    
    for (int i=0; i < labels->length; i++) {
        currentSoftmaxOutput.at({ i, static_cast<int>(labels->data[i]) }) -= 1; // substract the one hot encoded vector of labels
    }

    for (int i=0; i < logits->length; i++) {
        logits->grads[i] += currentSoftmaxOutput.data[i] / static_cast<dtype>(logits->dimensions[0]);
    }
}
