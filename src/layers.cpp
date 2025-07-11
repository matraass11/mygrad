#include <iostream>
#include "layers.hpp"

LinearLayer::LinearLayer( size_t inFeatures, size_t outFeatures,
                          const std::vector<dtype>& data ) : //this is for testing

    weights( data, { inFeatures, outFeatures } ), 
    biases( std::vector<dtype>(outFeatures, 0), {1, outFeatures} ),

    outputTensor( {default_batch_size, outFeatures} ) {}
    

LinearLayer::LinearLayer( size_t inFeatures, size_t outFeatures) : //default init
    LinearLayer( inFeatures, outFeatures,
                 KaimingWeightsVector(inFeatures, outFeatures) ) {}


void LinearLayer::forward( Tensor& inputTensor ) {
    
    checkDimensions(inputTensor); // we'll remove that later if the model guarantees that the dimensions are correct
    setInputTensorPointer( &inputTensor );
    matmulWithBias();
}

void LinearLayer::backward() {
    matmulWithBias_backward();
    setInputTensorPointer( nullptr );
}

void LinearLayer::matmulWithBias_backward() {
    if (!(inputTensor)) { 
        std::cerr << "backward before forward impossible. exiting\n";
        exit(1);
    }

    for (int row=0; row < outputTensor.dimensions[0]; row++) {
        for (int column=0; column < outputTensor.dimensions[1]; column++) {
            double& currentGradPassedDown = outputTensor.gradAt({row, column});
            
            for (int dotProductIterator=0; dotProductIterator < inputTensor->dimensions[1]; dotProductIterator++) {
                inputTensor->gradAt({row, dotProductIterator}) +=
                    weights.at({dotProductIterator, column}) * currentGradPassedDown; // this line under question

                weights.gradAt({dotProductIterator, column}) += 
                    inputTensor->at({row, dotProductIterator}) * currentGradPassedDown; 
                }

            biases.gradAt({0, column}) += currentGradPassedDown;
        }
    }
}

void LinearLayer::checkDimensions(Tensor& inputTensor) {
    if (
        inputTensor.dimensions.size() != 2
    ) {
        std::cerr << "dimensionality must be 2, exiting\n";
        exit(1);
    }

    if (
        inputTensor.dimensions[1] != weights.dimensions[0]
    ) {
        std::cerr << "wrong dimensions, exiting\n";
        exit(1);
    }

    if (
        inputTensor.dimensions[0] != outputTensor.dimensions[0]
    ) {
        adjustOutTensorDimensions(inputTensor.dimensions[0]);
    }
}

inline void LinearLayer::adjustOutTensorDimensions(size_t newRowsN) {
    outputTensor = Tensor( {newRowsN, weights.dimensions[1]});
}

inline void LinearLayer::setInputTensorPointer( Tensor* inputTensor) {
    this->inputTensor = inputTensor;
}

void LinearLayer::matmulWithBias() {

    for (int row=0; row < outputTensor.dimensions[0]; row++) {
        for (int column=0; column < outputTensor.dimensions[1]; column++) {
            double& currentElement = outputTensor.at({row, column}) = biases.at({0, column});

            for (int dotProductIterator=0; dotProductIterator < inputTensor->dimensions[1]; dotProductIterator++) {
                currentElement +=
                    inputTensor->at({row, dotProductIterator}) * weights.at({dotProductIterator, column});
            }
        }
    }
}