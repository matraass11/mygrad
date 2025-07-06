#include <iostream>
#include "layers.hpp"

LinearLayer::LinearLayer( int inFeatures, int outFeatures,
                          const std::vector<dtype>& data ) : //this is for testing
    weights( data, { inFeatures, outFeatures } ),
    biases( {}, {1, outFeatures} ) {}

void LinearLayer::forward( Tensor& inputTensor, Tensor& outputTensor ) {
    
    checkDimensions(inputTensor, outputTensor); // we'll remove that later if the model guarantees that the dimensions are correct
    setTensorPointers( &inputTensor, &outputTensor );
    matmulWithBias();
}

void LinearLayer::backward() {
    matmulWithBias_backward();
    setTensorPointers( nullptr, nullptr );
}

void LinearLayer::matmulWithBias_backward() {
    if (!(inputTensor and outputTensor)) { 
        std::cerr << "backward before forward impossible. exiting\n";
        exit(1);
    }

    for (int row=0; row < outputTensor->dimensions[0]; row++) {
        for (int column=0; column < outputTensor->dimensions[1]; column++) {
            double& currentGradPassedDown = outputTensor->gradAt({row, column});
            
            for (int dotProductIterator=0; dotProductIterator < inputTensor->dimensions[1]; dotProductIterator++) {
                inputTensor->gradAt({row, dotProductIterator}) +=
                    weights.at({dotProductIterator, column}) * currentGradPassedDown; // this line under question

                weights.gradAt({dotProductIterator, column}) += 
                    inputTensor->at({row, dotProductIterator}) * currentGradPassedDown; 

                biases.gradAt({0, column}) += currentGradPassedDown;
            }
        }
    }
}

void LinearLayer::checkDimensions(Tensor& inputTensor, Tensor& outputTensor) {
    if (
        inputTensor.dimensions.size() != 2 or
        outputTensor.dimensions.size() != 2
    ) {
        std::cerr << "dimensionality must be 2, exiting\n";
        exit(1);
    }

    if (
        inputTensor.dimensions[1] != weights.dimensions[0] or
        outputTensor.dimensions[0] != inputTensor.dimensions[0] or
        outputTensor.dimensions[1] != weights.dimensions[1]
    ) {
        std::cerr << "wrong dimensions, exiting\n";
        exit(1);
    }
}

inline void LinearLayer::setTensorPointers( Tensor* inputTensor, Tensor* outputTensor) {
    this->inputTensor = inputTensor, this->outputTensor = outputTensor;
}

void LinearLayer::matmulWithBias() {

    for (int row=0; row < outputTensor->dimensions[0]; row++) {
        for (int column=0; column < outputTensor->dimensions[1]; column++) {
            double& currentElement = outputTensor->at({row, column}) = biases.at({0, column});

            for (int dotProductIterator=0; dotProductIterator < inputTensor->dimensions[1]; dotProductIterator++) {
                currentElement +=
                    inputTensor->at({row, dotProductIterator}) * weights.at({dotProductIterator, column});
            }
        }
    }
}