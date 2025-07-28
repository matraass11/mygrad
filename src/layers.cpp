#include <iostream>
#include "mygrad/layers.hpp"

static const size_t defaultBatchSize = 32;

Layer::Layer(const std::vector<size_t> outDimensions) : outputTensor(outDimensions) {}

void Layer::setInputTensorPointer( Tensor* inputTensor) {
    this->currentInputTensor = inputTensor;
}

void Layer::adjustOutTensorDimensions( const std::vector<size_t>& newDimensions ) {
    outputTensor = Tensor( newDimensions );
}


LinearLayer::LinearLayer( size_t inFeatures, size_t outFeatures,
                          const std::vector<dtype>& data ) : //this is for testing
    Layer( {defaultBatchSize, outFeatures} ),

    weights( data, { inFeatures, outFeatures } ), 
    biases( std::vector<dtype>(outFeatures, 0), {1, outFeatures} ) {}
    

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
    if (!(currentInputTensor)) { 
        std::cerr << "backward before forward impossible. exiting\n";
        exit(1);
    }

    for (int row=0; row < outputTensor.dimensions[0]; row++) {
        for (int column=0; column < outputTensor.dimensions[1]; column++) {
            dtype currentGradPassedDown = outputTensor.gradAt({row, column});
            
            for (int dotProductIterator=0; dotProductIterator < currentInputTensor->dimensions[1]; dotProductIterator++) {
                currentInputTensor->gradAt({row, dotProductIterator}) +=
                    weights.at({dotProductIterator, column}) * currentGradPassedDown;

                weights.gradAt({dotProductIterator, column}) += 
                    currentInputTensor->at({row, dotProductIterator}) * currentGradPassedDown; 
                }

            biases.gradAt({0, column}) += currentGradPassedDown;
        }
    }
}

void LinearLayer::checkDimensions(const Tensor& inputTensor) {
    if (
        inputTensor.dimensions.size() != 2
    ) {
        std::cout << inputTensor.dimensions;
        throw std::runtime_error("dimensionality for linear layer input tensor must be 2. dimensions received are printed above");
    }

    if (
        inputTensor.dimensions[1] != weights.dimensions[0]
    ) {
        std::cout << inputTensor.dimensions[1] << " != " << weights.dimensions[0] << "\n";
        throw std::runtime_error("input tensor columns don't match weight tensor rows in linear layer. mismatch printed above");
    }

    if (
        inputTensor.dimensions[0] != outputTensor.dimensions[0]
    ) {
        adjustOutTensorDimensions( {inputTensor.dimensions[0], weights.dimensions[1]} );
    }
}

void LinearLayer::matmulWithBias() {

    for (int row=0; row < outputTensor.dimensions[0]; row++) {
        for (int column=0; column < outputTensor.dimensions[1]; column++) {
            double& currentElement = outputTensor.at({row, column}) = biases.at({0, column});

            for (int dotProductIterator=0; dotProductIterator < currentInputTensor->dimensions[1]; dotProductIterator++) {
                currentElement +=
                    currentInputTensor->at({row, dotProductIterator}) * weights.at({dotProductIterator, column});
            }
        }
    }
}

void ReLU::forward( Tensor& inputTensor ) {
    checkDimensions( inputTensor );
    setInputTensorPointer( &inputTensor );

    for (int i = 0; i < currentInputTensor->length; i++) {
        outputTensor.data[i] = (currentInputTensor->data[i] >= 0 ? currentInputTensor->data[i] : 0);
    }
}

void ReLU::checkDimensions( const Tensor& inputTensor ) {
    if (inputTensor.dimensions != outputTensor.dimensions ) {
        adjustOutTensorDimensions(inputTensor.dimensions);
    } 
}

void ReLU::backward() {
    for (int i = 0; i < currentInputTensor->length; i++) {
        currentInputTensor->grads[i] = (currentInputTensor->data[i] >= 0 ? outputTensor.grads[i] : 0);
    }

    setInputTensorPointer(nullptr);
}