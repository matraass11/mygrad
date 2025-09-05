#include <iostream>
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <execution> 

#include <cassert> 

#include "mygrad/layers.hpp"
#include "mygrad/threadPool.hpp"

namespace mygrad {

static const TensorDims defaultDimensions = {32, 100}; 
// we have to initialize the output tensor with some dimensions, so we pick some arbitrary ones.
// when needed, the dimensions are adjusted

Layer::Layer() : outputTensor(Tensor::zeros(defaultDimensions)) {}

void Layer::setInputTensorPointer( Tensor* inputTensor) {
    this->currentInputTensor = inputTensor;
}

void Layer::adjustOutTensorDimensions( const TensorDims& newDimensions ) {
    outputTensor = Tensor::zeros( newDimensions );
}

void Layer::zeroGrad() {
    for (Tensor* t : parameterTensors()) {
        t->zeroGrad();
    }
    for (Tensor* t : nonParameterTensors()) {
        t->zeroGrad();
    }
}

void ReLU::forward( Tensor& inputTensor ) {
    manageDimensions( inputTensor );
    setInputTensorPointer( &inputTensor );

    for (size_t i = 0; i < currentInputTensor->length; i++) {
        outputTensor.data[i] = (currentInputTensor->data[i] >= 0 ? currentInputTensor->data[i] : 0);
    }
}

void ReLU::manageDimensions( const Tensor& inputTensor ) {
    if (inputTensor.dimensions != outputTensor.dimensions ) {
        adjustOutTensorDimensions(inputTensor.dimensions);
    } 
}

void ReLU::backward() {
    #ifndef NDEBUG
        if (!(currentInputTensor)) throw std::runtime_error("backward before forward impossible");
    #endif

    for (size_t i = 0; i < currentInputTensor->length; i++) {
        currentInputTensor->grads[i] += (currentInputTensor->data[i] >= 0 ? outputTensor.grads[i] : 0);
    }

    setInputTensorPointer(nullptr);
}

void Sigmoid::manageDimensions( const Tensor& inputTensor ) {
    if (inputTensor.dimensions != outputTensor.dimensions ) {
        adjustOutTensorDimensions(inputTensor.dimensions);
    } 
}


void Sigmoid::forward( Tensor& inputTensor ) {
    manageDimensions( inputTensor );
    setInputTensorPointer( &inputTensor );

    for (size_t i = 0; i < inputTensor.length; i++) {
        outputTensor.data[i] = 1 / (1 + std::exp( -inputTensor.data[i]) );
    }
}

void Sigmoid::backward() {
    #ifndef NDEBUG
        if (!(currentInputTensor)) throw std::runtime_error("backward before forward impossible");
    #endif

    Tensor& inputTensor = *currentInputTensor;

    for (size_t i = 0; i < inputTensor.length; i++) {
        dtype negexp = std::exp(-inputTensor.data[i]);
        inputTensor.grads[i] += (negexp / ((1 + negexp) * (1 + negexp)) ) * outputTensor.grads[i];
    }

    setInputTensorPointer(nullptr);
}


void Upsample::manageDimensions( const Tensor& inputTensor ) {

    if (inputTensor.dimensions.size() != 4) {
        std::cerr << inputTensor.dimensions;
        throw std::runtime_error("Upsample expects a 4d tensor, dimensions received are printed above");
    }

    TensorDims neededOutDims = inputTensor.dimensions;
    neededOutDims[2] *= scalingFactor, neededOutDims[3] *= scalingFactor; 

    if (outputTensor.dimensions != neededOutDims) adjustOutTensorDimensions(neededOutDims);
}


void Upsample::forward( Tensor& inputTensor ) {
    manageDimensions(inputTensor);
    setInputTensorPointer(&inputTensor);

    for (size_t image = 0; image < inputTensor.dimensions[0]; image++) {
        for (size_t channel = 0; channel < inputTensor.dimensions[1]; channel++) {
            for (size_t row = 0; row < outputTensor.dimensions[2]; row++) {
                for (size_t col = 0; col < outputTensor.dimensions[3]; col++) {
                    outputTensor.at({image, channel, row, col}) = 
                        inputTensor.at({image, channel, row / scalingFactor, col / scalingFactor});
                }
            }
        }
    }
    // outputTensor.data[i] = inputTensor.data[i / scalingFactor];
}


void Upsample::backward() {
        
    #ifndef NDEBUG
        if (!(currentInputTensor)) throw std::runtime_error("backward before forward impossible");
    #endif

    for (size_t image = 0; image < currentInputTensor->dimensions[0]; image++) {
        for (size_t channel = 0; channel < currentInputTensor->dimensions[1]; channel++) {
            for (size_t row = 0; row < outputTensor.dimensions[2]; row++) {
                for (size_t col = 0; col < outputTensor.dimensions[3]; col++) {
                    currentInputTensor->gradAt({image, channel, row / scalingFactor, col / scalingFactor}) 
                        += outputTensor.gradAt({image, channel, row, col});
                }
            }
        }
    }

    setInputTensorPointer(nullptr);
}


Reshape::Reshape( const TensorDims& newDimensions, std::optional<size_t> freeDimension ) : 
    newDimensions(newDimensions), freeDimension(freeDimension)
    { adjustOutTensorDimensions(newDimensions); }


void Reshape::forward( Tensor& inputTensor ) {
    manageDimensions( inputTensor );
    setInputTensorPointer( &inputTensor );

    for (size_t i = 0; i < inputTensor.length; i++) {
        outputTensor.data[i] = inputTensor.data[i];
    }
}

void Reshape::backward() {
    
    #ifndef NDEBUG
        if (!(currentInputTensor)) throw std::runtime_error("backward before forward impossible");
    #endif

    for (size_t i = 0; i < currentInputTensor->length; i++) {
        currentInputTensor->grads[i] += outputTensor.grads[i];
    }

    setInputTensorPointer(nullptr);
}

void Reshape::manageDimensions( const Tensor& inputTensor ) {
    if (inputTensor.length != outputTensor.length) {
        if (freeDimension.has_value()) {
            size_t outLengthWithoutFreeDim = 1;
            for (size_t i = 0; i < newDimensions.size(); i++) {
                if (i != freeDimension) {
                    outLengthWithoutFreeDim *= newDimensions[i];
                }
            }
            if (inputTensor.length % outLengthWithoutFreeDim != 0) throw std::runtime_error("dimensions for reshape with free dimension are invalid");

            newDimensions[freeDimension.value()] = inputTensor.length / outLengthWithoutFreeDim;


            // std::cout << "new dims: " << newDimensions;
            // std::cout << "input dims: " << inputTensor.dimensions;

            outputTensor = Tensor::zeros(newDimensions);

            assert(inputTensor.length == outputTensor.length);

        }   

        else throw std::runtime_error("reshape can't be done when the length of the input and the output are different");
    }
}


dtype MaxPool2d::pool(size_t pictureIndex, size_t channelIndex, size_t inputRow, size_t inputCol) const {
    dtype max = currentInputTensor->at( {pictureIndex, channelIndex, inputRow, inputCol} );
    for (size_t row = inputRow; row < std::min(inputRow + kernelSize, currentInputTensor->dimensions[2]); row++) {
        for (size_t col = inputCol; col < std::min(inputCol + kernelSize, currentInputTensor->dimensions[3]); col++) {
            dtype value = currentInputTensor->at({pictureIndex, channelIndex, row, col});
            if (value > max) {
                max = value;
            }
        }
    }
    return max;
}


void MaxPool2d::forward( Tensor& inputTensor ) {
    manageDimensions(inputTensor);
    setInputTensorPointer(&inputTensor);

    size_t locInOutput = 0;

    for (size_t picture = 0; picture < inputTensor.dimensions[0]; picture++) {
        for (size_t channelOut = 0; channelOut < inputTensor.dimensions[1]; channelOut++) {
            for (size_t row = 0; row < inputTensor.dimensions[2] - kernelSize + 1; row += kernelSize) {
                for (size_t col = 0; col < inputTensor.dimensions[3] - kernelSize + 1; col += kernelSize) {
                
                    outputTensor.data[locInOutput] = pool(picture, channelOut, row, col);
                    locInOutput++;
                }
            }
        }
    }
}

void MaxPool2d::poolBackward(size_t pictureIndex, size_t channelIndex, size_t inputRow, size_t inputCol, dtype gradPassedDown) {
    TensorIndices maxIndices = {pictureIndex, channelIndex, inputRow, inputCol};
    dtype maxValue = currentInputTensor->at({pictureIndex, channelIndex, inputRow, inputCol});
    for (size_t row = inputRow; row < std::min(inputRow + kernelSize, currentInputTensor->dimensions[2]); row++) {
        for (size_t col = inputCol; col < std::min(inputCol + kernelSize, currentInputTensor->dimensions[3]); col++) {
            dtype value = currentInputTensor->at({pictureIndex, channelIndex, row, col});
            if (value > maxValue) {
                maxValue = value;
                maxIndices = {pictureIndex, channelIndex, row, col};
            }
        }
    }
    currentInputTensor->gradAt(maxIndices) += gradPassedDown;
}


void MaxPool2d::backward() {
    if (!(currentInputTensor)) throw std::runtime_error("backward before forward impossible");

    size_t locInOutput = 0;

    for (size_t picture = 0; picture < currentInputTensor->dimensions[0]; picture++) {
        for (size_t channelOut = 0; channelOut < currentInputTensor->dimensions[1]; channelOut++) {
            for (size_t row = 0; row < currentInputTensor->dimensions[2] - kernelSize + 1; row += kernelSize) {
                for (size_t col = 0; col < currentInputTensor->dimensions[3] - kernelSize + 1; col += kernelSize) {
                
                    poolBackward(picture, channelOut, row, col, outputTensor.grads[locInOutput]);
                    locInOutput++;
                }
            }
        }
    }

    setInputTensorPointer(nullptr);
}

void MaxPool2d::manageDimensions( const Tensor& inputTensor ) {
    if (inputTensor.dimensions.size() != 4) throw std::runtime_error("input tensor dimensionality must be four for MaxPool2d");

    TensorDims neededOutTensorDims = { inputTensor.dimensions[0],
                                       inputTensor.dimensions[1],
                                       inputTensor.dimensions[2] / kernelSize,
                                       inputTensor.dimensions[3] / kernelSize };

    if (outputTensor.dimensions != neededOutTensorDims) {
        adjustOutTensorDimensions(neededOutTensorDims);
    }
}


void Reparameterize::manageDimensions( const Tensor& inputTensor ) {
    #ifndef NDEBUG
        if (inputTensor.dimensions[1] % 2 != 0 or inputTensor.dimensions.size() != 2) 
            throw std::runtime_error("dimensions of distribution should be of size 2 and have n columns for means and n columns for logvariance");
    #endif

    TensorDims neededOutDims = {inputTensor.dimensions[0], inputTensor.dimensions[1] / 2};
    if (outputTensor.dimensions != neededOutDims) {
        adjustOutTensorDimensions(neededOutDims);
        currentEpsilons = Tensor::zeros(neededOutDims);
    }
}

void Reparameterize::forward( Tensor& inputTensor ) {
     
    manageDimensions( inputTensor );
    setInputTensorPointer( &inputTensor );

    for (size_t i = 0; i < inputTensor.length; i += 2) {
        dtype mean = inputTensor.data[i], std = std::exp(inputTensor.data[i + 1] / 2);

        dtype epsilon = normDist(generator);
        currentEpsilons.data[i / 2] = epsilon;
        outputTensor.data[i / 2] = mean + epsilon * std;
    }
}

void Reparameterize::backward() {
    if (!(currentInputTensor)) throw std::runtime_error("backward before forward impossible");

    Tensor& inputTensor = *currentInputTensor;

    for (size_t i = 0; i < inputTensor.length; i += 2) {
        dtype gradPassedDown = outputTensor.grads[i / 2];

        // std::cout << "gradPassedDown in reparam at i/2 = " << i/2 << ": " << outputTensor.grads[i / 2] << "\n";

        inputTensor.grads[i] += 1 * gradPassedDown;
        dtype std = std::exp(inputTensor.data[i + 1] / 2); 
        inputTensor.grads[i + 1] += 0.5 * std * currentEpsilons.data[i / 2] * gradPassedDown;
    }
    
    setInputTensorPointer( nullptr );
}

} // namespace mygrad