#include <iostream>
#include <stdexcept>

#include "mygrad/layers.hpp"

namespace mygrad {

static const std::vector<size_t> defaultDimensions = {32, 100}; 
// we have to initialize the output tensor with some dimensions, so we pick some arbitrary ones.
// when needed, the dimensions are adjusted

Layer::Layer() : outputTensor(Tensor::zeros(defaultDimensions)) {}

void Layer::setInputTensorPointer( Tensor* inputTensor) {
    this->currentInputTensor = inputTensor;
}

void Layer::adjustOutTensorDimensions( const std::vector<size_t>& newDimensions ) {
    outputTensor = Tensor::zeros( newDimensions );
}


LinearLayer::LinearLayer( size_t inFeatures, size_t outFeatures,
                          const std::vector<dtype>& data ) : //this is for testing
    weights( data, { inFeatures, outFeatures } ), 
    biases( std::vector<dtype>(outFeatures, 0), {1, outFeatures} ) {}
    

LinearLayer::LinearLayer( size_t inFeatures, size_t outFeatures) : //default init
    LinearLayer( inFeatures, outFeatures,
                 KaimingWeightsVector(inFeatures, outFeatures) ) {}


void LinearLayer::forward( Tensor& inputTensor ) {
    
    manageDimensions(inputTensor); 
    setInputTensorPointer( &inputTensor );
    matmulWithBias();
}

void LinearLayer::backward() {
    matmulWithBiasBackward();
    setInputTensorPointer( nullptr );
}

void LinearLayer::matmulWithBiasBackward() {
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

void LinearLayer::manageDimensions(const Tensor& inputTensor) {
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
        inputTensor.dimensions[0] != outputTensor.dimensions[0] or weights.dimensions[1] != outputTensor.dimensions[1]
    ) {
        adjustOutTensorDimensions( {inputTensor.dimensions[0], weights.dimensions[1]} );
    }
}

void LinearLayer::matmulWithBias() {

    for (int row=0; row < outputTensor.dimensions[0]; row++) {
        for (int column=0; column < outputTensor.dimensions[1]; column++) {
            dtype& currentElement = outputTensor.at({row, column}) = biases.at({0, column});

            for (int dotProductIterator=0; dotProductIterator < currentInputTensor->dimensions[1]; dotProductIterator++) {
                currentElement +=
                    currentInputTensor->at({row, dotProductIterator}) * weights.at({dotProductIterator, column});
            }
        }
    }
}

void ReLU::forward( Tensor& inputTensor ) {
    manageDimensions( inputTensor );
    setInputTensorPointer( &inputTensor );

    for (int i = 0; i < currentInputTensor->length; i++) {
        outputTensor.data[i] = (currentInputTensor->data[i] >= 0 ? currentInputTensor->data[i] : 0);
    }
}

void ReLU::manageDimensions( const Tensor& inputTensor ) {
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

Conv2d::Conv2d( size_t inChannels, size_t outChannels, size_t kernelSize, size_t stride, size_t paddingSize ) : 
        inChannels(inChannels), outChannels(outChannels), kernelSize(kernelSize), stride(stride), paddingSize(paddingSize),
        kernels( KaimingWeightsVector(kernelSize*kernelSize*inChannels, outChannels),
                 { outChannels, inChannels, kernelSize, kernelSize } ),
        biases( std::vector<dtype>(outChannels, 0), {outChannels} ) {}


dtype Conv2d::convolve(size_t pictureIndex, size_t filterIndex, size_t inputRow, size_t inputCol) {
    // filter = kernels[filterIndex]
    dtype sum = 0;

    for (size_t inputChannel = 0; inputChannel < inChannels; inputChannel++) {
        for (size_t kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
            for (size_t kernelCol = 0; kernelCol < kernelSize; kernelCol++) {
                sum +=  inputTensor.at( {pictureIndex, inputChannel, inputRow + kernelRow, inputCol + kernelCol} ) * 
                        kernels.at( {filterIndex, inputChannel, kernelRow, kernelCol} );
            }
        }
    }

    return sum;
}


void Conv2d::forward(Tensor& inputTensor) {
    manageDimensions( inputTensor );
    setInputTensorPointer( &inputTensor );

    size_t locInOutput = 0;

    for (size_t picture = 0; picture < inputTensor.dimensions[0]; picture++) {
        for (size_t channelOut = 0; channelOut < outChannels; channelOut++) {
            for (size_t row = -paddingSize; row < inputTensor.dimensions[2] + paddingSize - kernelSize + 1; row += stride) {
                for (size_t column = -paddingSize; column < inputTensor.dimensions[3] + paddingSize - kernelSize + 1; column += stride) {
                
                    outputTensor.data[locInOutput] = convolve(picture, channelOut, row, column);
                    locInOutput++;
                }
            }
        }
    }
}

void Conv2d::backward() {

}


void Conv2d::manageDimensions( const Tensor& inputTensor ) {
    if (inputTensor.dimensions.size() != 4) throw std::runtime_error("input tensor dimensionality must be four for Conv2d");

    auto convolvedSize = [this]( size_t size) {
        return (size + 2*paddingSize - kernelSize)/stride + 1;
    };

    std::vector<size_t> neededOutTensorDims = { inputTensor.dimensions[0],
                                                outChannels,
                                                convolvedSize(inputTensor.dimensions[2]),
                                                convolvedSize(inputTensor.dimensions[3]) };
    if (outputTensor.dimensions != neededOutTensorDims) {
        adjustOutTensorDimensions(neededOutTensorDims);
    }
}


} // namespace mygrad