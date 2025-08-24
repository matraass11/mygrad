#include <iostream>
#include <stdexcept>
#include <cmath>

#include <future>
#include <thread>

#include <numeric>
#include <execution>


#include <cassert>

#include "mygrad/layers.hpp"

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


LinearLayer::LinearLayer( size_t inFeatures, size_t outFeatures,
                          const std::vector<dtype>& data ) :
    weights( data, { outFeatures, inFeatures } ), // the tensor is transposed for matrix multiplication to work nicely.
                                                 // each outFeatures row has InFeatures weights. 
    biases( std::vector<dtype>(outFeatures, 0), {1, outFeatures} ) {}
    

LinearLayer::LinearLayer( size_t inFeatures, size_t outFeatures) : // default init
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
    if (!(currentInputTensor)) throw std::runtime_error("backward before forward impossible");

    auto processRow = [&] (size_t rowStart, size_t rowEnd) {
        const size_t inpTensorColumns = currentInputTensor->dimensions[1];
        const size_t weightColumns = weights.dimensions[1];

        for (size_t row = rowStart; row < rowEnd; row++){
            for (size_t column=0; column < outputTensor.dimensions[1]; column++) {
                dtype currentGradPassedDown = outputTensor.gradAt({row, column});
                for (size_t dotProductIterator=0; dotProductIterator < currentInputTensor->dimensions[1]; dotProductIterator++) {
                    
                    currentInputTensor->grads[row*inpTensorColumns + dotProductIterator] +=
                        weights.data[column*weightColumns + dotProductIterator] * currentGradPassedDown;

                    weights.grads[column*weightColumns + dotProductIterator] += 
                        currentInputTensor->data[row*inpTensorColumns + dotProductIterator] * currentGradPassedDown; 
                    }

                biases.grads[column] += currentGradPassedDown;
            }
        }
    };

    const size_t threads_n = std::thread::hardware_concurrency();
    std::vector<std::future<void>> futures(threads_n);
    const size_t chunkSize = std::ceil(outputTensor.dimensions[0] / threads_n);
    for (size_t t=0; t < futures.size(); t++) {
        size_t start = chunkSize * t, end = std::min(start+chunkSize, outputTensor.dimensions[0]); 
        futures[t] = std::async(std::launch::async, processRow, start, end);
    }

    for (auto& future: futures) future.get();
}

void LinearLayer::manageDimensions(const Tensor& inputTensor) {
    if (
        inputTensor.dimensions.size() != 2
    ) {
        std::cout << inputTensor.dimensions;
        throw std::runtime_error("dimensionality for linear layer input tensor must be 2. dimensions received are printed above");
    }

    if (
        inputTensor.dimensions[1] != weights.dimensions[1]
    ) {
        std::cout << inputTensor.dimensions[1] << " != " << weights.dimensions[1] << "\n";
        throw std::runtime_error("input tensor columns don't match weight tensor rows in linear layer. mismatch printed above");
    }

    if (
        inputTensor.dimensions[0] != outputTensor.dimensions[0] or weights.dimensions[0] != outputTensor.dimensions[1]
    ) {
        adjustOutTensorDimensions( {inputTensor.dimensions[0], weights.dimensions[0]} );
    }
}

void LinearLayer::matmulWithBias() {

    auto processRow = [&](size_t startRow, size_t endRow) {
        for (size_t inputRow=startRow; inputRow < endRow; inputRow++) {
            for (size_t weightRow=0; weightRow < outputTensor.dimensions[1]; weightRow++) {
                outputTensor.at({inputRow, weightRow}) = std::transform_reduce(
                    std::execution::par_unseq, 
                    &currentInputTensor->at({inputRow, 0}), 
                    &currentInputTensor->at({inputRow, currentInputTensor->dimensions[1] - 1}),
                    &weights.at({weightRow, 0}),
                    biases.at({0, weightRow})
                ); // dot product
            }
        }
    };

    const size_t threads_n = std::thread::hardware_concurrency();
    std::vector<std::future<void>> futures(threads_n);
    const size_t chunkSize = std::ceil( (double) outputTensor.dimensions[0] / threads_n);
    for (size_t t=0; t < futures.size(); t++) {
        size_t start = chunkSize * t, end = std::min(start+chunkSize, outputTensor.dimensions[0]); 
        futures[t] = std::async(std::launch::async, processRow, start, end);
    }

    for (auto& future: futures) future.get();

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
    if (!(currentInputTensor)) throw std::runtime_error("backward before forward impossible");

    for (size_t i = 0; i < currentInputTensor->length; i++) {
        currentInputTensor->grads[i] = (currentInputTensor->data[i] >= 0 ? outputTensor.grads[i] : 0);
    }

    setInputTensorPointer(nullptr);
}

Conv2d::Conv2d( size_t inChannels, size_t outChannels, size_t kernelSize, size_t stride, size_t paddingSize ) : 
        inChannels(inChannels), outChannels(outChannels), kernelSize(kernelSize), stride(stride), paddingSize(paddingSize),
        kernels( KaimingWeightsVector(kernelSize*kernelSize*inChannels, outChannels),
                 { outChannels, inChannels, kernelSize, kernelSize } ),
        biases( std::vector<dtype>(outChannels, 0), {outChannels} ) {}


void Conv2d::print() {
    std::cout << "conv2d. input channels: " << inChannels << ", output channels: " << outChannels << ", kernel size: "
            << kernelSize << ", stride: " << stride << ", padding size: " << paddingSize << "\n";
}


dtype Conv2d::convolve(size_t pictureIndex, size_t outChannel, int inputRow, int inputCol) const {
    dtype sum = biases.data[outChannel];

    for (size_t inputChannel = 0; inputChannel < inChannels; inputChannel++) {
        for (size_t kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
            for (size_t kernelCol = 0; kernelCol < kernelSize; kernelCol++) {

                if (inputRow + kernelRow < 0 or inputCol + kernelCol < 0 or 
                    inputRow + kernelRow >= currentInputTensor->dimensions[2] or
                    inputCol + kernelCol >= currentInputTensor->dimensions[3]) {
                    continue;
                }

                sum +=  currentInputTensor->at( {pictureIndex, inputChannel, inputRow + kernelRow, inputCol + kernelCol} ) * 
                        kernels.at( {outChannel, inputChannel, kernelRow, kernelCol} );
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
            for (int row = -static_cast<int>(paddingSize) ; row < static_cast<int>(inputTensor.dimensions[2] + paddingSize - kernelSize + 1); row += stride) {

                for (int column = - static_cast<int>(paddingSize); column < static_cast<int>(inputTensor.dimensions[3] + paddingSize - kernelSize + 1); column += stride) {
                
                    outputTensor.data[locInOutput] = convolve(picture, channelOut, row, column);
                    // std::cout << outputTensor.data[locInOutput] << ", ";
                    locInOutput++;
                }
            }
        }
    }
}

void Conv2d::convolveBackward(size_t pictureIndex, size_t outChannel, int inputRow, int inputCol, dtype gradPassedDown) {

    biases.gradAt( {outChannel} ) += gradPassedDown;

    for (size_t inputChannel = 0; inputChannel < inChannels; inputChannel++) {
        for (size_t kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
            for (size_t kernelCol = 0; kernelCol < kernelSize; kernelCol++) {

                if (inputRow + kernelRow < 0 or inputCol + kernelCol < 0 or 
                    inputRow + kernelRow >= currentInputTensor->dimensions[2] or
                    inputCol + kernelCol >= currentInputTensor->dimensions[3]) {
                    continue;
                }

                currentInputTensor->gradAt( {pictureIndex, inputChannel, inputRow + kernelRow, inputCol + kernelCol} ) += 
                    kernels.at( {outChannel, inputChannel, kernelRow, kernelCol} ) * gradPassedDown;

                kernels.gradAt( {outChannel, inputChannel, kernelRow, kernelCol} ) += gradPassedDown * 
                    currentInputTensor->at( {pictureIndex, inputChannel, inputRow + kernelRow, inputCol + kernelCol} ); 
            }
        }
    }
}

void Conv2d::backward() {
    if (!(currentInputTensor)) throw std::runtime_error("backward before forward impossible");

    size_t locInOutput = 0;

    for (size_t picture = 0; picture < currentInputTensor->dimensions[0]; picture++) {
        for (size_t channelOut = 0; channelOut < outChannels; channelOut++) {
            for (int row = -static_cast<int>(paddingSize) ; row < static_cast<int>(currentInputTensor->dimensions[2] + paddingSize - kernelSize + 1); row += stride) {

                for (int column = - static_cast<int>(paddingSize); column < static_cast<int>(currentInputTensor->dimensions[3] + paddingSize - kernelSize + 1); column += stride) {

                    dtype gradPassedDown = outputTensor.grads[locInOutput];
                    convolveBackward(picture, channelOut, row, column, gradPassedDown);
                    locInOutput++;
                }
            }
        }
    }

    setInputTensorPointer(nullptr);
}


void Conv2d::manageDimensions( const Tensor& inputTensor ) {
    if (inputTensor.dimensions.size() != 4) throw std::runtime_error("input tensor dimensionality must be four for Conv2d");

    auto convolvedSize = [this](size_t size) {
        return (size + 2*paddingSize - kernelSize)/stride + 1;
    };

    TensorDims neededOutTensorDims = { inputTensor.dimensions[0],
                                                outChannels,
                                                convolvedSize(inputTensor.dimensions[2]),
                                                convolvedSize(inputTensor.dimensions[3]) };
    if (outputTensor.dimensions != neededOutTensorDims) {
        adjustOutTensorDimensions(neededOutTensorDims);
    }

    // std::cout << "output of conv dimensions: " << outputTensor.dimensions << "output of conv length: " << outputTensor.length << "\n";
}


Reshape::Reshape( const std::vector<size_t>& newDimensions, std::optional<size_t> freeDimension ) : 
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

        else throw std::runtime_error("reshape can't be done when the length of the input and the output is different");
    }
}



} // namespace mygrad