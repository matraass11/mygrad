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
    #ifndef NDEBUG
        if (!(currentInputTensor)) throw std::runtime_error("backward before forward impossible");
    #endif

    const size_t threads_n = ThreadPool::size();
    const size_t chunkSize = std::ceil(outputTensor.dimensions[0] / (double) threads_n);
    for (size_t t=0; t < threads_n; t++) {
        size_t startRow = chunkSize * t, endRow = std::min(startRow+chunkSize, outputTensor.dimensions[0]); 
        ThreadPool::push(

            [this, startRow, endRow] () {
                const size_t inpTensorColumns = currentInputTensor->dimensions[1];
                const size_t weightColumns = weights.dimensions[1];

                for (size_t row = startRow; row < endRow; row++){
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
            });
    }

    ThreadPool::waitUntilDone();

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

    const size_t threads_n = ThreadPool::size();
    const size_t chunkSize = std::ceil( (double) outputTensor.dimensions[0] / threads_n);

    for (size_t t=0; t < threads_n; t++) {
        size_t startRow = chunkSize * t, endRow = std::min(startRow+chunkSize, outputTensor.dimensions[0]); 
        ThreadPool::push(
            [this, startRow, endRow] {
                for (size_t inputRow=startRow; inputRow < endRow; inputRow++) {
                    for (size_t weightRow=0; weightRow < outputTensor.dimensions[1]; weightRow++) {
                        outputTensor.at({inputRow, weightRow}) = std::transform_reduce( 
                            &currentInputTensor->at({inputRow, 0}), 
                            &currentInputTensor->at({inputRow, currentInputTensor->dimensions[1] - 1}),
                            &weights.at({weightRow, 0}),
                            biases.at({0, weightRow})
                        ); // dot product
                    }
                }
            }
        );
    }

    ThreadPool::waitUntilDone();

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

    Tensor& input = *currentInputTensor;
    const size_t almostInpLoc = pictureIndex * input.strides[0] + inputRow * input.strides[2] + inputCol * input.strides[3];

    for (size_t inputChannel = 0; inputChannel < inChannels; inputChannel++) {
        const size_t leftUpperCornerInpLoc = almostInpLoc + inputChannel * input.strides[1];
        for (size_t kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
            for (size_t kernelCol = 0; kernelCol < kernelSize; kernelCol++) {

                if (inputRow + kernelRow < 0 or inputCol + kernelCol < 0 or 
                    inputRow + kernelRow >= input.dimensions[2] or
                    inputCol + kernelCol >= input.dimensions[3]) {
                    continue;
                }

                const size_t inpLoc = leftUpperCornerInpLoc + kernelRow * input.strides[2] + kernelCol;
                const size_t kernelLoc = outChannel * kernels.strides[0] + inputChannel * kernels.strides[1] + 
                                         kernelRow  * kernels.strides[2] + kernelCol;

                sum += input.data[inpLoc] * kernels.data[kernelLoc];

                // sum +=  currentInputTensor->at( {pictureIndex, inputChannel, inputRow + kernelRow, inputCol + kernelCol} ) * 
                //         kernels.at( {outChannel, inputChannel, kernelRow, kernelCol} );
            }
        }
    }

    return sum;
}


void Conv2d::forward(Tensor& inputTensor) {
    manageDimensions( inputTensor );
    setInputTensorPointer( &inputTensor );


    const size_t threads_n = ThreadPool::size();
    const size_t chunkSize = std::ceil( (double) inputTensor.dimensions[0] / threads_n);

    for (size_t t=0; t < threads_n; t++) {
        size_t startPicture = chunkSize * t, endPicture = std::min(startPicture+chunkSize, outputTensor.dimensions[0]); 
        ThreadPool::push([this, &inputTensor, startPicture, endPicture] {

            size_t locInOutput = startPicture * outputTensor.strides[0];

            for (size_t picture = startPicture; picture < endPicture; picture++) {
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
        });
    }

    ThreadPool::waitUntilDone();
}

void Conv2d::convolveBackward(size_t pictureIndex, size_t outChannel, int inputRow, int inputCol, dtype gradPassedDown) {

    biases.gradAt( {outChannel} ) += gradPassedDown;
    Tensor& input = *currentInputTensor;
    const size_t almostInpLoc = pictureIndex * input.strides[0] + inputRow * input.strides[2] + inputCol * input.strides[3];

    for (size_t inputChannel = 0; inputChannel < inChannels; inputChannel++) {
        const size_t leftUpperCornerInpLoc = almostInpLoc + inputChannel * input.strides[1];
        for (size_t kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
            for (size_t kernelCol = 0; kernelCol < kernelSize; kernelCol++) {

                if (inputRow + kernelRow < 0 or inputCol + kernelCol < 0 or 
                    inputRow + kernelRow >= currentInputTensor->dimensions[2] or
                    inputCol + kernelCol >= currentInputTensor->dimensions[3]) {
                    continue;
                }

                // currentInputTensor->gradAt( {pictureIndex, inputChannel, inputRow + kernelRow, inputCol + kernelCol} ) += 
                //     kernels.at( {outChannel, inputChannel, kernelRow, kernelCol} ) * gradPassedDown;
                
                // kernels.gradAt( {outChannel, inputChannel, kernelRow, kernelCol} ) += gradPassedDown * 
                //     currentInputTensor->at( {pictureIndex, inputChannel, inputRow + kernelRow, inputCol + kernelCol} ); 
                
                // check if replacing at with manual indexing will speed it up

                const size_t inpLoc = leftUpperCornerInpLoc + kernelRow * input.strides[2] + kernelCol;
                const size_t kernelLoc = outChannel * kernels.strides[0] + inputChannel * kernels.strides[1] + 
                                         kernelRow  * kernels.strides[2] + kernelCol;

                input.grads[inpLoc] += kernels.data[kernelLoc] * gradPassedDown;
                kernels.grads[kernelLoc] += input.data[inpLoc] * gradPassedDown;
            }
        }
    }
}

void Conv2d::backward() {
    #ifndef NDEBUG
        if (!(currentInputTensor)) throw std::runtime_error("backward before forward impossible");
    #endif

    Tensor& inputTensor = *currentInputTensor;
    const size_t threads_n = ThreadPool::size();
    const size_t chunkSize = std::ceil( (double) inputTensor.dimensions[0] / threads_n);

    for (size_t t=0; t < threads_n; t++) {
        size_t startPicture = chunkSize * t, endPicture = std::min(startPicture+chunkSize, outputTensor.dimensions[0]); 
        ThreadPool::push([this, &inputTensor, startPicture, endPicture] {

            size_t locInOutput = startPicture * outputTensor.strides[0];

            for (size_t picture = startPicture; picture < endPicture; picture++) {
                for (size_t channelOut = 0; channelOut < outChannels; channelOut++) {
                    for (int row = -static_cast<int>(paddingSize) ; row < static_cast<int>(inputTensor.dimensions[2] + paddingSize - kernelSize + 1); row += stride) {

                        for (int column = - static_cast<int>(paddingSize); column < static_cast<int>(inputTensor.dimensions[3] + paddingSize - kernelSize + 1); column += stride) {
                        
                            dtype gradPassedDown = outputTensor.grads[locInOutput];
                            convolveBackward(picture, channelOut, row, column, gradPassedDown);
                            locInOutput++;
                        }
                    }
                }
            }
        });
    }

    ThreadPool::waitUntilDone();

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
        if (inputTensor.dimensions[0] % 2 != 0 or inputTensor.dimensions.size() != 2) 
            throw std::runtime_error("dimensions of distribution should be of size 2 and have n rows for means and n rows for logvariance");
    #endif

    TensorDims neededOutDims = {inputTensor.dimensions[0] / 2, inputTensor.dimensions[1]};
    if (outputTensor.dimensions != neededOutDims) {
        adjustOutTensorDimensions(neededOutDims);
        currentEpsilons = Tensor::zeros(neededOutDims);
    }
}

void Reparameterize::forward( Tensor& inputTensor ) {
     
    manageDimensions( inputTensor );
    setInputTensorPointer( &inputTensor );

    const size_t offsetBetweenMeanAndLogvar = outputTensor.length;
    for (size_t i = 0; i < outputTensor.length; i++) {
        dtype mean = inputTensor.data[i], std = std::exp(inputTensor.data[i + offsetBetweenMeanAndLogvar] / 2);
        std::cout << "mean: " << mean << ", std: " << std << "\n";

        dtype epsilon = normDist(generator);
        currentEpsilons.data[i] = epsilon;
        outputTensor.data[i] = mean + epsilon * std;
    }
}

void Reparameterize::backward() {
    if (!(currentInputTensor)) throw std::runtime_error("backward before forward impossible");

    Tensor& inputTensor = *currentInputTensor;

    const size_t offsetBetweenMeanAndLogvar = outputTensor.length;
    for (size_t i = 0; i < outputTensor.length; i++) {
        dtype gradPassedDown = outputTensor.grads[i];
        inputTensor.grads[i] += 1 * gradPassedDown;
        dtype std = std::exp(inputTensor.data[i + offsetBetweenMeanAndLogvar] / 2); 
        inputTensor.grads[i + offsetBetweenMeanAndLogvar] += 0.5 * std * currentEpsilons.data[i] * gradPassedDown;
    }
    
    setInputTensorPointer( nullptr );
}

} // namespace mygrad