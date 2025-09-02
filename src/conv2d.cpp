#include "mygrad/conv2d.hpp"
#include "mygrad/helper.hpp"
#include "mygrad/threadPool.hpp"

namespace mygrad {

Conv2d::Conv2d( size_t inChannels, size_t outChannels, size_t kernelSize, size_t stride, size_t paddingSize ) : 
        inChannels(inChannels), outChannels(outChannels), kernelSize(kernelSize), stride(stride), paddingSize(paddingSize),
        kernels( KaimingWeightsVector(kernelSize*kernelSize*inChannels, outChannels),
                 { outChannels, inChannels, kernelSize, kernelSize } ),
        biases( std::vector<dtype>(outChannels, 0), {outChannels} ),
        matrixFormCurrentInput(Tensor::zeros({1})) {}


void Conv2d::print() {
    std::cout << "conv2d. input channels: " << inChannels << ", output channels: " << outChannels << ", kernel size: "
            << kernelSize << ", stride: " << stride << ", padding size: " << paddingSize << "\n";
}

void Conv2d::forward2(Tensor& inputTensor) {

    if (inputTensor.dimensions.size() != 4) throw std::runtime_error("input tensor dimensionality must be four for Conv2d");

    TensorDims neededOutTensorDims = { inputTensor.dimensions[0],
                                       outChannels,
                                       convolvedSize(inputTensor.dimensions[2]),
                                       convolvedSize(inputTensor.dimensions[3]) };

    if (outputTensor.dimensions != neededOutTensorDims) {
        adjustOutTensorDimensions(neededOutTensorDims);
    }
    setInputTensorPointer( &inputTensor );

    
    im2col(inputTensor, matrixFormCurrentInput);
    kernels.reshape( {outChannels, static_cast<size_t>(kernels.strides[0])} ); // the second term should be equal to matrixFormColumns

    const size_t threads_n = ThreadPool::size();
    const size_t chunkSize = std::ceil( (double) kernels.dimensions[0] / threads_n);

    for (size_t t=0; t < threads_n; t++) {
        size_t startFilter = chunkSize * t, endFilter = std::min(startFilter+chunkSize, kernels.dimensions[0]); 
        if (startFilter == endFilter) break;
        ThreadPool::push(
            [this, startFilter, endFilter] {

                for (size_t filterRow=startFilter; filterRow < endFilter; filterRow++) {
                    for (size_t inputRow=0; inputRow < matrixFormCurrentInput.dimensions[0]; inputRow++) {

                        size_t inputImage = inputRow / outputTensor.strides[1];
                        size_t outputRow = (inputRow % outputTensor.strides[1]) / outputTensor.strides[2];
                        size_t outputCol = inputRow % outputTensor.strides[2];
                        outputTensor.at({inputImage, filterRow, outputRow, outputCol})
                        = std::transform_reduce( 
                            &matrixFormCurrentInput.data[inputRow * matrixFormCurrentInput.dimensions[1]], 
                            &matrixFormCurrentInput.data[(inputRow + 1) * matrixFormCurrentInput.dimensions[1]],
                            &kernels.at({filterRow, 0}),
                            biases.at({filterRow})
                        ); // dot product
                    }
                }
            }
        );
    }

    ThreadPool::waitUntilDone();

    // outputTensor.reshape(neededOutTensorDims);
    kernels.reshape( {outChannels, inChannels, kernelSize, kernelSize});
}


void Conv2d::movePatchToMatrixForm( size_t picture, int leftUpperRow, int leftUpperCol, Tensor& matrixFormTensor, size_t rowInMatrixForm ) {

    // ASSUMES MATRIXFORMTENSOR IS FILLED WITH ZEROS

    const Tensor& inputTensor = *currentInputTensor;
    size_t inpLoc = picture * inputTensor.strides[0] + leftUpperRow * inputTensor.strides[2] + leftUpperCol;

    size_t matrixFormLoc = rowInMatrixForm * matrixFormTensor.strides[0];

    for (size_t inputChannel = 0; inputChannel < inChannels; inputChannel++) {
        size_t inpLocAtChannel = inpLoc + inputChannel * inputTensor.strides[1];
        for (size_t patchRow = 0; patchRow < kernelSize; patchRow++) {
            for (size_t patchCol = 0; patchCol < kernelSize; patchCol++) {

                if (! (leftUpperRow + patchRow < 0 or leftUpperCol + patchCol < 0 or 
                    leftUpperRow + patchRow >= inputTensor.dimensions[2] or
                    leftUpperCol + patchCol >= inputTensor.dimensions[3])) {

                        matrixFormTensor.data[matrixFormLoc] = inputTensor.data[inpLocAtChannel + patchRow * inputTensor.strides[2] + patchCol];
                }
                matrixFormLoc++;
            }
        }
    }
}


void Conv2d::im2col( const Tensor& inputTensor, Tensor& matrixFormTensor ) {

    manageDimensions(inputTensor); // not sure about where to actually do this but for testing this will be here

    const size_t matrixFormColumns = kernelSize * kernelSize * inChannels;
    // const size_t matrixFormRows = outputTensor.dimensions[2]; // assuming manageDimensions has already taken place
    const size_t matrixFormRowsForSinglePicture = convolvedSize(inputTensor.dimensions[2]) * convolvedSize(inputTensor.dimensions[3]); 
    TensorDims neededMatrixFormDims = {matrixFormRowsForSinglePicture * inputTensor.dimensions[0], matrixFormColumns};
    // std::cout << "dimensions needed for matrix form: " << neededMatrixFormDims;

    if (matrixFormTensor.dimensions != neededMatrixFormDims) {
        matrixFormTensor = Tensor::zeros( neededMatrixFormDims );
    } 
    else { std::fill(matrixFormTensor.data.get(), matrixFormTensor.data.get() + matrixFormTensor.length, 0 ); } // instead of reassignment to avoid an unnecessary memory allocation


    const size_t threads_n = ThreadPool::size();
    const size_t chunkSize = std::ceil( (double) inputTensor.dimensions[0] / threads_n);

    for (size_t t=0; t < threads_n; t++) {
        size_t startPicture = chunkSize * t, endPicture = std::min(startPicture+chunkSize, outputTensor.dimensions[0]); 
        ThreadPool::push([this, &inputTensor, &matrixFormTensor, matrixFormRowsForSinglePicture, startPicture, endPicture] {

            size_t rowInMatrixForm = startPicture * matrixFormRowsForSinglePicture;

            for (size_t picture = startPicture; picture < endPicture; picture++) {
                for (int row = -static_cast<int>(paddingSize) ; row < static_cast<int>(inputTensor.dimensions[2] + paddingSize - kernelSize + 1); row += stride) {

                    for (int column = - static_cast<int>(paddingSize); column < static_cast<int>(inputTensor.dimensions[3] + paddingSize - kernelSize + 1); column += stride) {
                    
                        movePatchToMatrixForm(picture, row, column, matrixFormTensor, rowInMatrixForm); // is this line right?
                        rowInMatrixForm++;
                    }
                }
            }
        });
    }

    ThreadPool::waitUntilDone();
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


void Conv2d::forward1(Tensor& inputTensor) {
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

    TensorDims neededOutTensorDims = { inputTensor.dimensions[0],
                                       outChannels,
                                       convolvedSize(inputTensor.dimensions[2]),
                                       convolvedSize(inputTensor.dimensions[3]) };
    if (outputTensor.dimensions != neededOutTensorDims) {
        adjustOutTensorDimensions(neededOutTensorDims);
    }

    // std::cout << "output of conv dimensions: " << outputTensor.dimensions << "output of conv length: " << outputTensor.length << "\n";
}

} // namespace mygrad