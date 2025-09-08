#include "mygrad/conv2d.hpp"
#include "mygrad/helper.hpp"
#include "mygrad/threadPool.hpp"

namespace mygrad {

Conv2d::Conv2d( size_t inChannels, size_t outChannels, size_t kernelSize, size_t stride, size_t paddingSize ) : 
        inChannels(inChannels), outChannels(outChannels), kernelSize(kernelSize), stride(stride), paddingSize(paddingSize),
        kernels( KaimingWeightsVector(kernelSize*kernelSize*inChannels, outChannels),
                 { outChannels, inChannels, kernelSize, kernelSize } ),
        biases( std::vector<dtype>(outChannels, 0), {outChannels} ),
        matrixFormInput(Tensor::zeros({1})),
        kernelMutexes(outChannels) {}


void Conv2d::print() {
    std::cout << "conv2d. input channels: " << inChannels << ", output channels: " << outChannels << ", kernel size: "
            << kernelSize << ", stride: " << stride << ", padding size: " << paddingSize << "\n";
}

void Conv2d::forward( Tensor& inputTensor ) {

    manageDimensions( inputTensor );
    setInputTensorPointer( &inputTensor );

    im2col( inputTensor );
    kernels.reshape( {outChannels, static_cast<size_t>(kernels.strides[0])} ); // the second term should be equal to matrixFormColumns

    const size_t threads_n = ThreadPool::size();
    const size_t chunkSize = std::ceil( (double) kernels.dimensions[0] / threads_n);

    for (size_t t=0; t < threads_n; t++) {
        size_t startFilter = chunkSize * t, endFilter = std::min(startFilter+chunkSize, kernels.dimensions[0]); 
        if (startFilter == endFilter) break;
        ThreadPool::push(
            [this, startFilter, endFilter] {

                for (size_t filterRow=startFilter; filterRow < endFilter; filterRow++) {
                    for (size_t inputRow=0; inputRow < matrixFormInput.dimensions[0]; inputRow++) {

                        const size_t inputImage = inputRow / outputTensor.strides[1];
                        const size_t outputRow = (inputRow % outputTensor.strides[1]) / outputTensor.strides[2];
                        const size_t outputCol = inputRow % outputTensor.strides[2];

                        outputTensor.at({inputImage, filterRow, outputRow, outputCol}) = std::transform_reduce( 
                            &matrixFormInput.data[inputRow * matrixFormInput.dimensions[1]], 
                            &matrixFormInput.data[(inputRow + 1) * matrixFormInput.dimensions[1]],
                            &kernels.at({filterRow, 0}),
                            biases.at({filterRow})
                        ); // dot product
                    }
                }
            }
        );
    }

    ThreadPool::waitUntilDone();

    kernels.reshape( {outChannels, inChannels, kernelSize, kernelSize});
}


void Conv2d::movePatchToMatrixForm( size_t picture, int leftUpperRow, int leftUpperCol, Tensor& matrixFormInput, size_t rowInMatrixForm ) {

    // ASSUMES MATRIXFORMTENSOR IS FILLED WITH ZEROS

    const Tensor& inputTensor = *currentInputTensor;
    const size_t inpLoc = picture * inputTensor.strides[0] + leftUpperRow * inputTensor.strides[2] + leftUpperCol;

    size_t matrixFormLoc = rowInMatrixForm * matrixFormInput.strides[0];

    for (size_t inputChannel = 0; inputChannel < inChannels; inputChannel++) {
        size_t inpLocAtChannel = inpLoc + inputChannel * inputTensor.strides[1];
        for (size_t patchRow = 0; patchRow < kernelSize; patchRow++) {
            for (size_t patchCol = 0; patchCol < kernelSize; patchCol++) {

                if (! (leftUpperRow + patchRow < 0 or leftUpperCol + patchCol < 0 or 
                    leftUpperRow + patchRow >= inputTensor.dimensions[2] or
                    leftUpperCol + patchCol >= inputTensor.dimensions[3])) {

                        matrixFormInput.data[matrixFormLoc] = inputTensor.data[inpLocAtChannel + patchRow * inputTensor.strides[2] + patchCol];
                }
                matrixFormLoc++;
            }
        }
    }
}


void Conv2d::im2col( const Tensor& inputTensor ) {

    const size_t matrixFormColumns = kernelSize * kernelSize * inChannels;
    const size_t matrixFormRowsForSinglePicture = convolvedSize(inputTensor.dimensions[2]) * convolvedSize(inputTensor.dimensions[3]); 
    TensorDims neededMatrixFormDims = {matrixFormRowsForSinglePicture * inputTensor.dimensions[0], matrixFormColumns};

    if (matrixFormInput.dimensions != neededMatrixFormDims) {
        matrixFormInput = Tensor::zeros( neededMatrixFormDims );
    } 
    else { std::fill(matrixFormInput.data.get(), matrixFormInput.data.get() + matrixFormInput.length, 0 ); } // instead of reassignment to avoid an unnecessary memory allocation


    const size_t threads_n = ThreadPool::size();
    const size_t chunkSize = std::ceil( (double) inputTensor.dimensions[0] / threads_n);

    for (size_t t=0; t < threads_n; t++) {
        size_t startPicture = chunkSize * t, endPicture = std::min(startPicture+chunkSize, outputTensor.dimensions[0]); 
        ThreadPool::push([this, &inputTensor, matrixFormRowsForSinglePicture, startPicture, endPicture] {

            size_t rowInMatrixForm = startPicture * matrixFormRowsForSinglePicture;

            for (size_t picture = startPicture; picture < endPicture; picture++) {
                for (int row = -static_cast<int>(paddingSize) ; row < static_cast<int>(inputTensor.dimensions[2] + paddingSize - kernelSize + 1); row += stride) {

                    for (int column = -static_cast<int>(paddingSize); column < static_cast<int>(inputTensor.dimensions[3] + paddingSize - kernelSize + 1); column += stride) {
                    
                        movePatchToMatrixForm(picture, row, column, matrixFormInput, rowInMatrixForm);
                        rowInMatrixForm++;
                    }
                }
            }
        });
    }

    ThreadPool::waitUntilDone();
}


void Conv2d::convolveBackward(size_t pictureIndex, size_t outChannel, int inputRow, int inputCol, dtype gradPassedDown) {
    std::lock_guard lock(kernelMutexes[outChannel]);

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
        if (startPicture == endPicture) break;
        ThreadPool::push([this, &inputTensor, startPicture, endPicture] {

            size_t locInOutput = startPicture * outputTensor.strides[0];

            for (size_t picture = startPicture; picture < endPicture; picture++) {
                for (size_t channelOut = 0; channelOut < outChannels; channelOut++) {
                    for (int row = -static_cast<int>(paddingSize); row < static_cast<int>(inputTensor.dimensions[2] + paddingSize - kernelSize + 1); row += stride) {

                        for (int column = -static_cast<int>(paddingSize); column < static_cast<int>(inputTensor.dimensions[3] + paddingSize - kernelSize + 1); column += stride) {
                        
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
}

} // namespace mygrad