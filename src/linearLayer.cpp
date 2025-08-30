#include <vector>

#include "mygrad/helper.hpp"
#include "mygrad/linearLayer.hpp"
#include "mygrad/threadPool.hpp"


namespace mygrad {

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
    // std::cout << outputTensor.dimensions << "are the dimensions of the output\n";
    // std::cout << chunkSize << " is the chunk size\n";

    for (size_t t=0; t < threads_n; t++) {
        size_t startRow = chunkSize * t, endRow = std::min(startRow+chunkSize, outputTensor.dimensions[0]); 
        ThreadPool::push(
            [this, startRow, endRow] {
                for (size_t inputRow=startRow; inputRow < endRow; inputRow++) {
                    for (size_t weightRow=0; weightRow < outputTensor.dimensions[1]; weightRow++) {
                        outputTensor.at({inputRow, weightRow}) = std::transform_reduce( 
                            &currentInputTensor->at({inputRow, 0}), 
                            &currentInputTensor->at({inputRow, currentInputTensor->dimensions[1]}),
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

} // namespace mygrad