#pragma once

#include "layers.hpp"
#include <mutex>

namespace mygrad {

struct Conv2d : Layer {
    const size_t inChannels, outChannels, kernelSize, stride, paddingSize;
    Tensor kernels;
    Tensor biases;

    Conv2d( size_t inChannels, size_t outChannels, size_t kernelSize, size_t stride, size_t paddingSize = 0 );

    void print();

    void forward( Tensor& inputTensor ) override;
    void backward() override;

    std::vector<Tensor*> parameterTensors() override { return { &kernels, &biases }; }
    std::vector<Tensor*> nonParameterTensors() override { return { &outputTensor }; }

private:
    std::vector<std::mutex> kernelMutexes;

    Tensor matrixFormInput;

    void im2col( const Tensor& inputTensor );
    void movePatchToMatrixForm( size_t picture, int leftUpperRow, int leftUpperCol, Tensor& matrixFormTensor, size_t rowInMatrixForm );
    
    constexpr size_t convolvedSize( size_t size ) noexcept { return (size + 2*paddingSize - kernelSize)/stride + 1; } 
    void convolveBackward(size_t pictureIndex, size_t filterIndex, int inputRow, int inputCol, dtype gradPassedDown);

    inline void manageDimensions( const Tensor& inputTensor ) override;
};

} // namespace mygrad