#pragma once

#include "layers.hpp"

namespace mygrad {

struct Conv2d : Layer {
    const size_t inChannels, outChannels, kernelSize, stride, paddingSize;
    Tensor kernels;
    Tensor biases;

    Conv2d( size_t inChannels, size_t outChannels, size_t kernelSize, size_t stride, size_t paddingSize = 0 );

    void print();

    void forward( Tensor& inputTensor ) override {
        // forward2(inputTensor);
        forward1(inputTensor);
    }

    void forward1( Tensor& inputTensor );
    void forward2( Tensor& inputTensor );

    void backward() override;
    std::vector<Tensor*> parameterTensors() override { return { &kernels, &biases }; }
    std::vector<Tensor*> nonParameterTensors() override { return { &outputTensor }; }

// private:
    Tensor matrixFormCurrentInput;

    void im2col( const Tensor& inputTensor, Tensor& matrixFormTensor );
    void movePatchToMatrixForm( size_t picture, int leftUpperRow, int leftUpperCol, Tensor& matrixFormTensor, size_t rowInMatrixForm );
    constexpr size_t convolvedSize( size_t size ) noexcept { return (size + 2*paddingSize - kernelSize)/stride + 1; } 

private:

    dtype convolve(size_t pictureIndex, size_t filterIndex, int inputRow, int inputCol) const;
    void convolveBackward(size_t pictureIndex, size_t filterIndex, int inputRow, int inputCol, dtype gradPassedDown);

    inline void manageDimensions( const Tensor& inputTensor ) override;
};

} // namespace mygrad