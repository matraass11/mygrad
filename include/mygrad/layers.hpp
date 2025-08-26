#pragma once

#include <vector>
#include <optional>
#include "tensor.hpp" 
#include "helper.hpp"

namespace mygrad {

struct Layer {
    Tensor* currentInputTensor = nullptr;
    Tensor outputTensor;

    Layer();

    Layer( Layer&& other ) = default;
    virtual ~Layer() = default;

    virtual void forward( Tensor& inputTensor )        = 0;
    virtual void backward()                            = 0;
    virtual std::vector<Tensor*> parameterTensors()    = 0;
    virtual std::vector<Tensor*> nonParameterTensors() = 0;

    void zeroGrad();
    
protected:
    inline void setInputTensorPointer( Tensor* inputTensor ); // relies on the input tensor not changing
    virtual void manageDimensions( const Tensor& inputTensor ) = 0; 
    inline void adjustOutTensorDimensions( const TensorDims& newDimensions );
};


struct LinearLayer : Layer {

    Tensor weights;
    Tensor biases;
    
    LinearLayer( size_t inFeatures, size_t outFeatures);
    LinearLayer( size_t inFeatures, size_t outFeatures, const std::vector<dtype>& data);
    void forward( Tensor& inputTensor ) override;
    void backward() override;
    std::vector<Tensor*> parameterTensors() override { return { &weights, &biases }; }
    std::vector<Tensor*> nonParameterTensors() override { return { &outputTensor }; }
    
private:
    void matmulWithBias();
    void matmulWithBiasBackward();

    void manageDimensions( const Tensor& inputTensor ) override; 
};


struct ReLU : Layer {

    ReLU() : Layer() {};

    void forward( Tensor& inputTensor ) override;
    void backward() override;
    std::vector<Tensor*> parameterTensors() override { return {}; }
    std::vector<Tensor*> nonParameterTensors() override { return { &outputTensor }; }

private:
    inline void manageDimensions( const Tensor& inputTensor ) override;

};


struct Conv2d : Layer {
    const size_t inChannels, outChannels, kernelSize, stride, paddingSize;
    Tensor kernels;
    Tensor biases;

    Conv2d( size_t inChannels, size_t outChannels, size_t kernelSize, size_t stride, size_t paddingSize );

    void print();

    void forward( Tensor& inputTensor ) override;
    void backward() override;
    std::vector<Tensor*> parameterTensors() override { return { &kernels, &biases }; }
    std::vector<Tensor*> nonParameterTensors() override { return { &outputTensor }; }

private:
    dtype convolve(size_t pictureIndex, size_t filterIndex, int inputRow, int inputCol) const;
    void convolveBackward(size_t pictureIndex, size_t filterIndex, int inputRow, int inputCol, dtype gradPassedDown);

    inline void manageDimensions( const Tensor& inputTensor ) override;
};


struct Reshape : Layer {

    TensorDims newDimensions;
    std::optional<size_t> freeDimension;
    
    Reshape( const TensorDims& newDimensions, std::optional<size_t> freeDimension );
    
    void forward( Tensor& inputTensor ) override;
    void backward() override;
    
    std::vector<Tensor*> parameterTensors() override { return {}; }
    std::vector<Tensor*> nonParameterTensors() override { return { &outputTensor }; }
    
private:

    inline void manageDimensions( const Tensor& inputTensor ) override;
};


} // namespace mygrad