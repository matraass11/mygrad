#pragma once

#include <vector>
#include <optional>
#include <random>
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

    Tensor& operator()( Tensor& inputTensor ) { forward(inputTensor); return outputTensor; };
    void zeroGrad();
    
protected:
    void setInputTensorPointer( Tensor* inputTensor ); // relies on the input tensor not changing
    virtual void manageDimensions( const Tensor& inputTensor ) = 0; 
    void adjustOutTensorDimensions( const TensorDims& newDimensions );
};


struct ReLU : Layer {

    ReLU() {};

    void forward( Tensor& inputTensor ) override;
    void backward() override;
    std::vector<Tensor*> parameterTensors() override { return {}; }
    std::vector<Tensor*> nonParameterTensors() override { return { &outputTensor }; }

private:
    inline void manageDimensions( const Tensor& inputTensor ) override;

};


struct Sigmoid : Layer {

    Sigmoid() {};

    void forward( Tensor& inputTensor ) override;
    void backward() override;
    std::vector<Tensor*> parameterTensors() override { return {}; }
    std::vector<Tensor*> nonParameterTensors() override { return { &outputTensor }; }

private:
    inline void manageDimensions( const Tensor& inputTensor ) override;

};


struct Reshape : Layer {

    TensorDims newDimensions;
    std::optional<size_t> freeDimension;
    
    Reshape( const TensorDims& newDimensions, std::optional<size_t> freeDimension = {} );
    
    void forward( Tensor& inputTensor ) override;
    void backward() override;
    
    std::vector<Tensor*> parameterTensors() override { return {}; }
    std::vector<Tensor*> nonParameterTensors() override { return { &outputTensor }; }
    
private:

    inline void manageDimensions( const Tensor& inputTensor ) override;
};


struct Upsample : Layer {
    const size_t scalingFactor;
    
    Upsample( size_t scalingFactor ) : scalingFactor(scalingFactor) {};
    
    void forward( Tensor& inputTensor ) override;
    void backward() override;
    
    std::vector<Tensor*> parameterTensors() override { return {}; }
    std::vector<Tensor*> nonParameterTensors() override { return { &outputTensor }; }
    
private:

    inline void manageDimensions( const Tensor& inputTensor ) override;
};


struct MaxPool2d : Layer {

    const size_t kernelSize;
    
    MaxPool2d( size_t kernelSize ) : kernelSize(kernelSize) {};
    
    void forward( Tensor& inputTensor ) override;
    void backward() override;
    
    std::vector<Tensor*> parameterTensors() override { return {}; }
    std::vector<Tensor*> nonParameterTensors() override { return { &outputTensor }; }
    
private:
    dtype pool(size_t pictureIndex, size_t filterIndex, size_t inputRow, size_t inputCol) const;
    void poolBackward(size_t pictureIndex, size_t channelIndex, size_t inputRow, size_t inputCol, dtype gradPassedDown);

    inline void manageDimensions( const Tensor& inputTensor ) override;
};


struct Reparameterize : Layer {

    Reparameterize() : generator(std::random_device()()), currentEpsilons(Tensor::zeros({1})) {};

    void forward( Tensor& inputTensor ) override;
    void backward() override;

    std::vector<Tensor*> parameterTensors() override { return {}; }
    std::vector<Tensor*> nonParameterTensors() override { return { &outputTensor, &currentEpsilons }; }

private:
    Tensor currentEpsilons; // for backprop
    std::mt19937 generator;
    std::normal_distribution<dtype> normDist {0, 1};

    inline void manageDimensions( const Tensor& inputTensor ) override;
};


} // namespace mygrad