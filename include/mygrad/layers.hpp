#pragma once

#include "tensor.hpp" 
#include "helper.hpp"

namespace mygrad {

static const std::vector<size_t> defaultDimensions = {1, 1};


struct Layer {
    Tensor* currentInputTensor = nullptr;
    Tensor outputTensor;

    Layer(const std::vector<size_t> outDimensions = defaultDimensions);

    Layer( Layer&& other ) = default;
    virtual ~Layer() = default;

    virtual void forward( Tensor& inputTensor )        = 0;
    virtual void backward()                            = 0;
    virtual std::vector<Tensor*> parameterTensors()    = 0;
    virtual std::vector<Tensor*> nonParameterTensors() = 0;
    
protected:
    inline void setInputTensorPointer( Tensor* inputTensor ); // relies on the input tensor not changing
    virtual void checkDimensions( const Tensor& inputTensor ) = 0; 
    inline void adjustOutTensorDimensions( const std::vector<size_t>& newDimensions );
};


struct LinearLayer : Layer {

    Tensor weights;
    Tensor biases;
    
    LinearLayer( size_t inFeatures, size_t outFeatures);
    LinearLayer( size_t inFeatures, size_t outFeatures, const std::vector<dtype>& data);
    void forward( Tensor& inputTensor ) override;
    void backward() override;
    std::vector<Tensor*> parameterTensors() override { return {&weights, &biases}; }
    std::vector<Tensor*> nonParameterTensors() override { return {&outputTensor}; }
    
private:
    void matmulWithBias();
    void matmulWithBias_backward();

    void checkDimensions( const Tensor& inputTensor ) override; 
};


struct ReLU : Layer {

    ReLU() : Layer() {};

    void forward( Tensor& inputTensor ) override;
    void backward() override;
    std::vector<Tensor*> parameterTensors() override { return {}; }
    std::vector<Tensor*> nonParameterTensors() override { return { &outputTensor }; }

private:
    inline void checkDimensions( const Tensor& inputTensor ) override;

};

} // namespace mygrad