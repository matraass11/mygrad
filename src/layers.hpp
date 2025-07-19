#pragma once

#include "tensor.hpp" 
#include "helper.hpp"

static const std::vector<size_t> defaultDimensions = {1, 1};

struct Layer {
    Tensor* currentInputTensor = nullptr;
    Tensor outputTensor;

    Layer(const std::vector<size_t> outDimensions = defaultDimensions);

    virtual void forward( Tensor& inputTensor ) = 0;
    virtual void backward() = 0;
    
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
    
private:
    void matmulWithBias();
    void matmulWithBias_backward();

    void checkDimensions( const Tensor& inputTensor ) override; 
};


struct ReLU : Layer {

    ReLU() : Layer() {};

    void forward( Tensor& inputTensor ) override;
    void backward() override;

private:
    inline void checkDimensions( const Tensor& inputTensor ) override;

};