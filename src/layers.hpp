#pragma once

#include "tensor.hpp" 
#include "helper.hpp"

struct LinearLayer {
    Tensor weights;
    Tensor biases;
    Tensor* inputTensor = nullptr;
    Tensor* outputTensor = nullptr;

    LinearLayer( size_t inFeatures, size_t outFeatures);
    LinearLayer( size_t inFeatures, size_t outFeatures, const std::vector<dtype>& data);
    
    void forward( Tensor& inputTensor, Tensor& outputTensor );
    void checkDimensions( Tensor& inputTensor, Tensor& outputTensor  );
    void setTensorPointers( Tensor* inputTensor, Tensor* outputTensor );
    void matmulWithBias();
    void backward();
    void matmulWithBias_backward();
};