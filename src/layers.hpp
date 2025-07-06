#pragma once

#include "tensor.hpp" 

struct LinearLayer {
    Tensor weights;
    Tensor biases;
    Tensor* inputTensor = nullptr;
    Tensor* outputTensor = nullptr;

    LinearLayer( int inFeatures, int outFeatures, const std::vector<dtype>& data = {});
    
    void forward( Tensor& inputTensor, Tensor& outputTensor );
    void checkDimensions( Tensor& inputTensor, Tensor& outputTensor  );
    void setTensorPointers( Tensor* inputTensor, Tensor* outputTensor );
    void matmulWithBias();
    void backward();
    void matmulWithBias_backward();
};