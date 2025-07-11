#pragma once

#include "tensor.hpp" 
#include "helper.hpp"

struct LinearLayer {

    Tensor* inputTensor = nullptr;
    Tensor weights;
    Tensor biases;

    Tensor outputTensor;
    static const size_t default_batch_size = 32;

    LinearLayer( size_t inFeatures, size_t outFeatures);
    LinearLayer( size_t inFeatures, size_t outFeatures, const std::vector<dtype>& data);
    
    void forward( Tensor& inputTensor );
    void backward();
    
private:
    void checkDimensions( Tensor& inputTensor );
    inline void adjustOutTensorDimensions( size_t newRowsN );
    void setInputTensorPointer( Tensor* inputTensor ); // relies on the input tensor not changing
    void matmulWithBias();
    void matmulWithBias_backward();
};