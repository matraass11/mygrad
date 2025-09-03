#pragma once 

#include "layers.hpp"

namespace mygrad {

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

    void manageDimensions( const Tensor& inputTensor ) override; 
};

} // namespace mygrad