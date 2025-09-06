#pragma once 

#include "tensor.hpp"
#include "types.hpp"

#include <string>

namespace mygrad {

class CrossEntropyLoss {
public:

    CrossEntropyLoss();

    dtype operator()( Tensor& logits, const Tensor& labels );
    void backward();
    
private:

    const Tensor* labels = nullptr;
    Tensor* logits = nullptr;
    Tensor currentSoftmaxOutput;

    void setInputPointers( Tensor* logits, const Tensor* labels ) { this->logits = logits, this->labels = labels; } 
    inline void checkDimensions( Tensor& logits, const Tensor& labels );
};


class MSEloss {
public:

    const std::string reduction; // reduction = sum will divide by batch size. reduction = mean will divide by entire size of input

    MSEloss( const std::string& reduction) : reduction(reduction) {
        if (reduction != std::string("sum") and reduction != std::string("mean")) throw std::runtime_error("reduction for MSEloss must be one of: 'sum', 'mean'");
    };

    dtype operator()( Tensor& outputs, const Tensor& labels );
    void backward();

private:

    const Tensor* labels = nullptr;
    Tensor* outputs = nullptr;
    
    void setInputPointers( Tensor* outputs, const Tensor* labels ) { this->outputs = outputs, this->labels = labels; }
    inline void checkDimensions( Tensor& outputs, const Tensor& labels );
};


class KLdivWithStandardNormal {
public:

    KLdivWithStandardNormal() {};

    dtype operator()( Tensor& distribution, dtype beta );
    void backward();

private:

    dtype currentBeta;

    Tensor* distribution;

    void setInputPointers( Tensor* distribution ) { this->distribution = distribution; }
};

} // namespace mygrad