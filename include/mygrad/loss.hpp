#pragma once 

#include "tensor.hpp"
#include "types.hpp"

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

    MSEloss() {};

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

    dtype operator()( Tensor& distribution );
    void backward();

private:

    Tensor* distribution;

    void setInputPointers( Tensor* distribution ) { this->distribution = distribution; }
};

} // namespace mygrad