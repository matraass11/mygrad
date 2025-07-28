#pragma once 

#include "tensor.hpp"
#include "types.hpp"

static const size_t defaultBatchSize = 32;

class CrossEntropyLoss {
    const Tensor* labels = nullptr;
    Tensor* logits = nullptr;
    Tensor currentSoftmaxOutput;

public:

    CrossEntropyLoss( const size_t classes ) : currentSoftmaxOutput({ defaultBatchSize, classes} ) {};

    dtype operator()( Tensor& logits, const Tensor& labels );
    void backward();
    
private:
    inline void setInputPointers( Tensor* logits, const Tensor* labels ); // relies on the input tensor not changing
    inline void checkDimensions( Tensor& logits, const Tensor& labels );
};
