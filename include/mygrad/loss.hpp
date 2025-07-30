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

    inline void setInputPointers( Tensor* logits, const Tensor* labels ); // relies on the input tensor not changing
    inline void checkDimensions( Tensor& logits, const Tensor& labels );
};

} // namespace mygrad