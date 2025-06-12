#pragma once

class Tensor;

struct function {
public:
    virtual void forward() = 0;
    virtual void backward() = 0;
};

struct mat_mul2d : function {
    Tensor* const leftInputTensor_p;
    Tensor* const rightInputTensor_p;
    Tensor* const outputTensor_p;
    
    mat_mul2d(Tensor& leftInputTensor, Tensor& rightInputTensor, Tensor& outputTensor);

    void checkDimensions();

    void forward() override;
    void backward() override;
};


// tensors:
// input 
// w1, b1, w2, b2
// hidden1, hidden1WithBias, hidden1Act, output, outputWithBias, outputAct, loss

// functions: 
// MatMul(input, w1, hidden1)
// Sum(hidden1, b1)
// MatMul()
