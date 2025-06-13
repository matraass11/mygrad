#pragma once

class Tensor;

class Function {
public:
    virtual void forward() = 0;
    virtual void backward() = 0;
    virtual void checkDimensions() = 0;
};

class twoInputFunction : Function {
protected:
    Tensor& leftInputTensor;
    Tensor& rightInputTensor;
    Tensor& outputTensor;

    twoInputFunction(Tensor& leftInputTensor, Tensor& rightInputTensor, Tensor& outputTensor);
    virtual void checkDimensions() = 0;
};


class mat_mul2d : twoInputFunction {
public:
    mat_mul2d(Tensor& leftInputTensor, Tensor& rightInputTensor, Tensor& outputTensor);
    void forward() override;
    void backward() override;

protected:
    void checkDimensions() override;
};

class sum : twoInputFunction {
public:
    sum(Tensor& leftInputTensor, Tensor& rightInputTensor, Tensor& outputTensor);
    void forward() override;
    void backward() override;

protected:
    void checkDimensions() override;
};


// tensors:
// input 
// w1, b1, w2, b2
// hidden1, hidden1WithBias, hidden1Act, output, outputWithBias, outputAct, loss

// functions: 
// MatMul(input, w1, hidden1)
// Sum(hidden1, b1)
// MatMul()
