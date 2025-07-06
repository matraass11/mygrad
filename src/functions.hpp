#pragma once

struct Tensor;

class Function {
public:
    virtual void forward() = 0;
    virtual void backward() = 0;
    virtual void checkDimensions() = 0;
};

class twoInputFunction : Function {
protected:
    Tensor* leftInputTensor = nullptr;
    Tensor* rightInputTensor = nullptr;
    Tensor* outputTensor = nullptr;
};


class Sum : twoInputFunction {
public:
    Sum();
    void forward() override;
    void backward() override;

protected:
    void checkDimensions() override;
};


class LossFunction : Function {
protected:
    Tensor* dataTensor = nullptr;
    Tensor* modelOutputTensor = nullptr;
    Tensor* lossTensor = nullptr;

    LossFunction();
};


class MseLoss : LossFunction {
public: 
    MseLoss(); 
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
