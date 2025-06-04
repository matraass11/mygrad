#pragma once

class TensorMatMulProduct;

class Tensor {
protected:
    double* dataArrayPtr;
    double* gradArrayPtr;

public:
    const std::vector<int> dimensions;
    const size_t length;

public:

    Tensor( double* dataArrayPtr, double* gradArrayPtr,
            const std::vector<int>& dimensionsVector);

    void print() const;
    void printGrad() const;
    double at(const std::vector<int>& indices) const;
    double gradAt(const std::vector<int>& indices) const;
    
    virtual void backwardFurther() const {};
    void incrementGradAt(const std::vector<int>& indices, double increment);

    void setAllGradsTo(double newGrad);

protected:
    size_t lengthFromDimensionsVector(const std::vector<int>& dimensionsVector) const;
    void printRecursively(uint start, uint dimension, uint volumeOfPreviousDimension, bool printGrad) const;
    int indicesToLocationIn1dArray(const std::vector<int>& indices) const;
};



class TensorMatMulProduct : public Tensor {
public:
    TensorMatMulProduct (
        double* dataArrayPtr, double* gradArrayPtr,
        Tensor& rightTensor, Tensor& leftTensor);

protected:
    Tensor* leftParent; 
    Tensor* rightParent;       
    
    std::vector<int> dimensionsFromParents(const Tensor& leftParent, const Tensor& rightParent) const;
    void assignNewParents(Tensor* leftParent, Tensor* rightParent);
    
public:
    void matMul2dIntoSelf(Tensor& leftTensor, Tensor& rightTensor);
    void backwardFurther() const override;
    
};  