#pragma once

class TensorMatMulProduct;

class Tensor {
protected:
    double* dataArrayPtr;
    double* gradArrayPtr;

public:
    const std::vector<uint> dimensions;
    const size_t length;

public:

    Tensor( double* dataArrayPtr, double* gradArrayPtr,
            std::vector<uint> dimensionsVector);

    void print() const;
    double at(std::vector<int> indices) const;

protected:
    size_t lengthFromDimensionsVector(const std::vector<uint>& dimensionsVector) const;
    void printRecursively(uint start, uint dimension, uint volumeOfPreviousDimension) const;
    int indicesToLocationIn1dArray(std::vector<int> indices) const;
};



class TensorMatMulProduct : public Tensor {
public:
    TensorMatMulProduct (
        double* dataArrayPtr, double* gradArrayPtr,
        Tensor& rightTensor, Tensor& leftTensor);

protected:
    Tensor* leftParent; 
    Tensor* rightParent;       
    
    std::vector<uint> dimensionsFromParents(Tensor& leftParent, Tensor& rightParent);
    void assignNewParents(Tensor* leftParent, Tensor* rightParent);

public:
    void matMul2dIntoSelf(Tensor& leftTensor, Tensor& rightTensor);
    
};  