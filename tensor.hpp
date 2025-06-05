#pragma once

class Tensor {
protected:
    double* dataArrayPtr;
    double* gradArrayPtr;

public:
    const std::vector<int> dimensions;
    const size_t length;

public:

    Tensor( double* dataArrayPtr, double* gradArrayPtr,
            const std::vector<int>& dimensionsVector );

    void print() const;
    void printGrad() const;
    inline double at(const std::vector<int>& indices) const;
    inline double atLocationIn1dArray(uint location) const;
    double gradAt(const std::vector<int>& indices) const;
    
    
    void incrementGradAt(uint locationIn1dArray, double increment);
    void incrementGradAt(const std::vector<int>& indices, double increment);
    void setAllGradsTo(double newGrad);
    
    virtual void backwardFurther() const {};

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
    Tensor* const leftParent; 
    Tensor* const rightParent;       
    
    std::vector<int> dimensionsFromParents(const Tensor& leftParent, const Tensor& rightParent) const;
    
public:
    void matMulParentsIntoSelf2d();
    void backwardFurther() const override;
    
};  


class TensorSum : public Tensor {
public:
    TensorSum (
        double* dataArrayPtr, double* gradArrayPtr,
        Tensor& rightTensor, Tensor& leftTensor );

protected:
    Tensor* const leftParent;
    Tensor* const rightParent;

public:
    void sumParentsIntoSelf();
    void backwardFurther() const override;
};