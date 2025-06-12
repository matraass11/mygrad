#pragma once

struct Tensor {
    double* dataArrayPtr;
    double* gradArrayPtr;
    const std::vector<int> dimensions;
    const size_t length;

    Tensor( double* dataArrayPtr, double* gradArrayPtr,
            const std::vector<int>& dimensionsVector );

    void print() const;
    void printGrad() const;
    double& at(const std::vector<int>& indices);
    double& gradAt(const std::vector<int>& indices);

    size_t lengthFromDimensionsVector(const std::vector<int>& dimensionsVector) const;
    int indicesToLocationIn1dArray(const std::vector<int>& indices) const;

protected:    
    void printRecursively(uint start, uint dimension, uint volumeOfPreviousDimension, bool printGrad) const;
};