#pragma once

#include "types.hpp"

struct Tensor {
    dtype* data;
    dtype* grads;
    const std::vector<int> dimensions;
    const size_t length;

    Tensor( dtype* data, dtype* grads,
            const std::vector<int>& dimensionsVector );
    
    Tensor( const std::vector<int>& dimensionsVector );

    Tensor( const std::vector<dtype>& dataVector,
            const std::vector<int>& dimensionsVector );

    void print() const;
    void printGrad() const;
    dtype& at(const std::vector<int>& indices);
    dtype& gradAt(const std::vector<int>& indices);

    size_t lengthFromDimensionsVector(const std::vector<int>& dimensionsVector) const;
    int indicesToLocationIn1dArray(const std::vector<int>& indices) const;

protected:    
    void printRecursively(int start, int dimension, int volumeOfPreviousDimension, bool printGrad) const;
};