#pragma once

#include "types.hpp"

struct Tensor {
    size_t length;
    std::vector<size_t> dimensions;
    std::unique_ptr<dtype[]> data;
    std::unique_ptr<dtype[]> grads;

    Tensor( dtype* data, dtype* grads,
            const std::vector<size_t>& dimensionsVector );
    
    Tensor( const std::vector<size_t>& dimensionsVector );

    Tensor( const std::vector<dtype>& dataVector,
            const std::vector<size_t>& dimensionsVector );

    void print() const;
    void printGrad() const;
    inline dtype& at(const std::vector<int>& indices);
    inline dtype& gradAt(const std::vector<int>& indices);

    inline int lengthFromDimensionsVector(const std::vector<size_t>& dimensionsVector) const;
    int indicesToLocationIn1dArray(const std::vector<int>& indices) const;

    Tensor exp();
    Tensor max(int maxAlongDimension);

protected:    
    void printRecursively(int start, int dimension, int volumeOfPreviousDimension, bool printGrad) const;
};