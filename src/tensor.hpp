#pragma once

#include "types.hpp"

struct Tensor {
    size_t length;
    
    std::vector<size_t> dimensions;
    std::vector<int> strides;
    std::unique_ptr<dtype[]> data;
    std::unique_ptr<dtype[]> grads;

    Tensor( dtype* data, dtype* grads,
            const std::vector<size_t>& dimensions );
    
    Tensor( const std::vector<size_t>& dimensions );

    Tensor( const std::vector<dtype>& dataVector,
            const std::vector<size_t>& dimensions );

    void print() const;
    void printGrad() const;
    inline dtype& at(const std::vector<int>& indices) {
        return data[indicesToLocationIn1dArray(indices)];
    }

    inline dtype at(const std::vector<int>& indices) const {
        return data[indicesToLocationIn1dArray(indices)];
    }

    inline dtype& gradAt(const std::vector<int>& indices) {
        return grads[indicesToLocationIn1dArray(indices)];
    }

    inline dtype gradAt(const std::vector<int>& indices) const {
        return grads[indicesToLocationIn1dArray(indices)];
    }

    inline std::vector<int> stridesFromDimensions(const std::vector<size_t>& dimensions) const;
    inline int lengthFromDimensions(const std::vector<size_t>& dimensions) const;
    int indicesToLocationIn1dArray(const std::vector<int>& indices) const;

    Tensor operator+( const Tensor& other ) const;
    Tensor operator-( const Tensor& other ) const;
    Tensor substractColumn ( const Tensor& column ) const;
    Tensor addColumn( const Tensor& column ) const;
    Tensor exp() const;
    Tensor log() const;
    Tensor max(int maxAlongDimension) const;
    Tensor sum(int sumAlongDimension) const;

protected:    
    void printRecursively(int start, int dimension, bool printByBlocks, bool printGrad) const;
    inline void checkIndexDimension( int dimension ) const;
    inline void checkValidityOfDimension( int dimension ) const;

};