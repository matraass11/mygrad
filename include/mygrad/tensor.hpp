#pragma once

#include <memory>
#include <cstring>
#include <vector>
#include "smallArray.hpp"
#include "types.hpp"


namespace mygrad {

constexpr size_t MAX_TENSOR_DIMENSIONALITY = 10;
using TensorDims = SmallArray<size_t, MAX_TENSOR_DIMENSIONALITY>;
using TensorIndices = TensorDims;
using TensorStrides = SmallArray<int, MAX_TENSOR_DIMENSIONALITY>;

struct Tensor {
    size_t length;
    
    TensorDims dimensions;
    TensorStrides strides;
    std::unique_ptr<dtype[]> data;
    std::unique_ptr<dtype[]> grads;

    Tensor( const std::vector<dtype>& dataVector,
            const TensorDims& dimensions );

private:
    Tensor( const TensorDims& dimensions );
public:

    static Tensor zeros( const TensorDims& dimensions );

    void print() const;
    void printGrad() const;


    inline dtype& at(const TensorDims& indices)          { return data[indicesToLocationIn1dArray(indices)]; }

    inline dtype at(const TensorDims& indices)     const { return data[indicesToLocationIn1dArray(indices)]; }

    inline dtype& gradAt(const TensorDims& indices)      { return grads[indicesToLocationIn1dArray(indices)]; }

    inline dtype gradAt(const TensorDims& indices) const { return grads[indicesToLocationIn1dArray(indices)]; }


    inline void reshape(const TensorDims& newDimensions) {
        if (lengthFromDimensions(newDimensions) != length) 
            throw std::runtime_error("tensor reshaped into dimensions with length different from the current one");
        dimensions = newDimensions;
        strides = stridesFromDimensions(newDimensions);
    }

    inline void zeroGrad() { std::memset(grads.get(), 0, length*sizeof(dtype)); }
    inline void zeroData() { std::memset(grads.get(), 0, length*sizeof(dtype)); }


    static TensorStrides stridesFromDimensions(const TensorDims& dimensions);
    static size_t lengthFromDimensions(const TensorDims& dimensions);
    size_t indicesToLocationIn1dArray(const TensorDims& indices) const;
    TensorDims locationIn1dArrayToIndices(size_t location) const;

    Tensor operator+( const Tensor& other ) const;
    Tensor operator-( const Tensor& other ) const;
    Tensor substractColumn ( const Tensor& column ) const;
    Tensor addColumn( const Tensor& column ) const;
    Tensor exp() const;
    Tensor log() const;
    Tensor max(int maxAlongDimension) const;
    Tensor sum(int sumAlongDimension) const;
    Tensor argmax(int argmaxAlongDimension) const;
    dtype mean() const;
    dtype std() const;

private:    
    void printRecursively(size_t start, size_t dimension, bool printByBlocks, bool printGrad) const;
    // inline void checkIndexDimension( int dimension ) const;
    inline void checkValidityOfDimension( int dimension ) const;

};

} // namespace mygrad