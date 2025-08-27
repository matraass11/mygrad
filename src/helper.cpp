#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <exception>
#include <cmath>

#include "mygrad/helper.hpp"
#include "mygrad/types.hpp"

namespace mygrad { 

static std::random_device dev;
static std::mt19937 generator(dev());

std::vector<dtype> normDistVector(size_t length, dtype standardDeviation = 1) {
    std::normal_distribution<dtype> normDist{0, standardDeviation};
    std::vector<dtype> v(length);
    for (size_t i = 0; i < length; i++) {
        v[i] = normDist(generator);
    }

    return v;
}

std::vector<dtype> KaimingWeightsVector(size_t inFeatures, size_t outFeatures) {
    dtype variance = 2 / ( (dtype)inFeatures );
    return normDistVector(inFeatures*outFeatures, std::sqrt(variance));
}


Tensor retrieveBatchFromData(const Tensor& dataTensor, const std::vector<size_t>& indices) {

    TensorDims batchDimensions = dataTensor.dimensions;
    batchDimensions[0] = indices.size();
    Tensor batchTensor = Tensor::zeros(batchDimensions);
    const size_t subTensorSize = dataTensor.strides[0];
    for (size_t i=0; i < indices.size(); i++) {
        std::memcpy(batchTensor.data.get() + i*subTensorSize,
                    dataTensor.data.get() + indices[i]*subTensorSize, 
                    subTensorSize*sizeof(dtype));
    }
    return batchTensor;
}

Tensor retrieveBatchFromLabels(const Tensor& labelsTensor, const std::vector<size_t>& indices) {
    
    Tensor batchTensor = Tensor::zeros( {indices.size() } );
    for (size_t i=0; i < indices.size(); i++) {
        batchTensor.data[i] = labelsTensor.data[indices[i]];
    }
    return batchTensor;
}

Tensor standartize(const Tensor& tensor) {
    dtype mean = tensor.mean(), std = tensor.std();

    Tensor standartizedTensor = Tensor::zeros( tensor.dimensions );
    for (size_t i = 0; i < tensor.length; i++ ) {
        standartizedTensor.data[i] = (tensor.data[i] - mean) / std;
    }
    return standartizedTensor;
}


std::vector<size_t> shuffledIndices(size_t size) {
    std::vector<size_t> indices(size);
    std::iota(indices.begin(), indices.end(), 0);

    std::shuffle(indices.begin(), indices.end(), generator);

    return indices;
}

std::vector<size_t> slicedIndices(const std::vector<size_t>& indices, int start, int length) {
    int end = std::min(start + length, static_cast<int>(indices.size()));
    return std::vector<size_t>(indices.begin() + start, indices.begin() + end);
}

} // namespace mygrad