#pragma once

#include <vector>
#include <iostream>
#include "types.hpp"
#include "tensor.hpp"

namespace mygrad {

std::vector<dtype> normDistVector(size_t length);
std::vector<dtype> KaimingWeightsVector(size_t inFeatures, size_t outFeatures);

Tensor retrieveBatchFromData(const Tensor& dataTensor, const std::vector<size_t>& indices);
Tensor retrieveBatchFromLabels(const Tensor& labels, const std::vector<size_t>& indices);
Tensor standartize(const Tensor& tensor);
std::vector<size_t> shuffledIndices(size_t size);
std::vector<size_t> slicedIndices(const std::vector<size_t>& indices, int start, int length);


template<typename T>
inline std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
    out << "{";
    for (int i = 0; i < v.size(); i++) {
        out << v[i];
        if (i != v.size() - 1) {
            out << ", ";
        }
    }
    out << "}\n";
    return out;
}

}
