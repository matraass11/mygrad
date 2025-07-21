#pragma once

#include "types.hpp"
#include "tensor.hpp"

std::vector<dtype> normDistVector(size_t length);
std::vector<dtype> KaimingWeightsVector(size_t inFeatures, size_t outFeatures);

template<typename T>
inline std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
    for (auto n: v) {
        out << n << ", ";
    }
    out << "\n";
    return out;
}

Tensor load_mnist_images(const std::string& path);
Tensor load_mnist_labels(const std::string& path);
void visualize_image_ascii(const Tensor& images, const Tensor& labels, int index);
Tensor retrieveBatchFromData(const Tensor& dataTensor, const std::vector<size_t>& indices);
Tensor retrieveBatchFromLabels(const Tensor& labels, const std::vector<size_t>& indices);
