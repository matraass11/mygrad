#pragma once

#include "types.hpp"

std::vector<dtype> normDistVector(size_t length);
std::vector<dtype> KaimingWeightsVector(size_t inFeatures, size_t outFeatures);

template<typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
    for (auto n: v) {
        out << n << ", ";
    }
    out << "\n";
    return out;
}