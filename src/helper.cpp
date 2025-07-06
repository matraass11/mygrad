#include <iostream>
#include <random>

#include "helper.hpp"

static std::random_device dev;
static std::mt19937 generator(dev());

std::vector<dtype> normDistVector(size_t length, dtype standardDeviation = 1) {
    std::normal_distribution<double> normDist{0, standardDeviation};
    std::vector<dtype> v(length);
    for (int i = 0; i < length; i++) {
        v[i] = normDist(generator);
    }

    return v;
}

std::vector<dtype> KaimingWeightsVector(size_t inFeatures, size_t outFeatures) {
    dtype variance = 2 / ( (dtype)inFeatures );
    return normDistVector(inFeatures*outFeatures, sqrt(variance));
}