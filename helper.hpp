#include <iostream>
#include <random>
#pragma once

static std::random_device dev;
static std::mt19937 generator(dev());
static std::normal_distribution<double> normDist(0, 1);

template<size_t length>
const std::array<double, length> normDistArray() {
    std::array<double, length> arr;
    for (int i = 0; i < length; i++) {
        arr[i] = normDist(generator);
    }
    return arr;
}