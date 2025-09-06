#pragma once

#include "mygrad/mygrad.hpp"

using namespace mygrad;

struct Dataset {
    Tensor train; 
    Tensor eval;

    Dataset( Tensor&& train, Tensor&& eval) : 
        train(std::move(train)), 
        eval(std::move(eval)) {}
};

Dataset loadCatImages(float trainSplit, float evalSplit);
void convertTensorToPng(const Tensor& imgTensor, size_t index, const std::string& filename);