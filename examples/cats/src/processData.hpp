#include "mygrad/mygrad.hpp"

using namespace mygrad;

struct Dataset {
    Tensor train; 
    Tensor test;

    Dataset( Tensor&& train, Tensor&& test) : 
        train(std::move(train)), 
        test(std::move(test)) {}
};

Dataset loadCatImages(float trainSplit, float testSplit);
void convertTensorToPng(const Tensor& imgTensor, size_t index, const std::string& filename);