#include "mygrad/mygrad.hpp"

using namespace mygrad;

struct Dataset {
    Tensor train; 
    Tensor eval;
    Tensor test;

    Dataset( Tensor&& train, Tensor&& eval, Tensor&& test) : 
        train(std::move(train)), 
        eval(std::move(eval)),
        test(std::move(test)) {}
};

Dataset loadCatImages(float trainSplit, float evalSplit, float testSplit);
void convertTensorToPng(const Tensor& imgTensor, int index, const std::string& filename);