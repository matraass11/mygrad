#include <fstream>
#include <exception>
#include "model.hpp"
#include "helper.hpp"

Model::Model() {}

void Model::save(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("failed to open file " + filename);

    for (const Tensor* const parameterTensor : parameters) {
        file.write(reinterpret_cast<char*>(parameterTensor->data.get()), parameterTensor->length*sizeof(dtype));
        file.write(reinterpret_cast<char*>(parameterTensor->grads.get()), parameterTensor->length*sizeof(dtype));
    }
}

void Model::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("failed to open file " + filename);

    for (Tensor* const parameterTensor : parameters) {
        file.read(reinterpret_cast<char*>(parameterTensor->data.get()), parameterTensor->length*sizeof(dtype));
        file.read(reinterpret_cast<char*>(parameterTensor->grads.get()), parameterTensor->length*sizeof(dtype));
    }
}

void Model::print() const {
    for (const Tensor* const parameterTensor: parameters){
        parameterTensor->print();
    }
}

void Model::printGrads() const {
    for (const Tensor* const parameterTensor: parameters){
        parameterTensor->printGrad();
    }
}

void Model::zeroGrad() {
    for (Tensor *const parameterTensor: parameters) {
        parameterTensor->zeroGrad();
    }
    for (Tensor *const parameterTensor: intermediateTensors) {
        parameterTensor->zeroGrad();
    }
}


Tensor& Model::forward(Tensor& x) {
    l1.forward(x);
    rl1.forward(l1.outputTensor);
    l2.forward(rl1.outputTensor);
    rl2.forward(l2.outputTensor);
    l3.forward(rl2.outputTensor);
    return l3.outputTensor;
};

void Model::backward() {
    l3.backward();
    rl2.backward();
    l2.backward();
    rl1.backward();
    l1.backward();
}

