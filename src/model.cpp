#include <fstream>
#include "model.hpp"
#include "helper.hpp"

Model::Model() {}

void Model::save(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        // error
    }

    for (const Tensor* const parameterTensor : parameters) {
        file.write(reinterpret_cast<char*>(parameterTensor->data.get()), parameterTensor->length*sizeof(dtype));
        file.write(reinterpret_cast<char*>(parameterTensor->grads.get()), parameterTensor->length*sizeof(dtype));
    }
}

void Model::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        // error
    }

    for (Tensor* const parameterTensor : parameters) {
        file.read(reinterpret_cast<char*>(parameterTensor->data.get()), parameterTensor->length*sizeof(dtype));
        file.read(reinterpret_cast<char*>(parameterTensor->grads.get()), parameterTensor->length*sizeof(dtype));
    }
}

void Model::print() {
    for (const Tensor* const tensor: parameters){
        tensor->print();
    }
}

void Model::printGrads() {
    for (const Tensor* const tensor: parameters){
        tensor->printGrad();
    }
}


Tensor& Model::forward(Tensor& x) {
    l1.forward(x);
    l2.forward(l1.outputTensor);
    l3.forward(l2.outputTensor);
    return l3.outputTensor;
};

void Model::backward() {
    for (int i = 0; i < l3.outputTensor.length; i++) { // only for testing
        l3.outputTensor.grads[i] = 1;
    }
    l3.backward();
    l2.backward();
    l1.backward();
}

