#include <fstream>
#include <stdexcept>
#include "mygrad/model.hpp"
#include "mygrad/helper.hpp"

namespace mygrad {

const std::vector<Tensor*> Model::parametersOfLayers(LayersContainer& layers) {
    std::vector<Tensor*> parameterTensors;
    for (size_t i = 0; i < layers.size(); i++) {
        const std::vector<Tensor*>& parametersOfLayer = layers[i].parameterTensors();
        parameterTensors.insert(parameterTensors.end(), parametersOfLayer.begin(), parametersOfLayer.end());
    }
    return parameterTensors;
}

const std::vector<Tensor*> Model::nonParametersOfLayers(LayersContainer& layers) {
    std::vector<Tensor*> nonParameterTensors;
    for (size_t i = 0; i < layers.size(); i++) {
        const std::vector<Tensor*>& nonParametersOfLayer = layers[i].nonParameterTensors();
        nonParameterTensors.insert(nonParameterTensors.end(), nonParametersOfLayer.begin(), nonParametersOfLayer.end());
    }
    return nonParameterTensors;
}

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
    for (size_t i = 0; i < layers.size(); i++) {
        layers[i].zeroGrad();
    }
}


Tensor& Model::forward(Tensor& x) {

    layers[0].forward(x);
    for (size_t i = 1; i < layers.size(); i++){
        layers[i].forward(layers[i-1].outputTensor);
    }
    return layers[layers.size() - 1].outputTensor;
};


Tensor& Model::operator()(Tensor& x) {
    return forward(x);
}


void Model::backward() {
    for (int i = layers.size() - 1; i >= 0; i--) {
        layers[i].backward();
    }
}


} // namespace mygrad