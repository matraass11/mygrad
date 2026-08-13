#include <fstream>
#include <filesystem>
#include <iostream>
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

size_t Model::checkpointSize() const {
    size_t bytes = 0;
    for (const Tensor* const parameterTensor : parameters) {
        bytes += parameterTensor->length * sizeof(dtype) * 2; // data and grads
    }
    return bytes;
}

void Model::load(const std::string& filename) {
    // a checkpoint carries no shapes, so the file size is the only thing that
    // can catch weights belonging to some other architecture. without this the
    // read simply succeeds on the first bytes and the model predicts noise.
    std::error_code errorCode;
    const std::uintmax_t sizeOnDisk = std::filesystem::file_size(filename, errorCode);
    if (errorCode) throw std::runtime_error("failed to read the size of file " + filename);

    const size_t expectedSize = checkpointSize();
    if (sizeOnDisk != expectedSize) {
        std::cout << "this model wants " << expectedSize << " bytes, " << filename << " holds " << sizeOnDisk << "\n";
        throw std::runtime_error("checkpoint does not fit this model");
    }

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("failed to open file " + filename);

    for (Tensor* const parameterTensor : parameters) {
        file.read(reinterpret_cast<char*>(parameterTensor->data.get()), parameterTensor->length*sizeof(dtype));
        file.read(reinterpret_cast<char*>(parameterTensor->grads.get()), parameterTensor->length*sizeof(dtype));
    }
    if (!file) throw std::runtime_error("failed to read all of file " + filename);
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