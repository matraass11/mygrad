#include <fstream>
#include "model.hpp"
#include "helper.hpp"

Model::Model(size_t initialBatchSize) : l1out( {32, 200} ) {}

void Model::save(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        // error
    }

    for (const Tensor* parameterTensor : parameters) {
        file.write(reinterpret_cast<const char*>(parameterTensor->data), parameterTensor->length*sizeof(dtype));
        file.write(reinterpret_cast<const char*>(parameterTensor->grads), parameterTensor->length*sizeof(dtype));
    }
}

// Model::Model(const std::string& filename) {    
    // w1({ }, {10}),
    // b1({ }, {10}),
    // w2({ }, {10}),
    // b2({ }, {10}),
    // w3({ }, {10}) 

    // {
    // std::ifstream file(filename, std::ios::binary);
    // if (!file) {
    //     // error
    // }

    // for (Tensor& parameterTensor : weights) {
    //     file.read(reinterpret_cast<char*>(parameterTensor.data), parameterTensor->length*sizeof(dtype));
    //     file.read(reinterpret_cast<char*>(parameterTensor.grads), parameterTensor.length*sizeof(dtype));
    // }
// }

void Model::print() {
    for (const Tensor* tensor: parameters){
        tensor->print();
    }
}


