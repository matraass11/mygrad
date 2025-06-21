#include <fstream>
#include "model.hpp"
#include "helper.hpp"

Model::Model() :
    w1(normDistArray<10>(), {10}),
    b1({ }, {10}),
    w2(normDistArray<10>(), {10}),
    b2({ }, {10}),
    w3(normDistArray<10>(), {10})
    {}

void Model::save(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        // error
    }

    for (const Tensor& weightTensor : weights) {
        file.write(reinterpret_cast<const char*>(weightTensor.dataArrayPtr), weightTensor.length*sizeof(double));
        file.write(reinterpret_cast<const char*>(weightTensor.gradArrayPtr), weightTensor.length*sizeof(double));
    }
}

Model::Model(const std::string& filename) :     
    w1({ }, {10}),
    b1({ }, {10}),
    w2({ }, {10}),
    b2({ }, {10}),
    w3({ }, {10}) 

    {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        // error
    }

    for (Tensor& weightTensor : weights) {
        file.read(reinterpret_cast<char*>(weightTensor.dataArrayPtr), weightTensor.length*sizeof(double));
        file.read(reinterpret_cast<char*>(weightTensor.gradArrayPtr), weightTensor.length*sizeof(double));
    }
}

void Model::print() {
    for (const Tensor& tensor: weights){
        tensor.print();
    }
}


