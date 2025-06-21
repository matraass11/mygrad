#include <fstream>
#include "model.hpp"
#include "helper.hpp"

Model::Model() :
    w1(normDistArray<28*28*100>(), {28*28, 100}),
    b1({ }, {100}),
    w2(normDistArray<100*100>(), {100, 100}),
    b2({ }, {100}),
    w3(normDistArray<100*10>(), {100, 10})
    {}

void Model::save(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        // error
    }

    for (const Tensor& weightTensor : weights) {
        file.write(reinterpret_cast<const char*>(weightTensor.dataArrayPtr), weightTensor.length);
        file.write(reinterpret_cast<const char*>(weightTensor.gradArrayPtr), weightTensor.length);
    }
}

Model::Model(const std::string& filename) :     
    w1({ }, {28*28, 100}),
    b1({ }, {100}),
    w2({ }, {100, 100}),
    b2({ }, {100}),
    w3({ }, {100, 10}) 

    {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        // error
    }

    for (Tensor& weightTensor : weights) {
        file.read(reinterpret_cast<char*>(weightTensor.dataArrayPtr), weightTensor.length);
        file.read(reinterpret_cast<char*>(weightTensor.gradArrayPtr), weightTensor.length);
    }
}

void Model::print() {
    for (const Tensor& tensor: weights){
        tensor.print();
    }
}


