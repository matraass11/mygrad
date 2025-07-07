#include <fstream>
#include "model.hpp"
#include "helper.hpp"

Model::Model(size_t batchSize) : 
    l1out( {batchSize, NEURONS_N} ), 
    l2out( {batchSize, NEURONS_N} ),
    l3out( {batchSize, 1} ),
    currentBatchSize(batchSize) {}

void Model::save(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        // error
    }

    for (const Tensor* const parameterTensor : parameters) {
        file.write(reinterpret_cast<const char*>(parameterTensor->data), parameterTensor->length*sizeof(dtype));
        file.write(reinterpret_cast<const char*>(parameterTensor->grads), parameterTensor->length*sizeof(dtype));
    }
}

void Model::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        // error
    }

    for (Tensor* const parameterTensor : parameters) {
        file.read(reinterpret_cast<char*>(parameterTensor->data), parameterTensor->length*sizeof(dtype));
        file.read(reinterpret_cast<char*>(parameterTensor->grads), parameterTensor->length*sizeof(dtype));
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

void Model::changeBatchSize(size_t newBatchSize){
    l1out = Tensor( {newBatchSize, NEURONS_N} );
    l2out = Tensor( {newBatchSize, NEURONS_N} );
    l3out = Tensor( {newBatchSize, 1} );
}


Tensor& Model::forward(Tensor& x) {
    if (x.dimensions[0] != currentBatchSize){
        changeBatchSize(x.dimensions[0]);
    }

    l1.forward(x, l1out);
    l2.forward(l1out, l2out);
    l3.forward(l2out, l3out);
    return l3out;
};

void Model::backward() {
    for (int i = 0; i < l3out.length; i++) {
        l3out.grads[i] = 1;
    }
    l3.backward();
    l2.backward();
    l1.backward();
}

