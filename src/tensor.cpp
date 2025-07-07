#include <memory>
#include <iostream>
#include "tensor.hpp"

Tensor::Tensor( dtype* data, dtype* grads,
                const std::vector<size_t>& dimensionsVector) :  
    data(data), 
    grads(grads),
    dimensions(dimensionsVector),
    length(lengthFromDimensionsVector(dimensions)) {}

Tensor::Tensor( const std::vector<size_t>& dimensionsVector ) : 
    Tensor(
        new dtype[lengthFromDimensionsVector(dimensionsVector)], 
        new dtype[lengthFromDimensionsVector(dimensionsVector)],
        dimensionsVector
    ) {}

Tensor::Tensor( const std::vector<dtype>& dataVector,
                const std::vector<size_t>& dimensionsVector ) :
    Tensor(
        [&]() {
            dtype* data = new dtype[lengthFromDimensionsVector(dimensionsVector)];
            std::copy(dataVector.begin(), dataVector.end(), data);
            return data;
        }(), // lambda for copying data into the tensor
        new dtype[lengthFromDimensionsVector(dimensionsVector)],
        dimensionsVector
    ) {
        if (dataVector.size() != length) {
            std::cerr << "tensor initialized with vector of wrong size. exiting\n"; 
            // might remove later because this requires proper initialization even of an empty vector
            std::cerr << dataVector.size() << " != " << length << "\n";
            exit(1);
        }
    }


void Tensor::print() const {
    printRecursively(0, 0, length, false);
    std::cout << "\n";
}

void Tensor::printGrad() const {
    printRecursively(0, 0, length, true);
    std::cout << "\n";
}

dtype& Tensor::at(const std::vector<int>& indices) {
    return data[indicesToLocationIn1dArray(indices)];
}

dtype& Tensor::gradAt(const std::vector<int>& indices) {
    return grads[indicesToLocationIn1dArray(indices)];
}

int Tensor::lengthFromDimensionsVector(const std::vector<size_t>& dimensionsVector) const {
    size_t length = 1;
    for (int i=0; i<dimensionsVector.size(); i++){
        length *= dimensionsVector[i];
    }
    return length;
}

inline int Tensor::indicesToLocationIn1dArray(const std::vector<int>& indices) const {
    if (indices.size() != dimensions.size()){
        std::cerr << "wrong indices, exiting\n";
        exit(1);
    }  
    int locationOfElementInDataArray = 0;
    int increment = length;
    for (int i=0; i < indices.size(); i++){
        increment /= dimensions[i];
        locationOfElementInDataArray += indices[i] * increment;
    }
    if (locationOfElementInDataArray > length){
        std::cerr << "error: invalid indices, exiting\n";
        exit(1);
    }
    return locationOfElementInDataArray;
}

void Tensor::printRecursively(int start, int dimension, int volumeOfPreviousDimension, bool printGrad) const {
    int increment = volumeOfPreviousDimension / dimensions[dimension];
    int lastSubDimensionEntry = start + increment*(dimensions[dimension] - 1); 
    std::cout << "[";
    for (int i=start; i <= lastSubDimensionEntry; i+=increment){
        if (increment==1){
            if (printGrad) {
                std::cout << grads[i];
            }
            else {
                std::cout << data[i];
            }
        }
        else {
            printRecursively(i, dimension+1, increment, printGrad);
        }
        if (i != lastSubDimensionEntry){
            std::cout << ", ";
        }
    }
    std::cout << "]";
}