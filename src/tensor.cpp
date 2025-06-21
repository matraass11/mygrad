#include <memory>
#include <iostream>
#include "tensor.hpp"

Tensor::Tensor( dtype* dataArrayPtr, dtype* gradArrayPtr,
                const std::vector<int>& dimensionsVector) :  
    dataArrayPtr(dataArrayPtr), 
    gradArrayPtr(gradArrayPtr),
    dimensions(dimensionsVector),
    length(lengthFromDimensionsVector(dimensions)) {}


void Tensor::print() const {
    printRecursively(0, 0, length, false);
    std::cout << "\n";
}

void Tensor::printGrad() const {
    printRecursively(0, 0, length, true);
    std::cout << "\n";
}

dtype& Tensor::at(const std::vector<int>& indices) {
    return dataArrayPtr[indicesToLocationIn1dArray(indices)];
}

dtype& Tensor::gradAt(const std::vector<int>& indices) {
    return gradArrayPtr[indicesToLocationIn1dArray(indices)];
}

size_t Tensor::lengthFromDimensionsVector(const std::vector<int>& dimensionsVector) const {
    size_t length = 1;
    for (int i=0; i<dimensions.size(); i++){
        length *= dimensions[i];
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

void Tensor::printRecursively(uint start, uint dimension, uint volumeOfPreviousDimension, bool printGrad) const {
    uint increment = volumeOfPreviousDimension / dimensions[dimension];
    uint lastSubDimensionEntry = start + increment*(dimensions[dimension] - 1); 
    std::cout << "[";
    for (int i=start; i <= lastSubDimensionEntry; i+=increment){
        if (increment==1){
            if (printGrad) {
                std::cout << gradArrayPtr[i];
            }
            else {
                std::cout << dataArrayPtr[i];
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