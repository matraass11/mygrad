#include <memory>
#include <iostream>
#include "tensor.hpp"

Tensor::Tensor( const std::vector<size_t>& dimensionsVector ) : 
    length(lengthFromDimensionsVector(dimensionsVector)),
    data(std::make_unique<dtype[]>(length)),
    grads(std::make_unique<dtype[]>(length)),
    dimensions(dimensionsVector) {}

Tensor::Tensor( const std::vector<dtype>& dataVector,
                const std::vector<size_t>& dimensionsVector ) : 
                
    Tensor(dimensionsVector) 

    {
        std::copy(dataVector.begin(), dataVector.end(), data.get());

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

int Tensor::indicesToLocationIn1dArray(const std::vector<int>& indices) const {
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


Tensor Tensor::exp() {
    Tensor expedTensor(this->dimensions);
    for (int i = 0; i < length; i++) {
        expedTensor.data[i] = std::exp(this->data[i]);
    }
    return expedTensor;
}

Tensor Tensor::max(int maxAlongDimension) {
    std::vector<size_t> dimensionsOfMaxTensor = dimensions;
    dimensionsOfMaxTensor[maxAlongDimension] = 1;
    Tensor maxTensor( std::vector<dtype>(lengthFromDimensionsVector(dimensionsOfMaxTensor), INT_MIN), dimensionsOfMaxTensor );
    // keeping the previous line because i must resist premature optimization

    size_t offsetBetweenElementsOfThisDim = 1;
    for (int i = maxAlongDimension+1; i < dimensions.size(); i++) {
        offsetBetweenElementsOfThisDim *= dimensions[i];
    }
    size_t totalSizeOfThisDim = dimensions[maxAlongDimension] * offsetBetweenElementsOfThisDim;

    std::cout << "smaller offset: " << offsetBetweenElementsOfThisDim << ", bigger offset: " << totalSizeOfThisDim << "\n";

    int locInMax;
    for (int i = 0; i < length; i++) {
        locInMax = i % offsetBetweenElementsOfThisDim + (i / totalSizeOfThisDim) * offsetBetweenElementsOfThisDim;
        if (this->data[i] > maxTensor.data[locInMax]) {
            maxTensor.data[locInMax] = this->data[i];
        }
    }

    return maxTensor;
}   