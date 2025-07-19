#include <memory>
#include <iostream>
#include "tensor.hpp"

Tensor::Tensor( const std::vector<size_t>& dimensions ) : 
    length(lengthFromDimensions(dimensions)),
    dimensions(dimensions),
    strides(stridesFromDimensions(dimensions)),
    data(std::make_unique<dtype[]>(length)),
    grads(std::make_unique<dtype[]>(length)) {}

Tensor::Tensor( const std::vector<dtype>& dataVector,
                const std::vector<size_t>& dimensions ) : 
                
    Tensor(dimensions) 

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
    if (dimensions.size() > 2) {
        printRecursively(0, 0, true, false);
    }
    else {
        printRecursively(0, 0, false, false);
    }
    std::cout << "\n";
}

void Tensor::printGrad() const {
    if (dimensions.size() > 2) {
        printRecursively(0, 0, true, true);
    }
    else {
        printRecursively(0, 0, false, true);
    }
    std::cout << "\n";
}

int Tensor::lengthFromDimensions(const std::vector<size_t>& dimensions) const {
    size_t length = 1;
    for (int i=0; i<dimensions.size(); i++){
        length *= dimensions[i];
    }
    return length;
}

std::vector<int> Tensor::stridesFromDimensions(const std::vector<size_t>& dimensions) const {
    std::vector<int> strides(dimensions.size());
    int currentStride = 1;
    for (int i = strides.size()-1; i >= 0; i--) {
        strides[i] = currentStride;
        currentStride *= dimensions[i];
    }
    return strides;
}

int Tensor::indicesToLocationIn1dArray(const std::vector<int>& indices) const {
    if (indices.size() != dimensions.size()){
        std::cerr << "invalid size of indices to convert to location in 1d, exiting\n";
        exit(1);
    }  
    int locationOfElementInDataArray = 0;
    int dimensionality = dimensions.size();
    for (int i=0; i < dimensionality; i++){
        locationOfElementInDataArray += indices[i] * strides[i];
    }
    if (locationOfElementInDataArray > length){
        std::cerr << "indices to be converted to location in 1d are out of bound, exiting\n";
        exit(1);
    }
    return locationOfElementInDataArray;
}

void Tensor::printRecursively(int start, int dimension, bool printByBlocks, bool printGrad) const {
    int increment = strides[dimension];
    size_t thisDimShape = dimensions[dimension];
    std::cout << "[";
    if ( printByBlocks ) {
        std::cout << "  ";
    }
    for (int i=0; i < thisDimShape; i++) {
        if (dimension == dimensions.size() - 1){
            if (printGrad) {
                std::cout << grads[start+i];
            }
            else {
                std::cout << data[start+i];
            }
        }
        else {
            printRecursively(start + i*increment, dimension+1, false, printGrad);
        }
        if (i < thisDimShape-1) {
            std::cout << ", ";
            if (printByBlocks) {
                std::cout << "\n\n   ";
            }
        }
    }
    if (printByBlocks) {
        std::cout << "  ";
    }
    std::cout << "]";
}

Tensor Tensor::operator+( const Tensor& other ) const {
    if (dimensions != other.dimensions) {
        std::cerr << "dimensions must match for unary operations, exiting\n";
        exit(1);
    }
    Tensor output(dimensions);
    for (int i = 0; i < length; i++) {
        output.data[i] = data[i] + other.data[i];
    }
    return output;
}

Tensor Tensor::operator-( const Tensor& other ) const {
    if (dimensions != other.dimensions) {
        std::cerr << "dimensions must match for unary operations, exiting\n";
        exit(1);
    }
    Tensor output(dimensions);
    for (int i = 0; i < length; i++) {
        output.data[i] = data[i] - other.data[i];
    }
    return output;
}

Tensor Tensor::substractColumn( const Tensor& column ) const {

    if (dimensions.size() != 2 or column.dimensions.size() != 2 
        or column.dimensions[0] != dimensions[0] or column.dimensions[1] != 1) {
            std::cerr << "wrond dims for substractColumn, exiting\n";
            exit(1);
    }

    Tensor output(dimensions);
    for (int i = 0; i < length; i++) {
        output.data[i] = data[i] - column.data[i/dimensions[1]];
    }
    return output;
}

Tensor Tensor::addColumn( const Tensor& column ) const {

    if (dimensions.size() != 2 or column.dimensions.size() != 2 
        or column.dimensions[0] != dimensions[0] or column.dimensions[1] != 1) {
            std::cerr << "wrond dims for substractColumn, exiting\n";
            exit(1);
    }

    Tensor output(dimensions);
    for (int i = 0; i < length; i++) {
        output.data[i] = data[i] + column.data[i/dimensions[1]];
    }
    return output;
}

void Tensor::checkValidityOfDimension( int dimension ) const {
    if ( dimension >= dimensions.size() or dimension < 0) {
        std::cerr << "dimension out of range, exiting\n";
        exit(1);
    }
}


Tensor Tensor::exp() const {
    Tensor expedTensor( dimensions );
    for (int i = 0; i < length; i++) {
        expedTensor.data[i] = std::exp(data[i]);
    }
    return expedTensor;
}

Tensor Tensor::log() const {
    Tensor loggedTensor( dimensions );
    for (int i = 0; i < length; i++) {
        loggedTensor.data[i] = std::log(data[i]);
    }
    return loggedTensor;
}

Tensor Tensor::max(int maxAlongDimension) const {
    if ( maxAlongDimension < 0 ) {
        maxAlongDimension = dimensions.size() + maxAlongDimension;
    }
    checkValidityOfDimension( maxAlongDimension );

    std::vector<size_t> dimensionsOfMaxTensor = dimensions;
    dimensionsOfMaxTensor[maxAlongDimension] = 1;
    Tensor maxTensor( std::vector<dtype>(lengthFromDimensions(dimensionsOfMaxTensor), INT_MIN), dimensionsOfMaxTensor );

    size_t offsetBetweenElementsOfThisDim = strides[maxAlongDimension];
    size_t totalSizeOfThisDim = dimensions[maxAlongDimension] * offsetBetweenElementsOfThisDim;
    int locInMax;
    for (int i = 0; i < length; i++) {
        locInMax = i % offsetBetweenElementsOfThisDim + (i / totalSizeOfThisDim) * offsetBetweenElementsOfThisDim;
        if (this->data[i] > maxTensor.data[locInMax]) {
            maxTensor.data[locInMax] = this->data[i];
        }
    }

    return maxTensor;
}   

Tensor Tensor::sum(int sumAlongDimension) const {
    if ( sumAlongDimension < 0 ) {
        sumAlongDimension = dimensions.size() + sumAlongDimension;
    }
    checkValidityOfDimension( sumAlongDimension );

    std::vector<size_t> dimensionsOfSumTensor = dimensions;
    dimensionsOfSumTensor[sumAlongDimension] = 1;
    Tensor sumTensor( dimensionsOfSumTensor );

    size_t offsetBetweenElementsOfThisDim = strides[sumAlongDimension];
    size_t totalSizeOfThisDim = dimensions[sumAlongDimension] * offsetBetweenElementsOfThisDim;
    int locInSum;
    for (int i = 0; i < length; i++) {
        locInSum = i % offsetBetweenElementsOfThisDim + (i / totalSizeOfThisDim) * offsetBetweenElementsOfThisDim;
        sumTensor.data[locInSum] += this->data[i];
    }

    return sumTensor;
}   