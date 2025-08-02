#include <memory>
#include <iostream>
#include <format>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "mygrad/tensor.hpp"

#include "mygrad/helper.hpp"

namespace mygrad {

Tensor::Tensor( const std::vector<size_t>& dimensions )
try:
    length(lengthFromDimensions(dimensions)),
    dimensions(dimensions),
    strides(stridesFromDimensions(dimensions)),
    data(std::make_unique<dtype[]>(length)),
    grads(std::make_unique<dtype[]>(length)) {}
catch(const std::bad_alloc& e) {
    std::cerr << "check if the dimensions provided for tensor are not too big. dimensions: \n" << dimensions;
    throw;
}


Tensor::Tensor( const std::vector<dtype>& dataVector,
                const std::vector<size_t>& dimensions ) : 
                
    Tensor(dimensions) 

    {
        std::copy(dataVector.begin(), dataVector.end(), data.get());

        if (dataVector.size() != length) {
            std::cerr << "tensor initialized with vector of wrong size. exiting\n"; 
                std::cerr << dataVector.size() << " != " << length << "\n";
                exit(1);
            }
    }

Tensor Tensor::zeros( const std::vector<size_t>& dimensions ) {
    return Tensor(dimensions);
}


void Tensor::print() const {
    if (dimensions.size() > 1) {
        printRecursively(0, 0, true, false);
    }
    else {
        printRecursively(0, 0, false, false);
    }
    std::cout << "\n\n";
}

void Tensor::printGrad() const {
    if (dimensions.size() > 2) {
        printRecursively(0, 0, true, true);
    }
    else {
        printRecursively(0, 0, false, true);
    }
    std::cout << "\n\n";
}

int Tensor::lengthFromDimensions(const std::vector<size_t>& dimensions) {
    size_t length = 1;
    for (int i=0; i<dimensions.size(); i++){
        length *= dimensions[i];
    }
    return length;
}

std::vector<int> Tensor::stridesFromDimensions(const std::vector<size_t>& dimensions) {
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

std::vector<int> Tensor::locationIn1dArrayToIndices(int location) const {
    if (location < 0 or location >= length) {
        std::cerr << location << " - location\n";
        throw std::runtime_error("location to be converted to indices is out of range");
    }

    std::vector<int> indices(dimensions.size());
    for (int i = 0; i < indices.size() ; i++) {
        indices[i] = location / strides[i];
        location -= indices[i] * strides[i];
    }
    return indices;
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
                std::cout << std::format("{:#.4F}", grads[start+i]);
            }
            else {
                std::cout << std::format("{:#.4F}", data[start+i]);
            }
        }
        else {
            if (dimensions.size() - dimension == 2) {
                printByBlocks = false;
            }
            printRecursively(start + i*increment, dimension+1, printByBlocks, printGrad);
        }
        if (i < thisDimShape-1) {
            std::cout << ", ";
            if (printByBlocks) {
                std::cout << "\n   ";
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
    Tensor output(zeros(dimensions));
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
    Tensor output(zeros(dimensions));
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

    Tensor output(zeros(dimensions));
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

    Tensor output(zeros(dimensions));
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
    Tensor expedTensor( zeros(dimensions) );
    for (int i = 0; i < length; i++) {
        expedTensor.data[i] = std::exp(data[i]);
    }
    return expedTensor;
}

Tensor Tensor::log() const {
    Tensor loggedTensor( zeros(dimensions) );
    for (int i = 0; i < length; i++) {
        loggedTensor.data[i] = std::log(data[i]);
    }
    return loggedTensor;
}

Tensor Tensor::max(int dim) const {
    if ( dim < 0 ) {
        dim = dimensions.size() + dim;
    }
    checkValidityOfDimension( dim );

    std::vector<size_t> dimsOfMaxTensor = dimensions;
    dimsOfMaxTensor[dim] = 1;
    Tensor maxTensor( std::vector<dtype>(lengthFromDimensions(dimsOfMaxTensor), INT_MIN), dimsOfMaxTensor );

    size_t offsetBetweenElementsOfThisDim = strides[dim];
    size_t totalSizeOfThisDim = dimensions[dim] * offsetBetweenElementsOfThisDim;
    int locInMax;
    for (int i = 0; i < length; i++) {
        locInMax = i % offsetBetweenElementsOfThisDim + (i / totalSizeOfThisDim) * offsetBetweenElementsOfThisDim;
        if (this->data[i] > maxTensor.data[locInMax]) {
            maxTensor.data[locInMax] = this->data[i];
        }
    }

    return maxTensor;
}   

Tensor Tensor::sum(int dim) const {
    if ( dim < 0 ) {
        dim = dimensions.size() + dim;
    }
    checkValidityOfDimension( dim );

    std::vector<size_t> dimsOfSumTensor = dimensions;
    dimsOfSumTensor[dim] = 1;
    Tensor sumTensor( zeros(dimsOfSumTensor) );

    size_t offsetBetweenElementsOfThisDim = strides[dim];
    size_t totalSizeOfThisDim = dimensions[dim] * offsetBetweenElementsOfThisDim;
    int locInSum;
    for (int i = 0; i < length; i++) {
        locInSum = i % offsetBetweenElementsOfThisDim + (i / totalSizeOfThisDim) * offsetBetweenElementsOfThisDim;
        sumTensor.data[locInSum] += this->data[i];
    }

    return sumTensor;
}

Tensor Tensor::argmax(int dim) const {
    if ( dim < 0 ) {
        dim = dimensions.size() + dim;
    }
    checkValidityOfDimension( dim );

    std::vector<size_t> dimsOfArgmaxTensor = dimensions;
    dimsOfArgmaxTensor[dim] = 1;
    Tensor argmaxTensor( std::vector<dtype>(length/dimensions[dim], INT_MIN), dimsOfArgmaxTensor );

    size_t offsetBetweenElementsOfThisDim = strides[dim];
    size_t totalSizeOfThisDim = dimensions[dim] * offsetBetweenElementsOfThisDim;
    int locInArgmax;
    for (int i = 0; i < length; i++) {
        locInArgmax = i % offsetBetweenElementsOfThisDim + (i / totalSizeOfThisDim) * offsetBetweenElementsOfThisDim;
        std::vector<int> currentMaxIndices = locationIn1dArrayToIndices(i);
        currentMaxIndices[dim] = argmaxTensor.data[locInArgmax];
        if (argmaxTensor.data[locInArgmax] == INT_MIN or this->data[i] > this->at(currentMaxIndices)) {
            argmaxTensor.data[locInArgmax] = locationIn1dArrayToIndices(i)[dim];
        }
    }

    return argmaxTensor;
}

dtype Tensor::mean() const {
    dtype sum = 0;
    for (int i = 0; i < length; i++) {
        sum += data[i];
    }
    return sum / length;
}

dtype Tensor::std() const {
    dtype sumOfSquareDifferences = 0;
    dtype mean = this->mean();
    for (int i = 0; i < length; i++) {
        sumOfSquareDifferences += (data[i] - mean)*(data[i] - mean);
    }

    return std::sqrt(sumOfSquareDifferences/(length-1));
}

} // namespace mygrad