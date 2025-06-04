#include <memory>
#include <iostream>
#include "tensor.hpp"

Tensor::Tensor( double* dataArrayPtr, double* gradArrayPtr,
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

void Tensor::setAllGradsTo(double newGrad) {
    for (int i=0; i < length; i++){
        gradArrayPtr[i] = newGrad;
    }
}

size_t Tensor::lengthFromDimensionsVector(const std::vector<int>& dimensionsVector) const {
    size_t length = 1;
    for (int i=0; i<dimensions.size(); i++){
        length *= dimensions[i];
    }
    return length;
}

inline double Tensor::at(const std::vector<int>& indices) const {
    return dataArrayPtr[indicesToLocationIn1dArray(indices)];
}

inline double Tensor::gradAt(const std::vector<int>& indices) const {
    return gradArrayPtr[indicesToLocationIn1dArray(indices)];
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

void Tensor::incrementGradAt(const std::vector<int>& indices, double increment){
    this->gradArrayPtr[indicesToLocationIn1dArray(indices)] += increment;
}

TensorMatMulProduct::TensorMatMulProduct(
    double* dataArrayPtr, double* gradArrayPtr,
    Tensor& leftParent, Tensor& rightParent) :
        Tensor(dataArrayPtr, gradArrayPtr, dimensionsFromParents(leftParent, rightParent)), leftParent(&leftParent), rightParent(&rightParent) {}


std::vector<int> TensorMatMulProduct::dimensionsFromParents(const Tensor& leftParent, const Tensor& rightParent) const {
    // JUST 2D FOR NOW
    if (leftParent.dimensions[1] != rightParent.dimensions[0]){
        std::cerr << "wrong dims, cant matMul. exiting\n";
        exit(1);
    }
    return {leftParent.dimensions[0], rightParent.dimensions[1]};
}


void TensorMatMulProduct::assignNewParents(Tensor* leftHandSide, Tensor* rightHandSide) {
    leftParent = leftHandSide, rightParent = rightHandSide;
}

void TensorMatMulProduct::matMul2dIntoSelf(Tensor& leftTensor, Tensor& rightTensor){

    TensorMatMulProduct& productTensor = *this;

    if (leftTensor.dimensions.size() != 2 or rightTensor.dimensions.size() != 2){
        std::cerr << "only 2d for now, exiting\n";
        exit(1);
    }
    if (leftTensor.dimensions[1] != rightTensor.dimensions[0] or
        leftTensor.dimensions[0] != productTensor.dimensions[0] or
        rightTensor.dimensions[1] != productTensor.dimensions[1]) {
        std::cerr << "wrong dims, exiting\n";
        exit(1);
    } 

    for (int row=0; row < dimensions[0]; row++) {
        for (int column=0; column < dimensions[1]; column++) {
            uint locationOfElementInProductTensor = indicesToLocationIn1dArray({row, column});
            dataArrayPtr[locationOfElementInProductTensor] = 0;
            for (int dotProductIterator=0; dotProductIterator < leftTensor.dimensions[1]; dotProductIterator++) {
                dataArrayPtr[locationOfElementInProductTensor] +=
                leftTensor.at({row, dotProductIterator}) * rightTensor.at({dotProductIterator, column});
            }
        }
    }

    productTensor.assignNewParents(&leftTensor, &rightTensor);
}

void TensorMatMulProduct::backwardFurther() const {
    for (int row=0; row < dimensions[0]; row++) {
        for (int column=0; column < dimensions[1]; column++) {
            uint locationOfElementInProductTensor = indicesToLocationIn1dArray({row, column});
            
            for (int dotProductIterator=0; dotProductIterator < leftParent->dimensions[1]; dotProductIterator++) {
                leftParent->incrementGradAt({row, dotProductIterator}, 
                    rightParent->at({dotProductIterator, column}) * gradArrayPtr[locationOfElementInProductTensor]); 

                rightParent->incrementGradAt({dotProductIterator, column}, 
                    leftParent->at({row, dotProductIterator}) * gradArrayPtr[locationOfElementInProductTensor]); 
            }
        }
    }
}