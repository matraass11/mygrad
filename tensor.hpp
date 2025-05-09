#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "grad.hpp"
#include "helper.hpp"
// class Value;


class Tensor {
    const std::vector<int> dimensions;
    const int length;
    Value* valuesArray;

public:
    explicit Tensor(std::vector<int> dimensions, std::vector<double> dataVector) // pass a value creator function instead of a vector?
    : dimensions(dimensions), length(lengthOfTensorFromItsDimensionVector(dimensions))  {

        if (dataVector.size() != length){
            std::cout << "dataVector size doesn't correspond to the dimensions, exiting" << std::endl;
            exit(1);
        }

        valuesArray = new Value[length]; //delete[] ? 
                                         //is this a proper way to initialize my object?
        for (int i=0; i<length; i++){
            valuesArray[i] = Value(dataVector[i]);
        }
    }


    explicit Tensor(const Tensor& IndexedInto, int index) // constructor specifically for indexing
    : dimensions(std::vector<int>(IndexedInto.dimensions.begin() + 1, IndexedInto.dimensions.end())), 
    length(IndexedInto.length / IndexedInto.dimensions[0]), valuesArray(IndexedInto.valuesArray + index * length) {}
    
    
    Tensor operator[](int index) const {
        Tensor subTensor = Tensor(*this, index);
        return subTensor;
    }

    friend std::ostream& operator<<(std::ostream& outputStream, const Tensor& tensor);

    // Tensor matmul(const Tensor& other){
    //     if (this->dimensions.size() != 2 || other.dimensions.size() != 2){
    //         std::cout << "error: matrix multiplication only available for 2d tensors at the moment. exiting" << std::endl;
    //         exit(1);
    //     }
    //     if (this->dimensions[1] != other.dimensions[0]){
    //         std::cout << "error: matrix multiplication undefined for these dimensions. exiting" << std::endl;
    //         exit(1);
    //     }

    //     Tensor product({this->dimensions[0], other.dimensions[1]}, )

    // }
};

std::ostream& operator<<(std::ostream& outputStream, const Tensor& tensor){
    if (tensor.dimensions.size() == 0){
        outputStream << *tensor.valuesArray;
    }
    else {
        outputStream << "[";
        for (int i=0; i < tensor.dimensions[0]; i++){
            outputStream << tensor[i];
            if (i<tensor.dimensions[0]-1){
                outputStream << ", ";
            }
        }
        outputStream << "]";
    }
    return outputStream;
}

